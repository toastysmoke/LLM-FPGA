// XRT host for llm_top on the Alveo U50.
//
//   host_xrt --xclbin llm_top.xclbin --model model.flat --steps N --prompt "1 5 9" [--logits out.bin] [--expect "1 5 9 ..."] [--ports P (1-27)] [--share | --split PREFIX]
//
// --split PREFIX: the per-port layout made by sw/split_ports.py. Port p gets the file PREFIX.p<p> (port 0 also holds the header, so --model is not
// needed), in the memory bank of its own argument: every port reads its own part of the HBM and no two ports share a channel.
//
// Same job as tb_kernel.cpp: the weights go to the card's HBM, the prompt goes in the token buffer, the kernel runs once for the
// whole generation, and the tokens (and the last position's logits) are read back. Prints the token ids on one line, like main_pc.
//
// Buffers are created in the memory bank each kernel argument is linked to (hls_u50/u50_link_llama1b_p27.cfg): kernel.group_id(argument index).
// Argument order of llm_top: 0 weights, 1 state, 2 tokens, 3 n_prompt, 4 n_steps, then, when built with LLM_PORTS > 1, weights1..weights26 and the start_pos and stop-token arguments after n_steps (see LLM_PORT_ARGS in llm_kernel.h).
//
// The kernel's return code (0 ok, -1..-8 errors, see llm_kernel.cpp) is an AXI-Lite register that xrt::kernel does not hand back. So the
// generated positions of the token buffer are pre-filled with -1: any -1 left after the run means the kernel refused the job and wrote
// nothing (bad file, model over the kernel limits, bad n_prompt/n_steps).
#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"
extern "C" {
#include "llm.h"
}
#include "llm_kernel.h"
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

static std::vector<int> parse_ints(const std::string &s) {
    std::vector<int> v;
    std::istringstream in(s);
    int x;
    while (in >> x) v.push_back(x);
    return v;
}

int main(int argc, char **argv) {
    std::string xclbin_path, model_path, logits_path, prompt_s, expect_s;
    int steps = -1, ports = 1, dev_index = 0;
    std::string split_prefix;   // --split PREFIX: one file per port (PREFIX.p0, PREFIX.p1, ...)
    bool share = false;   // --share: every weight port reads the SAME buffer (one copy of the weights; needed when the model is too big for one copy per port)
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> std::string { if (i + 1 >= argc) { fprintf(stderr, "%s needs a value\n", a.c_str()); exit(2); } return argv[++i]; };
        if (a == "--xclbin") xclbin_path = next();
        else if (a == "--model") model_path = next();
        else if (a == "--steps") steps = atoi(next().c_str());
        else if (a == "--prompt") prompt_s = next();
        else if (a == "--logits") logits_path = next();
        else if (a == "--expect") expect_s = next();
        else if (a == "--ports") ports = atoi(next().c_str());
        else if (a == "--share") share = true;
        else if (a == "--split") split_prefix = next();
        else if (a == "--device") dev_index = atoi(next().c_str());
        else { fprintf(stderr, "unknown option %s\n", a.c_str()); return 2; }
    }
    std::vector<int> prompt = parse_ints(prompt_s), expect = parse_ints(expect_s);
    if (!split_prefix.empty()) model_path = split_prefix + ".p0";
    if (xclbin_path.empty() || model_path.empty() || steps < 0 || prompt.empty() || ports < 1 || ports > 27) {
        fprintf(stderr, "usage: %s --xclbin f --model f.flat --steps N --prompt \"t0 t1 ...\" [--logits f] [--expect \"...\"] [--ports 1-27] [--share | --split PREFIX] [--device N]\n", argv[0]);
        return 2;
    }

    // the weight file: only its header is parsed here, the rest goes straight into the card buffer
    std::ifstream mf(model_path, std::ios::binary | std::ios::ate);
    if (!mf) { perror(model_path.c_str()); return 1; }
    const size_t wbytes = (size_t)mf.tellg();
    mf.seekg(0);
    std::vector<char> header(4096);
    mf.read(header.data(), (std::streamsize)std::min(header.size(), wbytes));
    llm_config cfg;
    if (llm_read_config(header.data(), &cfg)) { fprintf(stderr, "bad weight file (llm_read_config failed)\n"); return 1; }
    const int kvd = cfg.dim * cfg.n_kv_heads / cfg.n_heads;
    const size_t nstate = llm_state_floats(cfg.vocab_size, cfg.n_layers, cfg.seq_len, kvd);
    const size_t ntok = llm_token_ints(cfg.seq_len);
    const int np = (int)prompt.size(), total = np + steps;
    if (total - 1 > cfg.seq_len) { fprintf(stderr, "prompt + steps (%d) is longer than the model's context (%d)\n", total, cfg.seq_len); return 1; }
    printf("model: %zu bytes, dim %d, layers %d, vocab %d, context %d; state %zu floats; %d prompt + %d steps\n",
           wbytes, cfg.dim, cfg.n_layers, cfg.vocab_size, cfg.seq_len, nstate, np, steps);

    try {
        xrt::device device(dev_index);
        xrt::uuid uuid = device.load_xclbin(xclbin_path);   // the shell stays; only the kernel (user partition) is loaded
        xrt::kernel krnl(device, uuid, "llm_top");

        xrt::bo w_bo(device, wbytes, krnl.group_id(0));
        xrt::bo s_bo(device, nstate * sizeof(float), krnl.group_id(1));
        xrt::bo t_bo(device, ntok * sizeof(int), krnl.group_id(2));

        mf.seekg(0);
        mf.read(w_bo.map<char *>(), (std::streamsize)wbytes);
        if (!mf) { fprintf(stderr, "short read of %s\n", model_path.c_str()); return 1; }
        std::memset(s_bo.map<char *>(), 0, nstate * sizeof(float));
        int *tk = t_bo.map<int *>();
        for (size_t i = 0; i < ntok; i++) tk[i] = 0;
        for (int i = 0; i < np; i++) tk[i] = prompt[i];
        for (int i = np; i < total; i++) tk[i] = -1;            // sentinel: the kernel overwrites these when it runs

        w_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        s_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        t_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        xrt::run run(krnl);
        run.set_arg(0, w_bo);
        run.set_arg(1, s_bo);
        run.set_arg(2, t_bo);
        run.set_arg(3, np);
        run.set_arg(4, steps);
        // extra weight ports read the same file; each is linked to its own bank, so give each its own copy of the weights
        std::vector<xrt::bo> extra;
        for (int p = 1; p < ports; p++) {
            if (!split_prefix.empty()) {   // this port's own file, in the bank of its own argument
                std::string fn = split_prefix + ".p" + std::to_string(p);
                std::ifstream pf(fn, std::ios::binary | std::ios::ate);
                if (!pf) { perror(fn.c_str()); return 1; }
                const size_t pbytes = (size_t)pf.tellg();
                pf.seekg(0);
                extra.emplace_back(device, pbytes, krnl.group_id(4 + p));
                pf.read(extra.back().map<char *>(), (std::streamsize)pbytes);
                if (!pf) { fprintf(stderr, "short read of %s\n", fn.c_str()); return 1; }
                extra.back().sync(XCL_BO_SYNC_BO_TO_DEVICE);
                run.set_arg(4 + p, extra.back());
                continue;
            }
            if (share) { run.set_arg(4 + p, w_bo); continue; }   // one copy: all ports are linked to the same HBM range (see the link cfg)
            extra.emplace_back(device, wbytes, krnl.group_id(4 + p));
            std::memcpy(extra.back().map<char *>(), w_bo.map<char *>(), wbytes);
            extra.back().sync(XCL_BO_SYNC_BO_TO_DEVICE);
            run.set_arg(4 + p, extra.back());
        }
        run.set_arg(4 + ports, 0);    // start_pos: a fresh run (xclbin built 2026-11-18 or later)
        run.set_arg(5 + ports, -1);   // stop0: none
        run.set_arg(6 + ports, -1);   // stop1: none

        auto t0 = std::chrono::steady_clock::now();
        run.start();
        run.wait();
        double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

        t_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        s_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

        bool refused = false;
        for (int i = np; i < total; i++) if (tk[i] < 0) refused = true;
        if (refused) { fprintf(stderr, "kernel did not write the generated tokens: it returned an error code (see llm_kernel.cpp, -1 .. -8)\n"); return 1; }

        for (int i = 0; i < total; i++) printf("%d%c", tk[i], i + 1 < total ? ' ' : '\n');
        printf("kernel time %.3f s for %d positions (%.2f positions/s)\n", secs, total - 1, (total - 1) / secs);

        if (!logits_path.empty()) {     // the kernel leaves only the last position's logits in the state area
            std::ofstream lf(logits_path, std::ios::binary);
            lf.write(s_bo.map<char *>(), (std::streamsize)(cfg.vocab_size * sizeof(float)));
        }
        if (!expect.empty()) {
            if ((int)expect.size() != total) { fprintf(stderr, "--expect has %zu tokens, the run has %d\n", expect.size(), total); return 1; }
            for (int i = 0; i < total; i++)
                if (tk[i] != expect[i]) { fprintf(stderr, "MISMATCH at position %d: card %d, expected %d\n", i, tk[i], expect[i]); return 1; }
            printf("tokens match the expected sequence\n");
        }
    } catch (const std::exception &e) {
        fprintf(stderr, "XRT error: %s\n", e.what());
        return 1;
    }
    return 0;
}
