// host_serve: keeps the model loaded on the card and answers one generation request per line (for a chat).
//
//   host_serve --xclbin llm_top.xclbin --split PREFIX --ports 12 [--no-resume] [--device N]
//
// It loads the xclbin and the per-port weight files (sw/split_ports.py) into the HBM ONCE, then reads lines from stdin:
//   GEN <steps> <tok0> <tok1> ...   run the kernel on the prompt tokens and generate <steps> tokens
//   QUIT                            exit
// and answers on stdout, one line each (stdout is flushed after every line):
//   READY <vocab> <context> <load seconds>        once, after the weights are loaded
//   TOKENS <t0> <t1> ...                          the generated tokens (not the prompt)
//   TIME <seconds> <positions computed> <positions/s> <start_pos>   kernel time of the request
//   ERR <text>                                    the request was refused (nothing was run, or the kernel refused it)
// The K/V cache is kept between requests (kernel argument start_pos, needs the xclbin built 2026-11-18 or later): the server remembers which
// tokens' K/V are in the state buffer, and a request only computes the positions after the longest common prefix of the new prompt and those
// tokens, so a chat client sends the whole conversation each turn and the card only reads the new part. --no-resume always starts at 0 (for
// an xclbin without start_pos: then the argument is not set either). The state buffer is never cleared: the kernel only reads K/V entries
// of positions it has written itself.
// Same buffers and arguments as host_xrt.cpp; never flashes or resets the card.
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
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

int main(int argc, char **argv) {
    std::string xclbin_path, split_prefix;
    int ports = 1, dev_index = 0;
    bool resume = true;
    int stop0 = -1, stop1 = -1;   // the kernel ends a reply early when it generates one of these token ids (-1 = none)
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> std::string { if (i + 1 >= argc) { fprintf(stderr, "%s needs a value\n", a.c_str()); exit(2); } return argv[++i]; };
        if (a == "--xclbin") xclbin_path = next();
        else if (a == "--split") split_prefix = next();
        else if (a == "--ports") ports = atoi(next().c_str());
        else if (a == "--device") dev_index = atoi(next().c_str());
        else if (a == "--no-resume") resume = false;
        else if (a == "--stop") { std::string v = next(); size_t c = v.find(','); stop0 = atoi(v.substr(0, c).c_str()); if (c != std::string::npos) stop1 = atoi(v.substr(c + 1).c_str()); }
        else { fprintf(stderr, "unknown option %s\n", a.c_str()); return 2; }
    }
    if (xclbin_path.empty() || split_prefix.empty() || ports < 1 || ports > 27) {
        fprintf(stderr, "usage: %s --xclbin f --split PREFIX [--ports 1-27] [--device N]\n", argv[0]);
        return 2;
    }
    try {
        auto t_load = std::chrono::steady_clock::now();
        const std::string p0 = split_prefix + ".p0";
        std::ifstream mf(p0, std::ios::binary | std::ios::ate);
        if (!mf) { perror(p0.c_str()); return 1; }
        const size_t wbytes = (size_t)mf.tellg();
        mf.seekg(0);
        std::vector<char> header(4096);
        mf.read(header.data(), (std::streamsize)std::min(header.size(), wbytes));
        llm_config cfg;
        if (llm_read_config(header.data(), &cfg)) { fprintf(stderr, "bad weight file (llm_read_config failed)\n"); return 1; }
        const int kvd = cfg.dim * cfg.n_kv_heads / cfg.n_heads;
        const size_t nstate = llm_state_floats(cfg.vocab_size, cfg.n_layers, cfg.seq_len, kvd);
        const size_t ntok = llm_token_ints(cfg.seq_len);

        xrt::device device(dev_index);
        xrt::uuid uuid = device.load_xclbin(xclbin_path);
        xrt::kernel krnl(device, uuid, "llm_top");
        xrt::bo w_bo(device, wbytes, krnl.group_id(0));
        xrt::bo s_bo(device, nstate * sizeof(float), krnl.group_id(1));
        xrt::bo t_bo(device, ntok * sizeof(int), krnl.group_id(2));
        mf.seekg(0);
        mf.read(w_bo.map<char *>(), (std::streamsize)wbytes);
        if (!mf) { fprintf(stderr, "short read of %s\n", p0.c_str()); return 1; }
        std::memset(s_bo.map<char *>(), 0, nstate * sizeof(float));
        int *tk = t_bo.map<int *>();
        for (size_t i = 0; i < ntok; i++) tk[i] = 0;
        w_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        s_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        t_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        xrt::run run(krnl);
        run.set_arg(0, w_bo);
        run.set_arg(1, s_bo);
        run.set_arg(2, t_bo);
        std::vector<xrt::bo> extra;
        for (int p = 1; p < ports; p++) {
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
        }
        std::vector<int> cached;   // the tokens whose K/V entries are in the state buffer: positions 0 .. cached.size()-1
        if (resume) { run.set_arg(4 + ports, 0); run.set_arg(5 + ports, stop0); run.set_arg(6 + ports, stop1); }   // start_pos, stop0, stop1 follow the weight ports
        const double load_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_load).count();
        printf("READY %d %d %.1f\n", cfg.vocab_size, cfg.seq_len, load_s);
        fflush(stdout);

        std::string line;
        while (std::getline(std::cin, line)) {
            std::istringstream in(line);
            std::string cmd;
            in >> cmd;
            if (cmd == "QUIT") break;
            if (cmd != "GEN") { printf("ERR unknown command\n"); fflush(stdout); continue; }
            int steps = -1;
            in >> steps;
            std::vector<int> prompt;
            int x;
            while (in >> x) prompt.push_back(x);
            const int np = (int)prompt.size(), total = np + steps;
            if (steps < 0 || np < 1) { printf("ERR need GEN <steps> <tokens...>\n"); fflush(stdout); continue; }
            if (total - 1 > cfg.seq_len) { printf("ERR prompt + steps (%d) is longer than the context (%d)\n", total, cfg.seq_len); fflush(stdout); continue; }
            for (int i = 0; i < np; i++) tk[i] = prompt[i];
            for (int i = np; i < total; i++) tk[i] = -1;   // sentinel: the kernel overwrites these
            t_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            int start = 0;   // longest common prefix of the cached tokens and the new prompt (the kernel needs start <= np-1)
            if (resume) {
                while (start < (int)cached.size() && start < np && cached[start] == prompt[start]) start++;
                if (start > np - 1) start = np - 1;
                run.set_arg(4 + ports, start);
            }
            run.set_arg(3, np);
            run.set_arg(4, steps);
            auto t0 = std::chrono::steady_clock::now();
            run.start();
            run.wait();
            const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            t_bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
            bool refused = tk[np] < 0;   // the first generated token is always written unless the kernel refused the job
            int n = np;                  // tokens now valid: the prompt plus what was generated (shorter than total when a stop token ended the reply)
            while (n < total && tk[n] >= 0) n++;
            if (refused) { cached.clear(); printf("ERR the kernel refused the job (error code -1 .. -9, see llm_kernel.cpp)\n"); fflush(stdout); continue; }
            printf("TOKENS");
            for (int i = np; i < n; i++) printf(" %d", tk[i]);
            // positions 0 .. n-2 now have K/V in the state buffer (the last token, a stop token or the last generated one, has none yet)
            cached.assign(tk, tk + n - 1);
            printf("\nTIME %.3f %d %.2f %d\n", secs, n - 1 - start, (n - 1 - start) / secs, start);
            fflush(stdout);
        }
    } catch (const std::exception &e) {
        printf("ERR XRT error: %s\n", e.what());
        fflush(stdout);
        return 1;
    }
    return 0;
}
