// Testbench / "host program" for llm_top.   tb_kernel model.flat steps logits.bin tok0 [tok1 ...]
//
// Plays the role of the host: weights go in "external memory", the prompt goes in the token buffer, the kernel is
// started, and the tokens and logits are read back. Output format matches main_pc (token ids on one line, every
// step's logits in logits.bin).
//
// The kernel leaves only the last position's logits in `state`, so to record every step this calls it once per
// position (prompt = first p+1 tokens, 1 step), then runs the whole sequence in one call and requires the same
// tokens. The second part is how the hardware will really be used.
//
// tb_kernel split:PREFIX steps logits.bin tok0 ...   reads the per-port files PREFIX.p0 .. PREFIX.p<LLM_PORTS-1> (sw/split_ports.py):
// port p of the kernel gets file p. The header (and so the model config) comes from PREFIX.p0.
#include "llm_kernel.h"
extern "C" {
#include "llm.h"
#include "portable_alloc.h"
}
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#if LLM_PORTS == 1
#define PORTV(a)
#elif LLM_PORTS == 2
#define PORTV(a) , (a)[1]
#elif LLM_PORTS == 3
#define PORTV(a) , (a)[1], (a)[2]
#elif LLM_PORTS == 4
#define PORTV(a) , (a)[1], (a)[2], (a)[3]
#elif LLM_PORTS == 5
#define PORTV(a) , (a)[1], (a)[2], (a)[3], (a)[4]
#elif LLM_PORTS == 6
#define PORTV(a) , (a)[1], (a)[2], (a)[3], (a)[4], (a)[5]
#elif LLM_PORTS == 7
#define PORTV(a) , (a)[1], (a)[2], (a)[3], (a)[4], (a)[5], (a)[6]
#elif LLM_PORTS == 8
#define PORTV(a) , (a)[1], (a)[2], (a)[3], (a)[4], (a)[5], (a)[6], (a)[7]
#elif LLM_PORTS == 9
#define PORTV(a) , (a)[1], (a)[2], (a)[3], (a)[4], (a)[5], (a)[6], (a)[7], (a)[8]
#elif LLM_PORTS == 10
#define PORTV(a) , (a)[1], (a)[2], (a)[3], (a)[4], (a)[5], (a)[6], (a)[7], (a)[8], (a)[9]
#elif LLM_PORTS == 11
#define PORTV(a) , (a)[1], (a)[2], (a)[3], (a)[4], (a)[5], (a)[6], (a)[7], (a)[8], (a)[9], (a)[10]
#elif LLM_PORTS == 12
#define PORTV(a) , (a)[1], (a)[2], (a)[3], (a)[4], (a)[5], (a)[6], (a)[7], (a)[8], (a)[9], (a)[10], (a)[11]
#elif LLM_PORTS == 13
#define PORTV(a) , (a)[1], (a)[2], (a)[3], (a)[4], (a)[5], (a)[6], (a)[7], (a)[8], (a)[9], (a)[10], (a)[11], (a)[12]
#elif LLM_PORTS == 14
#define PORTV(a) , (a)[1], (a)[2], (a)[3], (a)[4], (a)[5], (a)[6], (a)[7], (a)[8], (a)[9], (a)[10], (a)[11], (a)[12], (a)[13]
#elif LLM_PORTS == 15
#define PORTV(a) , (a)[1], (a)[2], (a)[3], (a)[4], (a)[5], (a)[6], (a)[7], (a)[8], (a)[9], (a)[10], (a)[11], (a)[12], (a)[13], (a)[14]
#elif LLM_PORTS == 16
#define PORTV(a) , (a)[1], (a)[2], (a)[3], (a)[4], (a)[5], (a)[6], (a)[7], (a)[8], (a)[9], (a)[10], (a)[11], (a)[12], (a)[13], (a)[14], (a)[15]
#elif LLM_PORTS == 17
#define PORTV(a) , (a)[1], (a)[2], (a)[3], (a)[4], (a)[5], (a)[6], (a)[7], (a)[8], (a)[9], (a)[10], (a)[11], (a)[12], (a)[13], (a)[14], (a)[15], (a)[16]
#elif LLM_PORTS == 18
#define PORTV(a) , (a)[1], (a)[2], (a)[3], (a)[4], (a)[5], (a)[6], (a)[7], (a)[8], (a)[9], (a)[10], (a)[11], (a)[12], (a)[13], (a)[14], (a)[15], (a)[16], (a)[17]
#elif LLM_PORTS == 19
#define PORTV(a) , (a)[1], (a)[2], (a)[3], (a)[4], (a)[5], (a)[6], (a)[7], (a)[8], (a)[9], (a)[10], (a)[11], (a)[12], (a)[13], (a)[14], (a)[15], (a)[16], (a)[17], (a)[18]
#elif LLM_PORTS == 20
#define PORTV(a) , (a)[1], (a)[2], (a)[3], (a)[4], (a)[5], (a)[6], (a)[7], (a)[8], (a)[9], (a)[10], (a)[11], (a)[12], (a)[13], (a)[14], (a)[15], (a)[16], (a)[17], (a)[18], (a)[19]
#elif LLM_PORTS == 21
#define PORTV(a) , (a)[1], (a)[2], (a)[3], (a)[4], (a)[5], (a)[6], (a)[7], (a)[8], (a)[9], (a)[10], (a)[11], (a)[12], (a)[13], (a)[14], (a)[15], (a)[16], (a)[17], (a)[18], (a)[19], (a)[20]
#elif LLM_PORTS == 22
#define PORTV(a) , (a)[1], (a)[2], (a)[3], (a)[4], (a)[5], (a)[6], (a)[7], (a)[8], (a)[9], (a)[10], (a)[11], (a)[12], (a)[13], (a)[14], (a)[15], (a)[16], (a)[17], (a)[18], (a)[19], (a)[20], (a)[21]
#elif LLM_PORTS == 23
#define PORTV(a) , (a)[1], (a)[2], (a)[3], (a)[4], (a)[5], (a)[6], (a)[7], (a)[8], (a)[9], (a)[10], (a)[11], (a)[12], (a)[13], (a)[14], (a)[15], (a)[16], (a)[17], (a)[18], (a)[19], (a)[20], (a)[21], (a)[22]
#elif LLM_PORTS == 24
#define PORTV(a) , (a)[1], (a)[2], (a)[3], (a)[4], (a)[5], (a)[6], (a)[7], (a)[8], (a)[9], (a)[10], (a)[11], (a)[12], (a)[13], (a)[14], (a)[15], (a)[16], (a)[17], (a)[18], (a)[19], (a)[20], (a)[21], (a)[22], (a)[23]
#elif LLM_PORTS == 25
#define PORTV(a) , (a)[1], (a)[2], (a)[3], (a)[4], (a)[5], (a)[6], (a)[7], (a)[8], (a)[9], (a)[10], (a)[11], (a)[12], (a)[13], (a)[14], (a)[15], (a)[16], (a)[17], (a)[18], (a)[19], (a)[20], (a)[21], (a)[22], (a)[23], (a)[24]
#elif LLM_PORTS == 26
#define PORTV(a) , (a)[1], (a)[2], (a)[3], (a)[4], (a)[5], (a)[6], (a)[7], (a)[8], (a)[9], (a)[10], (a)[11], (a)[12], (a)[13], (a)[14], (a)[15], (a)[16], (a)[17], (a)[18], (a)[19], (a)[20], (a)[21], (a)[22], (a)[23], (a)[24], (a)[25]
#else
#define PORTV(a) , (a)[1], (a)[2], (a)[3], (a)[4], (a)[5], (a)[6], (a)[7], (a)[8], (a)[9], (a)[10], (a)[11], (a)[12], (a)[13], (a)[14], (a)[15], (a)[16], (a)[17], (a)[18], (a)[19], (a)[20], (a)[21], (a)[22], (a)[23], (a)[24], (a)[25], (a)[26]
#endif

static float *read_file(const std::string &name, long *size) {
    FILE *f = fopen(name.c_str(), "rb");
    if (!f) { perror(name.c_str()); return nullptr; }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    float *w = (float *)llm_aligned_alloc(64, (size_t)n);
    if (fread(w, 1, n, f) != (size_t)n) { fprintf(stderr, "short read of %s\n", name.c_str()); return nullptr; }
    fclose(f);
    if (size) *size = n;
    return w;
}

int main(int argc, char **argv) {
    if (argc < 5) { fprintf(stderr, "usage: %s model.flat steps logits.bin tok0 [tok1 ...]\n", argv[0]); return 2; }
    const std::string a1 = argv[1];
    const bool split = a1.rfind("split:", 0) == 0;
    const llm_wvec_t *wp[27]; /* what each weight port is given */
    float *weights = nullptr;
    if (split) {
        const std::string prefix = a1.substr(6);
        for (int p = 0; p < LLM_PORTS; p++) {
            float *b = read_file(prefix + ".p" + std::to_string(p), nullptr);
            if (!b) return 1;
            if (p == 0) weights = b;
            wp[p] = (const llm_wvec_t *)b;
        }
    } else {
        weights = read_file(a1, nullptr);
        if (!weights) return 1;
        for (int p = 0; p < 8; p++) wp[p] = (const llm_wvec_t *)weights;
    }

    llm_config cfg;
    if (llm_read_config(weights, &cfg)) { fprintf(stderr, "bad weight file\n"); return 1; }
    const int kvd = cfg.dim * cfg.n_kv_heads / cfg.n_heads;
    const size_t nstate = llm_state_floats(cfg.vocab_size, cfg.n_layers, cfg.seq_len, kvd);
    const size_t ntok = llm_token_ints(cfg.seq_len);

    const int steps = atoi(argv[2]), np = argc - 4, total = np + steps;
    std::vector<int> seq(total, 0);
    for (int i = 0; i < np; i++) seq[i] = atoi(argv[4 + i]);

    // 1) one position at a time, collecting logits
    FILE *lf = fopen(argv[3], "wb");
    for (int p = 0; p < total - 1; p++) {
        std::vector<float> state(nstate, 0.0f);
        std::vector<int> tk(ntok, 0);
        for (int i = 0; i <= p; i++) tk[i] = seq[i];
        int rc = llm_top((const llm_wvec_t *)weights, (llm_svec_t *)state.data(), tk.data(), p + 1, 1 PORTV(wp), 0, -1, -1);
        if (rc) { fprintf(stderr, "llm_top returned %d\n", rc); return 1; }
        fwrite(state.data(), sizeof(float), cfg.vocab_size, lf);
        if (p >= np - 1) seq[p + 1] = tk[p + 1];
    }
    fclose(lf);

    // 2) the whole generation in a single kernel call
    std::vector<float> state(nstate, 0.0f);
    std::vector<int> tk(ntok, 0);
    for (int i = 0; i < np; i++) tk[i] = seq[i];
    int rc = llm_top((const llm_wvec_t *)weights, (llm_svec_t *)state.data(), tk.data(), np, steps PORTV(wp), 0, -1, -1);
    if (rc) { fprintf(stderr, "llm_top returned %d\n", rc); return 1; }
    for (int i = 0; i < total; i++) {
        if (tk[i] != seq[i]) { fprintf(stderr, "single-call run differs at %d: %d vs %d\n", i, tk[i], seq[i]); return 1; }
    }
    for (int i = 0; i < total; i++) printf("%d%c", tk[i], i + 1 < total ? ' ' : '\n');

    // 3) resume: the state now holds the K/V cache of positions 0 .. total-2. Continue the same conversation with 2 more prompt tokens and
    //    start_pos = total-1 (the last generated token has no K/V yet); the result must equal a fresh run (zero state, start_pos 0) on the same tokens.
    const int np2 = total + 2, steps2 = 3, total2 = np2 + steps2;
    if (total2 - 1 <= cfg.seq_len) {
        std::vector<int> tr(ntok, 0), tf(ntok, 0);
        for (int i = 0; i < total; i++) tr[i] = tf[i] = tk[i];
        tr[total] = tf[total] = seq[0];
        tr[total + 1] = tf[total + 1] = seq[1];
        rc = llm_top((const llm_wvec_t *)weights, (llm_svec_t *)state.data(), tr.data(), np2, steps2 PORTV(wp), total - 1, -1, -1);
        if (rc) { fprintf(stderr, "resume: llm_top returned %d\n", rc); return 1; }
        std::vector<float> fstate(nstate, 0.0f);
        rc = llm_top((const llm_wvec_t *)weights, (llm_svec_t *)fstate.data(), tf.data(), np2, steps2 PORTV(wp), 0, -1, -1);
        if (rc) { fprintf(stderr, "fresh: llm_top returned %d\n", rc); return 1; }
        for (int i = 0; i < total2; i++)
            if (tr[i] != tf[i]) { fprintf(stderr, "RESUME MISMATCH at token %d: %d vs %d\n", i, tr[i], tf[i]); return 1; }
        if (memcmp(state.data(), fstate.data(), sizeof(float) * cfg.vocab_size) != 0) { fprintf(stderr, "RESUME MISMATCH in the last logits\n"); return 1; }
        printf("resume check OK: start_pos %d gives the same %d tokens and the same last logits as a fresh run\n", total - 1, total2);
    }
    // 4) stop token: stopping at the 3rd generated token of the run above must end the loop there (that token is stored, later entries stay untouched)
    if (steps >= 4) {
        const int stop = seq[np + 2];
        int first = -1;
        for (int i = np; i < total; i++) if (seq[i] == stop) { first = i; break; }
        std::vector<float> s2(nstate, 0.0f);
        std::vector<int> ts(ntok, 0);
        for (int i = 0; i < np; i++) ts[i] = seq[i];
        for (int i = np; i < total; i++) ts[i] = -1;
        rc = llm_top((const llm_wvec_t *)weights, (llm_svec_t *)s2.data(), ts.data(), np, steps PORTV(wp), 0, stop, -1);
        if (rc) { fprintf(stderr, "stop: llm_top returned %d\n", rc); return 1; }
        for (int i = np; i <= first; i++) if (ts[i] != seq[i]) { fprintf(stderr, "STOP MISMATCH at %d\n", i); return 1; }
        for (int i = first + 1; i < total; i++) if (ts[i] != -1) { fprintf(stderr, "STOP FAILED: ran past the stop token at %d\n", i); return 1; }
        printf("stop check OK: stopped after token %d (position %d of %d)\n", stop, first, total - 1);
    }
    return 0;
}
