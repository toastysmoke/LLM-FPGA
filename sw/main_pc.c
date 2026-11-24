/* PC test driver.   main_pc model.flat steps logits.bin tok0 [tok1 ...]
 * Greedy-decodes `steps` new tokens after the prompt, prints all token ids on one line, and writes every
 * step's logits (float32) to logits.bin so they can be compared with ref/reference.py.
 */
#include "llm.h"
#include "portable_alloc.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    if (argc < 5) { fprintf(stderr, "usage: %s model.flat steps logits.bin tok0 [tok1 ...]\n", argv[0]); return 2; }
    FILE *f = fopen(argv[1], "rb");
    if (!f) { perror(argv[1]); return 1; }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    void *w = llm_aligned_alloc(64, (size_t)n);
    if (fread(w, 1, n, f) != (size_t)n) { fprintf(stderr, "short read\n"); return 1; }
    fclose(f);

    llm_config cfg;
    int rc = llm_read_config(w, &cfg);
    if (rc) { fprintf(stderr, "bad weight file (%d)\n", rc); return 1; }
    size_t wsb = llm_workspace_bytes(&cfg);
    void *ws = malloc(wsb);
    llm m;
    rc = llm_init(&m, w, ws, wsb);
    if (rc) { fprintf(stderr, "llm_init failed (%d)\n", rc); return 1; }

    int steps = atoi(argv[2]), np = argc - 4;
    int total = np + steps;
    if (total - 1 > cfg.seq_len) { fprintf(stderr, "too long for seq_len %d\n", cfg.seq_len); return 1; }
    int *toks = malloc(sizeof(int) * total);
    for (int i = 0; i < np; i++) toks[i] = atoi(argv[4 + i]);
    FILE *lf = fopen(argv[3], "wb");

    int len = np;
    for (int pos = 0; pos < total - 1; pos++) {
        const float *lg = llm_forward(&m, toks[pos], pos);
        fwrite(lg, sizeof(float), cfg.vocab_size, lf);
        if (pos >= np - 1) toks[len++] = llm_argmax(lg, cfg.vocab_size);
    }
    fclose(lf);
    for (int i = 0; i < len; i++) printf("%d%c", toks[i], i + 1 < len ? ' ' : '\n');
    return 0;
}
