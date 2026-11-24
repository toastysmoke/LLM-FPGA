/* Llama inference core. No file I/O, no malloc, no printf: it only needs
 *   - a pointer to the flat weight file (see ref/layout.py), 64-byte aligned, little-endian CPU
 *   - a pointer to a scratch workspace (size from llm_workspace_bytes)
 * On the PC both come from malloc. On the FPGA the weights sit in DDR4 and the workspace is another DDR4 region.
 */
#ifndef LLM_H
#define LLM_H

#include <stddef.h>
#include <stdint.h>

typedef struct {
    int dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len, shared;
} llm_config;

typedef struct {
    llm_config c;
    int head_size, kv_dim;
    const uint8_t *base;
    uint64_t off_tok, off_layers, layer_stride, off_final, off_cos, off_sin, off_wcls;
    uint64_t rel[9]; /* rms_att wq wk wv wo rms_ffn w1 w2 w3 */
    float *x, *xb, *xb2, *hb, *hb2, *q, *att, *logits, *kc, *vc;
} llm;

/* 0 on success, negative on a bad header. */
int llm_read_config(const void *weights, llm_config *cfg);
size_t llm_workspace_bytes(const llm_config *cfg);
/* The workspace is zeroed here. Returns 0 on success. */
int llm_init(llm *m, const void *weights, void *workspace, size_t workspace_bytes);
/* Runs one token at position pos and returns vocab_size logits (valid until the next call). */
const float *llm_forward(llm *m, int token, int pos);
int llm_argmax(const float *v, int n);

#endif
