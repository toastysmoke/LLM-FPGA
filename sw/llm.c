/* llm.c - plain-C reference implementation of one Llama forward pass (the readable twin of llm_kernel.cpp).
 *
 * It runs on the PC (main_pc.c is its driver) and reads the same flat fp32 weight file the kernel reads (layout: ref/layout.py).
 * test_c_port.py checks it against the numpy model in ref/. Written for clarity, not speed: the arithmetic is the llama2.c algorithm with
 * no tricks. llm_kernel.cpp computes the same function with a different summation order inside each dot product, so the two agree to
 * about 1e-5 in the logits rather than bit for bit (the kernel is checked against the numpy model by test_kernel.py).
 *
 * Layout of the flat file, as read below: a 256-byte header (magic 'LLMF', version, dtype, model sizes, section offsets), then the sections
 * 64-byte aligned: token embedding, per-layer blocks of nine tensors, final norm, RoPE cos/sin tables, optional separate classifier.
 */
#include "llm.h"
#include <math.h>
#include <string.h>

#define HDR_BYTES 256 /* size of the file header */
#define ALIGN 64      /* every section and tensor starts on a 64-byte boundary */

/* Little-endian reads from the header, so the file does not depend on the host's alignment or endianness rules. */
static uint32_t rd32(const uint8_t *p) { return p[0] | (p[1] << 8) | (p[2] << 16) | ((uint32_t)p[3] << 24); }
static uint64_t rd64(const uint8_t *p) { return (uint64_t)rd32(p) | ((uint64_t)rd32(p + 4) << 32); }
static uint64_t align_up(uint64_t n) { return (n + ALIGN - 1) / ALIGN * ALIGN; }

/* Parse and validate the header (magic, version 1, fp32, head and KV-head counts that divide evenly).
 * Returns 0, -1 bad magic, -2 bad version or dtype, -3 inconsistent head counts: same codes as the kernel's load_cfg. */
int llm_read_config(const void *weights, llm_config *c) {
    const uint8_t *h = (const uint8_t *)weights;
    if (memcmp(h, "LLMF", 4) != 0) return -1;
    if (rd32(h + 4) != 1 || rd32(h + 8) != 0) return -2; /* version 1, fp32 */
    c->dim = (int)rd32(h + 12);
    c->hidden_dim = (int)rd32(h + 16);
    c->n_layers = (int)rd32(h + 20);
    c->n_heads = (int)rd32(h + 24);
    c->n_kv_heads = (int)rd32(h + 28);
    c->vocab_size = (int)rd32(h + 32);
    c->seq_len = (int)rd32(h + 36);
    c->shared = (int)rd32(h + 40);
    if (c->n_kv_heads <= 0 || c->n_heads % c->n_kv_heads || c->dim % c->n_heads) return -3;
    return 0;
}

/* Scratch memory one run needs: the activation vectors, the attention scores of every head, the logits and the K and V caches. */
size_t llm_workspace_bytes(const llm_config *c) {
    int kvd = c->dim * c->n_kv_heads / c->n_heads;
    size_t floats = 3 * (size_t)c->dim + 2 * (size_t)c->hidden_dim + (size_t)c->dim +
                    (size_t)c->n_heads * c->seq_len + (size_t)c->vocab_size +
                    2 * (size_t)c->n_layers * c->seq_len * kvd;
    return floats * sizeof(float);
}

/* Bind a model to a weight file and a zeroed workspace: reads the section offsets from the header, computes where each of the nine
 * tensors of a layer sits (-5 if that disagrees with the file's layer stride) and carves the workspace into the working vectors. */
int llm_init(llm *m, const void *weights, void *ws, size_t ws_bytes) {
    int rc = llm_read_config(weights, &m->c);
    if (rc) return rc;
    const llm_config *c = &m->c;
    if (ws_bytes < llm_workspace_bytes(c)) return -4;
    const uint8_t *h = (const uint8_t *)weights;
    m->base = h;
    m->head_size = c->dim / c->n_heads;
    m->kv_dim = c->dim * c->n_kv_heads / c->n_heads;
    m->off_tok = rd64(h + 44);
    m->off_layers = rd64(h + 52);
    m->layer_stride = rd64(h + 60);
    m->off_final = rd64(h + 68);
    m->off_cos = rd64(h + 76);
    m->off_sin = rd64(h + 84);
    m->off_wcls = rd64(h + 92);

    /* per-layer tensor offsets, same order and alignment as ref/layout.py */
    uint64_t d = c->dim, k = m->kv_dim, hid = c->hidden_dim;
    uint64_t n[9] = {d, d * d, k * d, k * d, d * d, d, hid * d, d * hid, hid * d}, pos = 0;
    for (int i = 0; i < 9; i++) {
        m->rel[i] = pos;
        pos = align_up(pos + 4 * n[i]);
    }
    if (pos != m->layer_stride) return -5;

    memset(ws, 0, llm_workspace_bytes(c));
    float *p = (float *)ws;
    m->x = p;       p += c->dim;
    m->xb = p;      p += c->dim;
    m->xb2 = p;     p += c->dim;
    m->hb = p;      p += c->hidden_dim;
    m->hb2 = p;     p += c->hidden_dim;
    m->q = p;       p += c->dim;
    m->att = p;     p += (size_t)c->n_heads * c->seq_len;
    m->logits = p;  p += c->vocab_size;
    m->kc = p;      p += (size_t)c->n_layers * c->seq_len * m->kv_dim;
    m->vc = p;
    return 0;
}

#define W(m, off) ((const float *)((m)->base + (off)))

/* RMSNorm: out = w * x / sqrt(mean(x^2) + 1e-5). */
static void rmsnorm(float *out, const float *x, const float *w, int n) {
    float ss = 0.0f;
    for (int j = 0; j < n; j++) ss += x[j] * x[j];
    ss = ss / n + 1e-5f;
    ss = 1.0f / sqrtf(ss);
    for (int j = 0; j < n; j++) out[j] = w[j] * (ss * x[j]);
}

/* In-place softmax over n values (max subtracted first for stability). */
static void softmax(float *x, int n) {
    float mx = x[0];
    for (int i = 1; i < n; i++) if (x[i] > mx) mx = x[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - mx); sum += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= sum;
}

/* out[d] = W[d x n] @ x[n]. This is where almost all the time goes, and the part to move into hardware later. */
static void matmul(float *out, const float *x, const float *w, int n, int d) {
    for (int i = 0; i < d; i++) {
        float acc = 0.0f;
        const float *row = w + (size_t)i * n;
        for (int j = 0; j < n; j++) acc += row[j] * x[j];
        out[i] = acc;
    }
}

/* One transformer step for `token` at position `pos`: embedding lookup, then per layer RMSNorm, Q/K/V projections, RoPE, causal
 * attention over positions 0..pos (grouped-query: several query heads share one K/V head), output projection with residual, RMSNorm,
 * SwiGLU feed-forward with residual; finally RMSNorm and the output head. The K and V of this position are written into the cache. */
const float *llm_forward(llm *m, int token, int pos) {
    const llm_config *c = &m->c;
    const int dim = c->dim, hid = c->hidden_dim, hs = m->head_size, kvd = m->kv_dim, seq = c->seq_len;
    const int kv_mul = c->n_heads / c->n_kv_heads;

    memcpy(m->x, W(m, m->off_tok) + (size_t)token * dim, dim * sizeof(float));
    const float *cosr = W(m, m->off_cos) + (size_t)pos * (hs / 2);
    const float *sinr = W(m, m->off_sin) + (size_t)pos * (hs / 2);

    for (int l = 0; l < c->n_layers; l++) {
        const uint64_t lb = m->off_layers + (uint64_t)l * m->layer_stride;
        float *k = m->kc + ((size_t)l * seq + pos) * kvd;
        float *v = m->vc + ((size_t)l * seq + pos) * kvd;

        rmsnorm(m->xb, m->x, W(m, lb + m->rel[0]), dim);
        matmul(m->q, m->xb, W(m, lb + m->rel[1]), dim, dim);
        matmul(k, m->xb, W(m, lb + m->rel[2]), dim, kvd);
        matmul(v, m->xb, W(m, lb + m->rel[3]), dim, kvd);

        for (int i = 0; i < dim; i += 2) { /* RoPE on consecutive pairs inside each head */
            int f = (i % hs) / 2;
            float fc = cosr[f], fs = sinr[f];
            float a = m->q[i], b = m->q[i + 1];
            m->q[i] = a * fc - b * fs;
            m->q[i + 1] = a * fs + b * fc;
            if (i < kvd) {
                a = k[i]; b = k[i + 1];
                k[i] = a * fc - b * fs;
                k[i + 1] = a * fs + b * fc;
            }
        }

        for (int h = 0; h < c->n_heads; h++) {
            const float *qh = m->q + h * hs;
            float *att = m->att + (size_t)h * seq;
            const int g = h / kv_mul;
            for (int t = 0; t <= pos; t++) {
                const float *kt = m->kc + ((size_t)l * seq + t) * kvd + g * hs;
                float s = 0.0f;
                for (int i = 0; i < hs; i++) s += qh[i] * kt[i];
                att[t] = s / sqrtf((float)hs);
            }
            softmax(att, pos + 1);
            float *out = m->xb + h * hs;
            for (int i = 0; i < hs; i++) out[i] = 0.0f;
            for (int t = 0; t <= pos; t++) {
                const float *vt = m->vc + ((size_t)l * seq + t) * kvd + g * hs;
                const float a = att[t];
                for (int i = 0; i < hs; i++) out[i] += a * vt[i];
            }
        }
        matmul(m->xb2, m->xb, W(m, lb + m->rel[4]), dim, dim);
        for (int i = 0; i < dim; i++) m->x[i] += m->xb2[i];

        rmsnorm(m->xb, m->x, W(m, lb + m->rel[5]), dim);
        matmul(m->hb, m->xb, W(m, lb + m->rel[6]), dim, hid);
        matmul(m->hb2, m->xb, W(m, lb + m->rel[8]), dim, hid);
        for (int i = 0; i < hid; i++) {
            float a = m->hb[i];
            m->hb[i] = a * (1.0f / (1.0f + expf(-a))) * m->hb2[i]; /* SiLU(w1 x) * (w3 x) */
        }
        matmul(m->xb, m->hb, W(m, lb + m->rel[7]), hid, dim);
        for (int i = 0; i < dim; i++) m->x[i] += m->xb[i];
    }

    rmsnorm(m->x, m->x, W(m, m->off_final), dim);
    const float *wcls = c->shared ? W(m, m->off_tok) : W(m, m->off_wcls);
    matmul(m->logits, m->x, wcls, dim, c->vocab_size);
    return m->logits;
}

/* Greedy decoding: index of the largest logit (the first one wins ties, as in the kernel). */
int llm_argmax(const float *v, int n) {
    int best = 0;
    for (int i = 1; i < n; i++) if (v[i] > v[best]) best = i;
    return best;
}
