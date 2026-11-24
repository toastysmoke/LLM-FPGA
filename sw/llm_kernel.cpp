#include "llm_kernel.h"
#include <math.h>
#include <string.h>

/* Error codes: -1 bad magic, -2 bad version/dtype, -3 bad config, -4 config exceeds LLM_MAX_*,
 *              -5 layer layout mismatch, (-6 unused: weight offsets are 64-bit), -7 bad n_prompt/n_steps/start_pos,
 *              -8 dim or hidden is not a multiple of LLM_LANES, or head size not a multiple of LLM_LANES_ATT,
 *              -9 split weight layout (header word 27) made for another LLM_PORTS or head chunk size */

/* Port depths (in elements). Used only by C/RTL co-simulation (it must know how much of each pointer to copy); they have
 * no effect on the synthesized hardware. Defaults fit stories260K; the cosim config overrides them with -D for small models. */
#ifndef LLM_DEPTH_WEIGHTS
#define LLM_DEPTH_WEIGHTS 264192
#endif
#ifndef LLM_DEPTH_STATE
#define LLM_DEPTH_STATE 164352
#endif
#ifndef LLM_DEPTH_TOKENS
#define LLM_DEPTH_TOKENS 513
#endif

#define MAX_COLS (LLM_MAX_DIM > LLM_MAX_HIDDEN ? LLM_MAX_DIM : LLM_MAX_HIDDEN)
#define MAX_KVD LLM_MAX_DIM
#define ALIGN_F 16u /* 64 bytes in floats */
#define NOSPLIT (~(uint64_t)0) /* 'loff' of a matrix read from the one shared weight file */

struct Cfg {
    int dim, hid, layers, heads, kv_heads, vocab, seq, shared, hs, kvd, kv_mul;
    uint64_t tok, lay0, stride, fin, cos_, sin_, wcls; /* offsets in floats (64-bit: weight files may be over 4 GiB) */
    uint64_t rel[9];                                    /* rms_att wq wk wv wo rms_ffn w1 w2 w3 */
    /* split layout (see sw/split_ports.py): every weight port holds only its own rows of every matrix. Port 0 also holds the header,
     * the embedding table, the norm vectors and the rope tables (tok, norms, fin, cos_, sin_ then point there) before its rows (pb0). */
    int split;
    uint64_t pb0, lstride, chunkstride, norms;          /* floats */
    uint64_t prel[9];                                   /* rows of each matrix of a layer, per port, relative to the layer's block */
};

/* ---- on-chip buffers (BRAM/URAM in hardware) ---- */
static float x[LLM_MAX_DIM], xb[LLM_MAX_DIM], xb2[LLM_MAX_DIM], q[LLM_MAX_DIM];
static float kvec[MAX_KVD], vvec[MAX_KVD];
static float hb[LLM_MAX_HIDDEN], hb2[LLM_MAX_HIDDEN];
static float att[LLM_MAX_HEADS][LLM_MAX_SEQ];  /* attention scores of every head */
static float normw[LLM_MAX_DIM];
static float kvrow[MAX_KVD];              /* one position's keys or values, all heads */
static float ropec[LLM_MAX_HEAD_SIZE / 2], ropes[LLM_MAX_HEAD_SIZE / 2];

/* One float of the weight file, by its position in floats. The port is LLM_LANES floats wide, so this reads one wide word. */
static inline float wf(const llm_wvec_t *w, uint64_t i) { return llm_get(w[i / LLM_LANES], (int)(i % LLM_LANES)); }

static uint32_t align_f(uint32_t n) { return (n + ALIGN_F - 1) / ALIGN_F * ALIGN_F; }

static int load_cfg(const llm_wvec_t *w, Cfg &c) {
    uint32_t h[64];
    for (int k = 0; k < 64; k++) { float f = wf(w, k); memcpy(&h[k], &f, 4); } /* the 256-byte header */
    if (h[0] != 0x464D4C4Cu) return -1;       /* "LLMF" */
    if (h[1] != 1u || h[2] != 0u) return -2;
    c.dim = (int)h[3]; c.hid = (int)h[4]; c.layers = (int)h[5]; c.heads = (int)h[6];
    c.kv_heads = (int)h[7]; c.vocab = (int)h[8]; c.seq = (int)h[9]; c.shared = (int)h[10];
    if (c.dim <= 0 || c.heads <= 0 || c.kv_heads <= 0 || c.dim % c.heads || c.heads % c.kv_heads) return -3;
    c.hs = c.dim / c.heads;
    c.kvd = c.dim * c.kv_heads / c.heads;
    c.kv_mul = c.heads / c.kv_heads;
    if (c.hs % 2) return -3;
    if (c.dim > LLM_MAX_DIM || c.hid > LLM_MAX_HIDDEN || c.seq > LLM_MAX_SEQ || c.heads > LLM_MAX_HEADS ||
        c.hs > LLM_MAX_HEAD_SIZE) return -4;
    if (c.dim % LLM_LANES || c.hid % LLM_LANES || c.hs % LLM_LANES_ATT) return -8; /* rows and K/V slices must be whole wide words */
    /* byte offsets are 64-bit words at h[11..] (low word first); convert to floats */
    uint64_t o[7];
    for (int i = 0; i < 7; i++) o[i] = ((uint64_t)h[12 + 2 * i] << 32 | (uint64_t)h[11 + 2 * i]) / 4u;
    c.tok = o[0]; c.lay0 = o[1]; c.stride = o[2]; c.fin = o[3]; c.cos_ = o[4]; c.sin_ = o[5]; c.wcls = o[6];

    uint32_t d = (uint32_t)c.dim, k = (uint32_t)c.kvd, hd = (uint32_t)c.hid;
    uint32_t n[9] = {d, d * d, k * d, k * d, d * d, d, hd * d, d * hd, hd * d}, pos = 0;
    for (int i = 0; i < 9; i++) { c.rel[i] = pos; pos = align_f(pos + n[i]); }
    if (pos != c.stride) return -5;
    c.split = 0; c.pb0 = 0; c.lstride = 0; c.chunkstride = 0; c.norms = 0;
    for (int i = 0; i < 9; i++) c.prel[i] = 0;
    if (h[27] == 0x53504C54u) { /* "SPLT": the per-port layout */
        if (h[28] != (uint32_t)MAX_COLS || h[29] != (uint32_t)LLM_PORTS) return -9;
        c.split = 1;
        c.pb0 = h[30]; c.tok = h[31]; c.norms = h[32]; c.fin = h[33]; c.cos_ = h[34]; c.sin_ = h[35]; c.lstride = h[36];
        const uint32_t dout[9] = {0, d, k, k, d, 0, hd, d, hd}, nin[9] = {0, d, d, d, d, 0, d, hd, d};
        uint32_t sp = 0;
        for (int i = 1; i < 9; i++) {
            if (i == 5) continue; /* rms_ffn is not a matrix */
            c.prel[i] = sp;
            sp = align_f(sp + ((dout[i] + (uint32_t)LLM_PORTS - 1) / (uint32_t)LLM_PORTS) * nin[i]);
        }
        if (sp != h[36]) return -5;
        c.chunkstride = align_f(((uint32_t)(MAX_COLS + LLM_PORTS - 1) / (uint32_t)LLM_PORTS) * d);
    }
    return 0;
}

static void load_vec(float *dst, const float *src, int n) {
    for (int i = 0; i < n; i++) dst[i] = src[i];
}

static void load_vecw(float *dst, const llm_wvec_t *w, uint64_t off, int n) {
    for (int i = 0; i < n; i++) dst[i] = wf(w, off + i);
}

/* LLM_LANES floats per clock cycle: off and n are multiples of LLM_LANES */
static void load_vecw_wide(float *dst, const llm_wvec_t *w, uint64_t off, int n) {
    const llm_wvec_t *p = w + off / LLM_LANES;
    for (int j = 0; j < n / LLM_LANES; j++) {
#pragma HLS PIPELINE II = 1
        const llm_wvec_t tv = p[j];
        for (int l = 0; l < LLM_LANES; l++) {
#pragma HLS UNROLL
            dst[j * LLM_LANES + l] = llm_get(tv, l);
        }
    }
}

/* Wide reads and writes of the state area (K/V cache): `foff` is the position in floats and must be a multiple of LLM_LANES_ATT, as is n. */
static void load_kv(float *dst, const llm_svec_t *s, uint64_t foff, int n) {
    const llm_svec_t *p = s + foff / LLM_LANES_ATT;
    for (int j = 0; j < n / LLM_LANES_ATT; j++) {
#pragma HLS PIPELINE II = 1
        const llm_svec_t t = p[j];
        for (int l = 0; l < LLM_LANES_ATT; l++) {
#pragma HLS UNROLL
            dst[j * LLM_LANES_ATT + l] = llm_get(t, l);
        }
    }
}

static void store_kv(llm_svec_t *s, uint64_t foff, const float *src, int n) {
    llm_svec_t *p = s + foff / LLM_LANES_ATT;
    for (int j = 0; j < n / LLM_LANES_ATT; j++) {
#pragma HLS PIPELINE II = 1
        llm_svec_t t{};
        for (int l = 0; l < LLM_LANES_ATT; l++) {
#pragma HLS UNROLL
            llm_set(t, l, src[j * LLM_LANES_ATT + l]);
        }
        p[j] = t;
    }
}

/* Sum of n values coming one per clock cycle, in LLM_VEC_SUMS running totals that form a shift register (each element is added to the
 * oldest total, so a floating-point add has LLM_VEC_SUMS cycles to finish). With one total this is the plain sequential sum. */
struct VecSums {
    float t[LLM_VEC_SUMS];
    int s; /* the slot the next value is added to: slots are used in turn, so a slot is read again LLM_VEC_SUMS values later */
    VecSums() : s(0) { for (int e = 0; e < LLM_VEC_SUMS; e++) t[e] = 0.0f; }
    inline void add(float x) {
        t[s] = t[s] + x;
        s = (s + 1 == LLM_VEC_SUMS) ? 0 : s + 1;
    }
    inline float total() const {
        /* the slots added in pairs (a tree): log2(LLM_VEC_SUMS) additions in a row, not LLM_VEC_SUMS; oldest slot first */
        float v[LLM_VEC_SUMS];
#pragma HLS ARRAY_PARTITION variable = v complete
        for (int e = 0; e < LLM_VEC_SUMS; e++) {
#pragma HLS UNROLL
            v[e] = t[(s + e) % LLM_VEC_SUMS];
        }
        for (int step = 1; step < LLM_VEC_SUMS; step *= 2) {
#pragma HLS UNROLL
            for (int i = 0; i + step < LLM_VEC_SUMS; i += 2 * step) {
#pragma HLS UNROLL
                v[i] += v[i + step];
            }
        }
        return v[0];
    }
};

static void rmsnorm(float *out, const float *in, const float *w, int n) {
    VecSums acc;
#pragma HLS ARRAY_PARTITION variable = acc.t complete
    for (int j = 0; j < n; j++) {
#pragma HLS PIPELINE II = 1
#pragma HLS DEPENDENCE variable = acc.t type = inter direction = RAW dependent = true distance = LLM_VEC_SUMS
        acc.add(in[j] * in[j]);
    }
    float ss = acc.total();
    ss = ss / n + 1e-5f;
    ss = 1.0f / sqrtf(ss);
    for (int j = 0; j < n; j++) out[j] = w[j] * (ss * in[j]);
}

/* Adds N values in pairs (a tree): log2(N) additions in a row instead of N. */
template <int N>
static inline float tree_sum(float *v) {
#pragma HLS ARRAY_PARTITION variable = v complete
    for (int step = 1; step < N; step *= 2) {
#pragma HLS UNROLL
        for (int i = 0; i + step < N; i += 2 * step) {
#pragma HLS UNROLL
            v[i] += v[i + step];
        }
    }
    return v[0];
}

/* ---- the matrix unit: weight rows stream through stages that run at the same time (a dataflow region).
 * Stage 1 reads the weight words in order, one wide word per clock cycle, multiplies each with the input vector and adds into
 * LLM_ACC_DEPTH running totals per lane; when a row ends it passes its totals to stage 2 and starts the next row at once.
 * Stage 2 adds the totals of each row (tree_sum) and stores the result. So the final sum of one row overlaps the reading of the next,
 * and no row is ever stored: the weights are used as they arrive.
 * With LLM_PORTS > 1 there is one such pair of stages per weight port; the rows of a matrix are split into LLM_PORTS blocks of
 * consecutive rows, and each port streams its own block. A row is always summed by one port, in the same order as with a single
 * port, so the result does not depend on the number of ports. */
struct RowTotals { float v[LLM_LANES * LLM_ACC_DEPTH]; };

#ifdef __SYNTHESIS__
#include <hls_stream.h>
template <typename T> using llm_fifo = hls::stream<T>;
#else
#include <deque>
template <typename T> struct llm_fifo {
    std::deque<T> q;
    void write(const T &x) { q.push_back(x); }
    T read() { T x = q.front(); q.pop_front(); return x; }
};
#endif

#define PORT_ROWS ((MAX_COLS + LLM_PORTS - 1) / LLM_PORTS)

static void row_stream(llm_fifo<RowTotals> &totals, const float *vin, const llm_wvec_t *base, int n_in, int rows) {
    const int words = n_in / LLM_LANES;
    float part[LLM_ACC_DEPTH][LLM_LANES];
#pragma HLS ARRAY_PARTITION variable = part complete dim = 0
    for (int d = 0; d < LLM_ACC_DEPTH; d++)
        for (int l = 0; l < LLM_LANES; l++) part[d][l] = 0.0f;
    /* The running totals of a lane form a shift register: each word takes the oldest total (the one added LLM_ACC_DEPTH words ago),
     * adds its product, and the sum becomes the newest. Fixed positions, so no multiplexers; and the oldest total is always
     * LLM_ACC_DEPTH words old, which is longer than a floating-point add takes. */
    int j = 0; /* word inside the row */
    int s = 0; /* j % LLM_ACC_DEPTH: the slot this word adds into */
    bool young = true; /* j < LLM_ACC_DEPTH: the first words of a row start from zero */
    bool valid[LLM_ACC_DEPTH]; /* slot e holds a total of this row (a row shorter than LLM_ACC_DEPTH words only used the first `words` slots) */
#pragma HLS ARRAY_PARTITION variable = valid complete
    for (int e = 0; e < LLM_ACC_DEPTH; e++) valid[e] = (words >= LLM_ACC_DEPTH || e < words);
    const int last = words - 1;
    const uint64_t total = (uint64_t)words * (uint64_t)rows;
    for (uint64_t t = 0; t < total; t++) {
#pragma HLS PIPELINE II = 1
/* slot s is read again exactly LLM_ACC_DEPTH words later: tell HLS that distance (it then gives the add that many cycles instead of one) */
#pragma HLS DEPENDENCE variable = part type = inter direction = RAW dependent = true distance = LLM_ACC_DEPTH
        const llm_wvec_t wv = base[t]; /* one wide read: LLM_LANES weights */
        for (int l = 0; l < LLM_LANES; l++) {
#pragma HLS UNROLL
            const float prod = llm_get(wv, l) * vin[j * LLM_LANES + l];
            const float oldest = young ? 0.0f : part[s][l]; /* the first words of a row start the totals from zero */
            part[s][l] = oldest + prod;
        }
        if (j == last) {
            RowTotals r;
            /* the slots in fixed order (no rotation, so no wide multiplexers); a row shorter than LLM_ACC_DEPTH words only used the first `words` slots */
            for (int l = 0; l < LLM_LANES; l++)
                for (int e = 0; e < LLM_ACC_DEPTH; e++) r.v[l * LLM_ACC_DEPTH + e] = valid[e] ? part[e][l] : 0.0f; /* only totals of this row */
            totals.write(r);
            j = 0;
            s = 0;
            young = true;
        } else {
            j++;
            young = young && (s != LLM_ACC_DEPTH - 1);
            s = (s + 1 == LLM_ACC_DEPTH) ? 0 : s + 1;
        }
    }
}

/* The final sum of a row in two small stages, so no wide multiplexers and few adders are needed (a first version that made HLS share the
 * adders of one big tree (II = LLM_ACC_DEPTH) used 68% of the LUTs and could not be routed):
 *   rs_slots: one slot per clock cycle: the LLM_LANES totals of slot e are added in a tree (LLM_LANES - 1 adders, one slot per cycle)
 *   rs_final: the LLM_ACC_DEPTH slot sums of a row are added in a tree and stored. */
static void rs_slots(llm_fifo<float> &part, llm_fifo<RowTotals> &totals, int rows) {
    for (int i = 0; i < rows; i++) {
        const RowTotals r = totals.read();
        for (int e = 0; e < LLM_ACC_DEPTH; e++) {
#pragma HLS PIPELINE II = 1
            float v[LLM_LANES];
#pragma HLS ARRAY_PARTITION variable = v complete
            for (int l = 0; l < LLM_LANES; l++) {
#pragma HLS UNROLL
                v[l] = r.v[l * LLM_ACC_DEPTH + e];
            }
            part.write(tree_sum<LLM_LANES>(v));
        }
    }
}

static void rs_final(llm_fifo<float> &outf, llm_fifo<float> &part, int rows) {
    for (int i = 0; i < rows; i++) {
        float v[LLM_ACC_DEPTH];
#pragma HLS ARRAY_PARTITION variable = v complete
        for (int e = 0; e < LLM_ACC_DEPTH; e++) {
#pragma HLS UNROLL
            v[e] = part.read();
        }
        outf.write(tree_sum<LLM_ACC_DEPTH>(v));
    }
}

static void row_sum(llm_fifo<float> &outf, llm_fifo<RowTotals> &totals, int rows) {
#pragma HLS DATAFLOW
    llm_fifo<float> part;
#pragma HLS STREAM variable = part depth = 16
    rs_slots(part, totals, rows);
    rs_final(outf, part, rows);
}

/* rows of port p: [p * per_port, p * per_port + rows_of(p)) */
static inline int rows_of(int p, int per_port, int d_out) {
    int r = d_out - p * per_port;
    return r < 0 ? 0 : (r > per_port ? per_port : r);
}

/* Which on-chip vector a matrix multiplication reads and writes. The matrix unit is one piece of hardware shared by all of them, so it
 * takes a number instead of a pointer. SRC_NONE: the input vector is in `hvin`; DST_RES: the result stays in `res`. */
enum { SRC_XB, SRC_HB, SRC_HEAD };
enum { DST_Q, DST_KVEC, DST_VVEC, DST_XB2, DST_HB, DST_HB2, DST_XB, DST_HEAD };

static float hvin_g[MAX_COLS]; /* the input of the output head */
static float head_res[MAX_COLS]; /* one chunk of logits */

/* Merge stage: takes the finished rows of every port (row r of port p is row p * per_port + r of the result) and stores them, while the
 * ports are still streaming the next rows. The row-by-row order matches the order the ports produce them, so no port waits.
 * Per destination: Q, K, V, HB, head: plain store. XB2 and XB (the output and down projections): residual add x += result.
 * HB2 (the up projection): the SiLU gate hb = silu(hb) * result is applied here, so no separate pass over the hidden vector is needed.
 * Same operations in the same order as the separate loops they replace (bit-identical). */
static void merge_out(llm_fifo<float> &o0, llm_fifo<float> &o1, llm_fifo<float> &o2, llm_fifo<float> &o3, llm_fifo<float> &o4, llm_fifo<float> &o5, llm_fifo<float> &o6, llm_fifo<float> &o7, llm_fifo<float> &o8, llm_fifo<float> &o9, llm_fifo<float> &o10, llm_fifo<float> &o11, llm_fifo<float> &o12, llm_fifo<float> &o13, llm_fifo<float> &o14, llm_fifo<float> &o15, llm_fifo<float> &o16, llm_fifo<float> &o17, llm_fifo<float> &o18, llm_fifo<float> &o19, llm_fifo<float> &o20, llm_fifo<float> &o21, llm_fifo<float> &o22, llm_fifo<float> &o23, llm_fifo<float> &o24, llm_fifo<float> &o25, llm_fifo<float> &o26, int dst, int per_port, int d_out) {
    for (int r = 0; r < per_port; r++) {
        for (int p = 0; p < LLM_PORTS; p++) {
#pragma HLS PIPELINE II = 1
#pragma HLS DEPENDENCE variable = hb type = inter direction = RAW dependent = false
#pragma HLS DEPENDENCE variable = hb type = inter direction = WAR dependent = false
            const int i = p * per_port + r;
            if (i < d_out) {
                float v = 0.0f;
                switch (p) {
                case 0: v = o0.read(); break;
                case 1: v = o1.read(); break;
                case 2: v = o2.read(); break;
                case 3: v = o3.read(); break;
                case 4: v = o4.read(); break;
                case 5: v = o5.read(); break;
                case 6: v = o6.read(); break;
                case 7: v = o7.read(); break;
                case 8: v = o8.read(); break;
                case 9: v = o9.read(); break;
                case 10: v = o10.read(); break;
                case 11: v = o11.read(); break;
                case 12: v = o12.read(); break;
                case 13: v = o13.read(); break;
                case 14: v = o14.read(); break;
                case 15: v = o15.read(); break;
                case 16: v = o16.read(); break;
                case 17: v = o17.read(); break;
                case 18: v = o18.read(); break;
                case 19: v = o19.read(); break;
                case 20: v = o20.read(); break;
                case 21: v = o21.read(); break;
                case 22: v = o22.read(); break;
                case 23: v = o23.read(); break;
                case 24: v = o24.read(); break;
                case 25: v = o25.read(); break;
                case 26: v = o26.read(); break;
                default: break;
                }
                switch (dst) {
                case DST_Q: q[i] = v; break;
                case DST_KVEC: kvec[i] = v; break;
                case DST_VVEC: vvec[i] = v; break;
                case DST_XB2: xb2[i] = v; break;
                case DST_HB: hb[i] = v; break;
                case DST_HB2: {
                    const float a = hb[i];
                    hb[i] = a * (1.0f / (1.0f + expf(-a))) * v;
                    break;
                }
                case DST_HEAD: head_res[i] = v; break;
                default: xb[i] = v; break;
                }
            }
        }
    }
}

/* The scalars of one matrix multiplication, computed in their own (registered) stage: per_port rows, block size in words, offsets. */
struct MvPrep { int per_port, n_in, d_out; uint64_t blk, first, lw, pbw; bool split; };
static MvPrep mv_prep(uint64_t woff, uint64_t loff, uint64_t pb0, int n_in, int d_out) {
#pragma HLS INLINE off
    MvPrep m;
    m.n_in = n_in;
    m.d_out = d_out;
    m.per_port = (d_out + LLM_PORTS - 1) / LLM_PORTS;
    m.blk = (uint64_t)m.per_port * (uint64_t)n_in / LLM_LANES; /* words in a block of rows */
    m.first = woff / LLM_LANES;
    m.split = (loff != NOSPLIT); /* split layout: port p reads its own block, starting at loff in its own buffer (port 0's buffer also holds the shared part before it) */
    m.lw = m.split ? loff / LLM_LANES : 0;
    m.pbw = m.split ? pb0 / LLM_LANES : 0;
    return m;
}

static void matvec_flow(                        const float *v0, const float *v1, const float *v2, const float *v3, const float *v4, const float *v5, const float *v6, const float *v7, const float *v8, const float *v9, const float *v10, const float *v11, const float *v12, const float *v13, const float *v14, const float *v15, const float *v16, const float *v17, const float *v18, const float *v19, const float *v20, const float *v21, const float *v22, const float *v23, const float *v24, const float *v25, const float *v26,
                        const llm_wvec_t *w0, const llm_wvec_t *w1, const llm_wvec_t *w2, const llm_wvec_t *w3, const llm_wvec_t *w4, const llm_wvec_t *w5, const llm_wvec_t *w6, const llm_wvec_t *w7, const llm_wvec_t *w8, const llm_wvec_t *w9, const llm_wvec_t *w10, const llm_wvec_t *w11, const llm_wvec_t *w12, const llm_wvec_t *w13, const llm_wvec_t *w14, const llm_wvec_t *w15, const llm_wvec_t *w16, const llm_wvec_t *w17, const llm_wvec_t *w18, const llm_wvec_t *w19, const llm_wvec_t *w20, const llm_wvec_t *w21, const llm_wvec_t *w22, const llm_wvec_t *w23, const llm_wvec_t *w24, const llm_wvec_t *w25, const llm_wvec_t *w26, const MvPrep mp, int n_in, int d_out, int dst) {
#pragma HLS DATAFLOW
    const int per_port = mp.per_port;
    const uint64_t blk = mp.blk, first = mp.first, lw = mp.lw, pbw = mp.pbw;
    const bool split = mp.split;
    llm_fifo<float> o0;
#pragma HLS STREAM variable = o0 depth = 4
    llm_fifo<float> o1;
#pragma HLS STREAM variable = o1 depth = 4
    llm_fifo<float> o2;
#pragma HLS STREAM variable = o2 depth = 4
    llm_fifo<float> o3;
#pragma HLS STREAM variable = o3 depth = 4
    llm_fifo<float> o4;
#pragma HLS STREAM variable = o4 depth = 4
    llm_fifo<float> o5;
#pragma HLS STREAM variable = o5 depth = 4
    llm_fifo<float> o6;
#pragma HLS STREAM variable = o6 depth = 4
    llm_fifo<float> o7;
#pragma HLS STREAM variable = o7 depth = 4
    llm_fifo<float> o8;
#pragma HLS STREAM variable = o8 depth = 4
    llm_fifo<float> o9;
#pragma HLS STREAM variable = o9 depth = 4
    llm_fifo<float> o10;
#pragma HLS STREAM variable = o10 depth = 4
    llm_fifo<float> o11;
#pragma HLS STREAM variable = o11 depth = 4
    llm_fifo<float> o12;
#pragma HLS STREAM variable = o12 depth = 4
    llm_fifo<float> o13;
#pragma HLS STREAM variable = o13 depth = 4
    llm_fifo<float> o14;
#pragma HLS STREAM variable = o14 depth = 4
    llm_fifo<float> o15;
#pragma HLS STREAM variable = o15 depth = 4
    llm_fifo<float> o16;
#pragma HLS STREAM variable = o16 depth = 4
    llm_fifo<float> o17;
#pragma HLS STREAM variable = o17 depth = 4
    llm_fifo<float> o18;
#pragma HLS STREAM variable = o18 depth = 4
    llm_fifo<float> o19;
#pragma HLS STREAM variable = o19 depth = 4
    llm_fifo<float> o20;
#pragma HLS STREAM variable = o20 depth = 4
    llm_fifo<float> o21;
#pragma HLS STREAM variable = o21 depth = 4
    llm_fifo<float> o22;
#pragma HLS STREAM variable = o22 depth = 4
    llm_fifo<float> o23;
#pragma HLS STREAM variable = o23 depth = 4
    llm_fifo<float> o24;
#pragma HLS STREAM variable = o24 depth = 4
    llm_fifo<float> o25;
#pragma HLS STREAM variable = o25 depth = 4
    llm_fifo<float> o26;
#pragma HLS STREAM variable = o26 depth = 4
    llm_fifo<RowTotals> f0;
#pragma HLS STREAM variable = f0 depth = 2
#pragma HLS BIND_STORAGE variable = f0 type = fifo impl = srl
    row_stream(f0, v0, w0 + (split ? pbw + lw : first), n_in, rows_of(0, per_port, d_out));
    row_sum(o0, f0, rows_of(0, per_port, d_out));
#if LLM_PORTS >= 2
    llm_fifo<RowTotals> f1;
#pragma HLS STREAM variable = f1 depth = 2
#pragma HLS BIND_STORAGE variable = f1 type = fifo impl = srl
    row_stream(f1, v1, w1 + (split ? lw : first + blk), n_in, rows_of(1, per_port, d_out));
    row_sum(o1, f1, rows_of(1, per_port, d_out));
#endif
#if LLM_PORTS >= 3
    llm_fifo<RowTotals> f2;
#pragma HLS STREAM variable = f2 depth = 2
#pragma HLS BIND_STORAGE variable = f2 type = fifo impl = srl
    row_stream(f2, v2, w2 + (split ? lw : first + 2 * blk), n_in, rows_of(2, per_port, d_out));
    row_sum(o2, f2, rows_of(2, per_port, d_out));
#endif
#if LLM_PORTS >= 4
    llm_fifo<RowTotals> f3;
#pragma HLS STREAM variable = f3 depth = 2
#pragma HLS BIND_STORAGE variable = f3 type = fifo impl = srl
    row_stream(f3, v3, w3 + (split ? lw : first + 3 * blk), n_in, rows_of(3, per_port, d_out));
    row_sum(o3, f3, rows_of(3, per_port, d_out));
#endif
#if LLM_PORTS >= 5
    llm_fifo<RowTotals> f4;
#pragma HLS STREAM variable = f4 depth = 2
#pragma HLS BIND_STORAGE variable = f4 type = fifo impl = srl
    row_stream(f4, v4, w4 + (split ? lw : first + 4 * blk), n_in, rows_of(4, per_port, d_out));
    row_sum(o4, f4, rows_of(4, per_port, d_out));
#endif
#if LLM_PORTS >= 6
    llm_fifo<RowTotals> f5;
#pragma HLS STREAM variable = f5 depth = 2
#pragma HLS BIND_STORAGE variable = f5 type = fifo impl = srl
    row_stream(f5, v5, w5 + (split ? lw : first + 5 * blk), n_in, rows_of(5, per_port, d_out));
    row_sum(o5, f5, rows_of(5, per_port, d_out));
#endif
#if LLM_PORTS >= 7
    llm_fifo<RowTotals> f6;
#pragma HLS STREAM variable = f6 depth = 2
#pragma HLS BIND_STORAGE variable = f6 type = fifo impl = srl
    row_stream(f6, v6, w6 + (split ? lw : first + 6 * blk), n_in, rows_of(6, per_port, d_out));
    row_sum(o6, f6, rows_of(6, per_port, d_out));
#endif
#if LLM_PORTS >= 8
    llm_fifo<RowTotals> f7;
#pragma HLS STREAM variable = f7 depth = 2
#pragma HLS BIND_STORAGE variable = f7 type = fifo impl = srl
    row_stream(f7, v7, w7 + (split ? lw : first + 7 * blk), n_in, rows_of(7, per_port, d_out));
    row_sum(o7, f7, rows_of(7, per_port, d_out));
#endif
#if LLM_PORTS >= 9
    llm_fifo<RowTotals> f8;
#pragma HLS STREAM variable = f8 depth = 2
#pragma HLS BIND_STORAGE variable = f8 type = fifo impl = srl
    row_stream(f8, v8, w8 + (split ? lw : first + 8 * blk), n_in, rows_of(8, per_port, d_out));
    row_sum(o8, f8, rows_of(8, per_port, d_out));
#endif
#if LLM_PORTS >= 10
    llm_fifo<RowTotals> f9;
#pragma HLS STREAM variable = f9 depth = 2
#pragma HLS BIND_STORAGE variable = f9 type = fifo impl = srl
    row_stream(f9, v9, w9 + (split ? lw : first + 9 * blk), n_in, rows_of(9, per_port, d_out));
    row_sum(o9, f9, rows_of(9, per_port, d_out));
#endif
#if LLM_PORTS >= 11
    llm_fifo<RowTotals> f10;
#pragma HLS STREAM variable = f10 depth = 2
#pragma HLS BIND_STORAGE variable = f10 type = fifo impl = srl
    row_stream(f10, v10, w10 + (split ? lw : first + 10 * blk), n_in, rows_of(10, per_port, d_out));
    row_sum(o10, f10, rows_of(10, per_port, d_out));
#endif
#if LLM_PORTS >= 12
    llm_fifo<RowTotals> f11;
#pragma HLS STREAM variable = f11 depth = 2
#pragma HLS BIND_STORAGE variable = f11 type = fifo impl = srl
    row_stream(f11, v11, w11 + (split ? lw : first + 11 * blk), n_in, rows_of(11, per_port, d_out));
    row_sum(o11, f11, rows_of(11, per_port, d_out));
#endif
#if LLM_PORTS >= 13
    llm_fifo<RowTotals> f12;
#pragma HLS STREAM variable = f12 depth = 2
#pragma HLS BIND_STORAGE variable = f12 type = fifo impl = srl
    row_stream(f12, v12, w12 + (split ? lw : first + 12 * blk), n_in, rows_of(12, per_port, d_out));
    row_sum(o12, f12, rows_of(12, per_port, d_out));
#endif
#if LLM_PORTS >= 14
    llm_fifo<RowTotals> f13;
#pragma HLS STREAM variable = f13 depth = 2
#pragma HLS BIND_STORAGE variable = f13 type = fifo impl = srl
    row_stream(f13, v13, w13 + (split ? lw : first + 13 * blk), n_in, rows_of(13, per_port, d_out));
    row_sum(o13, f13, rows_of(13, per_port, d_out));
#endif
#if LLM_PORTS >= 15
    llm_fifo<RowTotals> f14;
#pragma HLS STREAM variable = f14 depth = 2
#pragma HLS BIND_STORAGE variable = f14 type = fifo impl = srl
    row_stream(f14, v14, w14 + (split ? lw : first + 14 * blk), n_in, rows_of(14, per_port, d_out));
    row_sum(o14, f14, rows_of(14, per_port, d_out));
#endif
#if LLM_PORTS >= 16
    llm_fifo<RowTotals> f15;
#pragma HLS STREAM variable = f15 depth = 2
#pragma HLS BIND_STORAGE variable = f15 type = fifo impl = srl
    row_stream(f15, v15, w15 + (split ? lw : first + 15 * blk), n_in, rows_of(15, per_port, d_out));
    row_sum(o15, f15, rows_of(15, per_port, d_out));
#endif
#if LLM_PORTS >= 17
    llm_fifo<RowTotals> f16;
#pragma HLS STREAM variable = f16 depth = 2
#pragma HLS BIND_STORAGE variable = f16 type = fifo impl = srl
    row_stream(f16, v16, w16 + (split ? lw : first + 16 * blk), n_in, rows_of(16, per_port, d_out));
    row_sum(o16, f16, rows_of(16, per_port, d_out));
#endif
#if LLM_PORTS >= 18
    llm_fifo<RowTotals> f17;
#pragma HLS STREAM variable = f17 depth = 2
#pragma HLS BIND_STORAGE variable = f17 type = fifo impl = srl
    row_stream(f17, v17, w17 + (split ? lw : first + 17 * blk), n_in, rows_of(17, per_port, d_out));
    row_sum(o17, f17, rows_of(17, per_port, d_out));
#endif
#if LLM_PORTS >= 19
    llm_fifo<RowTotals> f18;
#pragma HLS STREAM variable = f18 depth = 2
#pragma HLS BIND_STORAGE variable = f18 type = fifo impl = srl
    row_stream(f18, v18, w18 + (split ? lw : first + 18 * blk), n_in, rows_of(18, per_port, d_out));
    row_sum(o18, f18, rows_of(18, per_port, d_out));
#endif
#if LLM_PORTS >= 20
    llm_fifo<RowTotals> f19;
#pragma HLS STREAM variable = f19 depth = 2
#pragma HLS BIND_STORAGE variable = f19 type = fifo impl = srl
    row_stream(f19, v19, w19 + (split ? lw : first + 19 * blk), n_in, rows_of(19, per_port, d_out));
    row_sum(o19, f19, rows_of(19, per_port, d_out));
#endif
#if LLM_PORTS >= 21
    llm_fifo<RowTotals> f20;
#pragma HLS STREAM variable = f20 depth = 2
#pragma HLS BIND_STORAGE variable = f20 type = fifo impl = srl
    row_stream(f20, v20, w20 + (split ? lw : first + 20 * blk), n_in, rows_of(20, per_port, d_out));
    row_sum(o20, f20, rows_of(20, per_port, d_out));
#endif
#if LLM_PORTS >= 22
    llm_fifo<RowTotals> f21;
#pragma HLS STREAM variable = f21 depth = 2
#pragma HLS BIND_STORAGE variable = f21 type = fifo impl = srl
    row_stream(f21, v21, w21 + (split ? lw : first + 21 * blk), n_in, rows_of(21, per_port, d_out));
    row_sum(o21, f21, rows_of(21, per_port, d_out));
#endif
#if LLM_PORTS >= 23
    llm_fifo<RowTotals> f22;
#pragma HLS STREAM variable = f22 depth = 2
#pragma HLS BIND_STORAGE variable = f22 type = fifo impl = srl
    row_stream(f22, v22, w22 + (split ? lw : first + 22 * blk), n_in, rows_of(22, per_port, d_out));
    row_sum(o22, f22, rows_of(22, per_port, d_out));
#endif
#if LLM_PORTS >= 24
    llm_fifo<RowTotals> f23;
#pragma HLS STREAM variable = f23 depth = 2
#pragma HLS BIND_STORAGE variable = f23 type = fifo impl = srl
    row_stream(f23, v23, w23 + (split ? lw : first + 23 * blk), n_in, rows_of(23, per_port, d_out));
    row_sum(o23, f23, rows_of(23, per_port, d_out));
#endif
#if LLM_PORTS >= 25
    llm_fifo<RowTotals> f24;
#pragma HLS STREAM variable = f24 depth = 2
#pragma HLS BIND_STORAGE variable = f24 type = fifo impl = srl
    row_stream(f24, v24, w24 + (split ? lw : first + 24 * blk), n_in, rows_of(24, per_port, d_out));
    row_sum(o24, f24, rows_of(24, per_port, d_out));
#endif
#if LLM_PORTS >= 26
    llm_fifo<RowTotals> f25;
#pragma HLS STREAM variable = f25 depth = 2
#pragma HLS BIND_STORAGE variable = f25 type = fifo impl = srl
    row_stream(f25, v25, w25 + (split ? lw : first + 25 * blk), n_in, rows_of(25, per_port, d_out));
    row_sum(o25, f25, rows_of(25, per_port, d_out));
#endif
#if LLM_PORTS >= 27
    llm_fifo<RowTotals> f26;
#pragma HLS STREAM variable = f26 depth = 2
#pragma HLS BIND_STORAGE variable = f26 type = fifo impl = srl
    row_stream(f26, v26, w26 + (split ? lw : first + 26 * blk), n_in, rows_of(26, per_port, d_out));
    row_sum(o26, f26, rows_of(26, per_port, d_out));
#endif
    merge_out(o0, o1, o2, o3, o4, o5, o6, o7, o8, o9, o10, o11, o12, o13, o14, o15, o16, o17, o18, o19, o20, o21, o22, o23, o24, o25, o26, dst, per_port, d_out);
}


/* dst[d_out] = W[d_out x n_in] @ src[n_in]; W streams from external memory through the weight ports. */
static void matvec(int src, int dst, const llm_wvec_t *w, const llm_wvec_t *w1, const llm_wvec_t *w2, const llm_wvec_t *w3, const llm_wvec_t *w4, const llm_wvec_t *w5, const llm_wvec_t *w6, const llm_wvec_t *w7, const llm_wvec_t *w8, const llm_wvec_t *w9, const llm_wvec_t *w10, const llm_wvec_t *w11, const llm_wvec_t *w12, const llm_wvec_t *w13, const llm_wvec_t *w14, const llm_wvec_t *w15, const llm_wvec_t *w16, const llm_wvec_t *w17, const llm_wvec_t *w18, const llm_wvec_t *w19, const llm_wvec_t *w20, const llm_wvec_t *w21, const llm_wvec_t *w22, const llm_wvec_t *w23, const llm_wvec_t *w24, const llm_wvec_t *w25, const llm_wvec_t *w26,
                   uint64_t woff, uint64_t loff, uint64_t pb0, int n_in, int d_out) {
#pragma HLS INLINE off
    float vin0[MAX_COLS], vin1[MAX_COLS], vin2[MAX_COLS], vin3[MAX_COLS], vin4[MAX_COLS], vin5[MAX_COLS], vin6[MAX_COLS], vin7[MAX_COLS], vin8[MAX_COLS], vin9[MAX_COLS], vin10[MAX_COLS], vin11[MAX_COLS], vin12[MAX_COLS], vin13[MAX_COLS], vin14[MAX_COLS], vin15[MAX_COLS], vin16[MAX_COLS], vin17[MAX_COLS], vin18[MAX_COLS], vin19[MAX_COLS], vin20[MAX_COLS], vin21[MAX_COLS], vin22[MAX_COLS], vin23[MAX_COLS], vin24[MAX_COLS], vin25[MAX_COLS], vin26[MAX_COLS];
#pragma HLS ARRAY_PARTITION variable = vin0 cyclic factor = LLM_LANES
#pragma HLS ARRAY_PARTITION variable = vin1 cyclic factor = LLM_LANES
#pragma HLS ARRAY_PARTITION variable = vin2 cyclic factor = LLM_LANES
#pragma HLS ARRAY_PARTITION variable = vin3 cyclic factor = LLM_LANES
#pragma HLS ARRAY_PARTITION variable = vin4 cyclic factor = LLM_LANES
#pragma HLS ARRAY_PARTITION variable = vin5 cyclic factor = LLM_LANES
#pragma HLS ARRAY_PARTITION variable = vin6 cyclic factor = LLM_LANES
#pragma HLS ARRAY_PARTITION variable = vin7 cyclic factor = LLM_LANES
#pragma HLS ARRAY_PARTITION variable = vin8 cyclic factor = LLM_LANES
#pragma HLS ARRAY_PARTITION variable = vin9 cyclic factor = LLM_LANES
#pragma HLS ARRAY_PARTITION variable = vin10 cyclic factor = LLM_LANES
#pragma HLS ARRAY_PARTITION variable = vin11 cyclic factor = LLM_LANES
#pragma HLS ARRAY_PARTITION variable = vin12 cyclic factor = LLM_LANES
#pragma HLS ARRAY_PARTITION variable = vin13 cyclic factor = LLM_LANES
#pragma HLS ARRAY_PARTITION variable = vin14 cyclic factor = LLM_LANES
#pragma HLS ARRAY_PARTITION variable = vin15 cyclic factor = LLM_LANES
#pragma HLS ARRAY_PARTITION variable = vin16 cyclic factor = LLM_LANES
#pragma HLS ARRAY_PARTITION variable = vin17 cyclic factor = LLM_LANES
#pragma HLS ARRAY_PARTITION variable = vin18 cyclic factor = LLM_LANES
#pragma HLS ARRAY_PARTITION variable = vin19 cyclic factor = LLM_LANES
#pragma HLS ARRAY_PARTITION variable = vin20 cyclic factor = LLM_LANES
#pragma HLS ARRAY_PARTITION variable = vin21 cyclic factor = LLM_LANES
#pragma HLS ARRAY_PARTITION variable = vin22 cyclic factor = LLM_LANES
#pragma HLS ARRAY_PARTITION variable = vin23 cyclic factor = LLM_LANES
#pragma HLS ARRAY_PARTITION variable = vin24 cyclic factor = LLM_LANES
#pragma HLS ARRAY_PARTITION variable = vin25 cyclic factor = LLM_LANES
#pragma HLS ARRAY_PARTITION variable = vin26 cyclic factor = LLM_LANES
    /* LLM_LANES elements per clock cycle (was one): the loop was n_in cycles in front of every matrix, now n_in / LLM_LANES */
    for (int jw = 0; jw < n_in / LLM_LANES; jw++) {
#pragma HLS PIPELINE II = 1
        for (int l = 0; l < LLM_LANES; l++) {
#pragma HLS UNROLL
            const int j = jw * LLM_LANES + l;
            const float v = (src == SRC_XB) ? xb[j] : (src == SRC_HB) ? hb[j] : hvin_g[j];
            vin0[jw * LLM_LANES + l] = v;
            vin1[jw * LLM_LANES + l] = v;
            vin2[jw * LLM_LANES + l] = v;
            vin3[jw * LLM_LANES + l] = v;
            vin4[jw * LLM_LANES + l] = v;
            vin5[jw * LLM_LANES + l] = v;
            vin6[jw * LLM_LANES + l] = v;
            vin7[jw * LLM_LANES + l] = v;
            vin8[jw * LLM_LANES + l] = v;
            vin9[jw * LLM_LANES + l] = v;
            vin10[jw * LLM_LANES + l] = v;
            vin11[jw * LLM_LANES + l] = v;
            vin12[jw * LLM_LANES + l] = v;
            vin13[jw * LLM_LANES + l] = v;
            vin14[jw * LLM_LANES + l] = v;
            vin15[jw * LLM_LANES + l] = v;
            vin16[jw * LLM_LANES + l] = v;
            vin17[jw * LLM_LANES + l] = v;
            vin18[jw * LLM_LANES + l] = v;
            vin19[jw * LLM_LANES + l] = v;
            vin20[jw * LLM_LANES + l] = v;
            vin21[jw * LLM_LANES + l] = v;
            vin22[jw * LLM_LANES + l] = v;
            vin23[jw * LLM_LANES + l] = v;
            vin24[jw * LLM_LANES + l] = v;
            vin25[jw * LLM_LANES + l] = v;
            vin26[jw * LLM_LANES + l] = v;
        }
    }
    const MvPrep mp0 = mv_prep(woff, loff, pb0, n_in, d_out);
    matvec_flow(vin0, vin1, vin2, vin3, vin4, vin5, vin6, vin7, vin8, vin9, vin10, vin11, vin12, vin13, vin14, vin15, vin16, vin17, vin18, vin19, vin20, vin21, vin22, vin23, vin24, vin25, vin26, w, w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11, w12, w13, w14, w15, w16, w17, w18, w19, w20, w21, w22, w23, w24, w25, w26, mp0, mp0.n_in, mp0.d_out, dst);
}

static void softmax(float *v, int n) {
    float mx = v[0];
    for (int i = 1; i < n; i++) if (v[i] > mx) mx = v[i];
    VecSums acc;
#pragma HLS ARRAY_PARTITION variable = acc.t complete
    for (int i = 0; i < n; i++) {
#pragma HLS PIPELINE II = 1
#pragma HLS DEPENDENCE variable = acc.t type = inter direction = RAW dependent = true distance = LLM_VEC_SUMS
        const float e = expf(v[i] - mx);
        v[i] = e;
        acc.add(e);
    }
    const float sum = acc.total();
    for (int i = 0; i < n; i++) v[i] /= sum;
}

static void rope(float *vec, int n, int hs) {
    for (int i = 0; i < n; i += 2) {
        int f = (i % hs) / 2;
        float a = vec[i], b = vec[i + 1];
        vec[i] = a * ropec[f] - b * ropes[f];
        vec[i + 1] = a * ropes[f] + b * ropec[f];
    }
}

/* ---- the attention unit: one pass over all heads per earlier position t, LLM_LANES_ATT numbers per clock cycle.
 * Scores: the query (all heads, in q) times position t's keys (kr), with LLM_ACC_DEPTH_ATT running totals per lane in a shift register
 * as in the matrix unit; at the end of each head its totals are added (tree_sum) and divided by sqrt(head size). With grouped-query
 * attention every kv_mul heads share one key/value head. Each head's numbers are added in the same order as one head at a time.
 * This replaces one dot-product call per head, whose start-up and final sum cost far more than the multiplying for small heads. */
static void scores_all_heads(int t, const float *kr, int total, int hs, int kv_mul, float sq) {
    const int words = hs / LLM_LANES_ATT; /* total = heads * words, computed once by the caller (a multiplier here ended up in the address paths) */
    float part[LLM_ACC_DEPTH_ATT][LLM_LANES_ATT];
#pragma HLS ARRAY_PARTITION variable = part complete dim = 0
    for (int d = 0; d < LLM_ACC_DEPTH_ATT; d++)
        for (int l = 0; l < LLM_LANES_ATT; l++) part[d][l] = 0.0f;
    int j = 0, h = 0, qb = 0, kb = 0, kn = 0; /* word inside the head, head, where its query and keys start, heads on this key head */
    int sl = 0;                                /* j % LLM_ACC_DEPTH_ATT: the slot this word adds into (as in row_stream) */
    for (int w = 0; w < total; w++) {
#pragma HLS PIPELINE II = 1
#pragma HLS DEPENDENCE variable = part type = inter direction = RAW dependent = true distance = LLM_ACC_DEPTH_ATT
        for (int l = 0; l < LLM_LANES_ATT; l++) {
#pragma HLS UNROLL
            const float prod = q[qb + j * LLM_LANES_ATT + l] * kr[kb + j * LLM_LANES_ATT + l];
            const float oldest = (j < LLM_ACC_DEPTH_ATT) ? 0.0f : part[sl][l];
            part[sl][l] = oldest + prod;
        }
        if (j == words - 1) {
            float v[LLM_LANES_ATT * LLM_ACC_DEPTH_ATT];
#pragma HLS ARRAY_PARTITION variable = v complete
            for (int l = 0; l < LLM_LANES_ATT; l++)
                for (int e = 0; e < LLM_ACC_DEPTH_ATT; e++) v[l * LLM_ACC_DEPTH_ATT + e] = (words >= LLM_ACC_DEPTH_ATT || e < words) ? part[e][l] : 0.0f;
            att[h][t] = tree_sum<LLM_LANES_ATT * LLM_ACC_DEPTH_ATT>(v) / sq;
            h++;
            qb += hs;
            j = 0;
            sl = 0;
            if (++kn == kv_mul) { kn = 0; kb += hs; }
        } else {
            j++;
            sl = (sl + 1 == LLM_ACC_DEPTH_ATT) ? 0 : sl + 1;
        }
    }
}

/* Weighted sum of position t's values (vr) into every head's output (xb), all heads in one pass, LLM_LANES_ATT numbers per cycle. */
static void weighted_all_heads(int t, const float *vr, int total, int hs, int kv_mul) {
    const int words = hs / LLM_LANES_ATT;
    int j = 0, h = 0, ob = 0, vb = 0, kn = 0;
    for (int w = 0; w < total; w++) {
#pragma HLS PIPELINE II = 1
#pragma HLS DEPENDENCE variable = xb inter false
        const float a = att[h][t];
        for (int l = 0; l < LLM_LANES_ATT; l++) {
#pragma HLS UNROLL
            xb[ob + j * LLM_LANES_ATT + l] += a * vr[vb + j * LLM_LANES_ATT + l];
        }
        if (j == words - 1) {
            j = 0;
            h++;
            ob += hs;
            if (++kn == kv_mul) { kn = 0; vb += hs; }
        } else {
            j++;
        }
    }
}

/* One token through the whole model. Returns the greedy next token. */
static int forward(const Cfg &c, const llm_wvec_t *w, const llm_wvec_t *w1, const llm_wvec_t *w2, const llm_wvec_t *w3, const llm_wvec_t *w4, const llm_wvec_t *w5, const llm_wvec_t *w6, const llm_wvec_t *w7, const llm_wvec_t *w8, const llm_wvec_t *w9, const llm_wvec_t *w10, const llm_wvec_t *w11, const llm_wvec_t *w12, const llm_wvec_t *w13, const llm_wvec_t *w14, const llm_wvec_t *w15, const llm_wvec_t *w16, const llm_wvec_t *w17, const llm_wvec_t *w18, const llm_wvec_t *w19, const llm_wvec_t *w20, const llm_wvec_t *w21, const llm_wvec_t *w22, const llm_wvec_t *w23, const llm_wvec_t *w24, const llm_wvec_t *w25, const llm_wvec_t *w26,
                   llm_svec_t *state, int token, int pos) {
    const int dim = c.dim, hid = c.hid, hs = c.hs, kvd = c.kvd, seq = c.seq;
    const int att_total = c.heads * (c.hs / LLM_LANES_ATT); /* words of one position's K (or V) row, all heads */
#pragma HLS ALLOCATION function instances = matvec limit = 1 /* one matrix unit, used by every matrix multiplication in turn */
#pragma HLS ARRAY_PARTITION variable = kvrow cyclic factor = LLM_LANES_ATT
#pragma HLS ARRAY_PARTITION variable = q cyclic factor = LLM_LANES_ATT
#pragma HLS ARRAY_PARTITION variable = kvec cyclic factor = LLM_LANES_ATT
#pragma HLS ARRAY_PARTITION variable = vvec cyclic factor = LLM_LANES_ATT
#pragma HLS ARRAY_PARTITION variable = xb cyclic factor = LLM_LANES_ATT
#pragma HLS ARRAY_PARTITION variable = hb cyclic factor = LLM_LANES_ATT
#pragma HLS ARRAY_PARTITION variable = hvin_g cyclic factor = LLM_LANES_ATT
#pragma HLS ARRAY_PARTITION variable = head_res cyclic factor = LLM_LANES_ATT
#pragma HLS ARRAY_PARTITION variable = normw cyclic factor = LLM_LANES
#pragma HLS ARRAY_PARTITION variable = ropec cyclic factor = LLM_LANES
#pragma HLS ARRAY_PARTITION variable = ropes cyclic factor = LLM_LANES
    /* positions in the state area, in floats: logits first, then the K cache, then the V cache */
    const uint64_t kc = align_f((uint32_t)c.vocab);
    const uint64_t vc = kc + (uint64_t)c.layers * (uint64_t)seq * (uint64_t)kvd;

    load_vecw(x, w, c.tok + (uint64_t)token * (uint64_t)dim, dim);
    load_vecw_wide(ropec, w, c.cos_ + (uint64_t)pos * (uint64_t)(hs / 2), hs / 2);
    load_vecw_wide(ropes, w, c.sin_ + (uint64_t)pos * (uint64_t)(hs / 2), hs / 2);

    for (int l = 0; l < c.layers; l++) {
        const uint64_t lb = c.lay0 + (uint64_t)l * c.stride;
        const uint64_t row0 = ((uint64_t)l * (uint64_t)seq + (uint64_t)pos) * (uint64_t)kvd;
        const uint64_t ll = (uint64_t)l * c.lstride; /* this layer's block in a port's buffer (split layout) */

        load_vecw_wide(normw, w, c.split ? c.norms + (uint64_t)l * 2u * (uint64_t)dim : lb + c.rel[0], dim);
        rmsnorm(xb, x, normw, dim);
        matvec(SRC_XB, DST_Q, w, w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11, w12, w13, w14, w15, w16, w17, w18, w19, w20, w21, w22, w23, w24, w25, w26, lb + c.rel[1], c.split ? ll + c.prel[1] : NOSPLIT, c.pb0, dim, dim);
        matvec(SRC_XB, DST_KVEC, w, w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11, w12, w13, w14, w15, w16, w17, w18, w19, w20, w21, w22, w23, w24, w25, w26, lb + c.rel[2], c.split ? ll + c.prel[2] : NOSPLIT, c.pb0, dim, kvd);
        matvec(SRC_XB, DST_VVEC, w, w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11, w12, w13, w14, w15, w16, w17, w18, w19, w20, w21, w22, w23, w24, w25, w26, lb + c.rel[3], c.split ? ll + c.prel[3] : NOSPLIT, c.pb0, dim, kvd);
        rope(q, dim, hs);
        rope(kvec, kvd, hs);
        store_kv(state, kc + row0, kvec, kvd);
        store_kv(state, vc + row0, vvec, kvd);

        /* Attention. Each earlier position's keys (and later its values) are read once for all heads, in one burst of kvd floats:
         * every separate read waits for the DDR4 (about 100 cycles measured), so one read per position instead of one per head.
         * The scores and sums are the same as one head at a time, in the same order, bit for bit. */
        const float sq = sqrtf((float)hs);
        for (int t = 0; t <= pos; t++) {
            load_kv(kvrow, state, kc + ((uint64_t)l * seq + t) * kvd, kvd);
            scores_all_heads(t, kvrow, att_total, hs, c.kv_mul, sq);
        }
        for (int h = 0; h < c.heads; h++) softmax(att[h], pos + 1);
        for (int jw = 0; jw < dim / LLM_LANES_ATT; jw++) {
#pragma HLS PIPELINE II = 1
            for (int l = 0; l < LLM_LANES_ATT; l++) {
#pragma HLS UNROLL
                xb[jw * LLM_LANES_ATT + l] = 0.0f;
            }
        }
        for (int t = 0; t <= pos; t++) {
            load_kv(kvrow, state, vc + ((uint64_t)l * seq + t) * kvd, kvd);
            weighted_all_heads(t, kvrow, att_total, hs, c.kv_mul);
        }
        matvec(SRC_XB, DST_XB2, w, w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11, w12, w13, w14, w15, w16, w17, w18, w19, w20, w21, w22, w23, w24, w25, w26, lb + c.rel[4], c.split ? ll + c.prel[4] : NOSPLIT, c.pb0, dim, dim);
        for (int i = 0; i < dim; i++) x[i] += xb2[i];

        load_vecw_wide(normw, w, c.split ? c.norms + (uint64_t)l * 2u * (uint64_t)dim + (uint64_t)dim : lb + c.rel[5], dim);
        rmsnorm(xb, x, normw, dim);
        matvec(SRC_XB, DST_HB, w, w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11, w12, w13, w14, w15, w16, w17, w18, w19, w20, w21, w22, w23, w24, w25, w26, lb + c.rel[6], c.split ? ll + c.prel[6] : NOSPLIT, c.pb0, dim, hid);
        matvec(SRC_XB, DST_HB2, w, w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11, w12, w13, w14, w15, w16, w17, w18, w19, w20, w21, w22, w23, w24, w25, w26, lb + c.rel[8], c.split ? ll + c.prel[8] : NOSPLIT, c.pb0, dim, hid);
        matvec(SRC_HB, DST_XB, w, w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11, w12, w13, w14, w15, w16, w17, w18, w19, w20, w21, w22, w23, w24, w25, w26, lb + c.rel[7], c.split ? ll + c.prel[7] : NOSPLIT, c.pb0, hid, dim);
        for (int i = 0; i < dim; i++) x[i] += xb[i];
    }

    /* final norm into a separate buffer: the in-place form rmsnorm(x, x, ...) was mis-synthesized (head input shifted by
     * 8 elements in RTL; C sim was fine) */
    load_vecw_wide(normw, w, c.fin, dim);
    rmsnorm(xb, x, normw, dim);

    /* output head: stream every vocab row, write the logit to memory, track the argmax on the fly */
    const uint64_t wcls = c.shared ? c.tok : c.wcls;
    for (int j = 0; j < dim; j++) {
#pragma HLS PIPELINE II = 1
        hvin_g[j] = xb[j];
    }
    /* the vocabulary rows go through the matrix unit in chunks; each chunk's logits are written (one wide word at a time) and the
     * largest is tracked in vocabulary order, so the first of equal values wins */
    int best = 0;
    float best_val = 0.0f;
    float lbuf[LLM_LANES_ATT];
#pragma HLS ARRAY_PARTITION variable = lbuf complete
    for (int l = 0; l < LLM_LANES_ATT; l++) lbuf[l] = 0.0f;
    for (int v0 = 0; v0 < c.vocab; v0 += MAX_COLS) {
        const int n = (c.vocab - v0 < MAX_COLS) ? (c.vocab - v0) : MAX_COLS;
        matvec(SRC_HEAD, DST_HEAD, w, w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11, w12, w13, w14, w15, w16, w17, w18, w19, w20, w21, w22, w23, w24, w25, w26, wcls + (uint64_t)v0 * (uint64_t)dim,
               c.split ? (uint64_t)c.layers * c.lstride + (uint64_t)(v0 / MAX_COLS) * c.chunkstride : NOSPLIT, c.pb0, dim, n);
        /* LLM_LANES_ATT logits per iteration: one wide store, and the largest of the group by a tree in which the lower index wins ties,
         * then compared with the best so far (strictly greater replaces): the first of equal values wins, as in vocabulary order */
        const int nw = (n + LLM_LANES_ATT - 1) / LLM_LANES_ATT;
        for (int jw = 0; jw < nw; jw++) {
#pragma HLS PIPELINE II = 2
            float bv[LLM_LANES_ATT];
            int bi[LLM_LANES_ATT];
#pragma HLS ARRAY_PARTITION variable = bv complete
#pragma HLS ARRAY_PARTITION variable = bi complete
            llm_svec_t tv{};
            for (int l = 0; l < LLM_LANES_ATT; l++) {
#pragma HLS UNROLL
                const int i = jw * LLM_LANES_ATT + l;
                const float val = head_res[i];
                llm_set(tv, l, val);
                bv[l] = (i < n) ? val : -3.0e38f;
                bi[l] = v0 + i;
            }
            for (int step = 1; step < LLM_LANES_ATT; step *= 2) {
#pragma HLS UNROLL
                for (int i = 0; i + step < LLM_LANES_ATT; i += 2 * step) {
#pragma HLS UNROLL
                    if (bv[i + step] > bv[i]) { bv[i] = bv[i + step]; bi[i] = bi[i + step]; }
                }
            }
            state[v0 / LLM_LANES_ATT + jw] = tv;
            if ((v0 == 0 && jw == 0) || bv[0] > best_val) { best_val = bv[0]; best = bi[0]; }
        }
    }
    return best;
}

extern "C" int llm_top(const llm_wvec_t *weights, llm_svec_t *state, int *tokens, int n_prompt, int n_steps LLM_PORT_ARGS, int start_pos, int stop0, int stop1) {
#pragma HLS INTERFACE m_axi port = weights offset = slave bundle = gmem0 depth = ((LLM_DEPTH_WEIGHTS + LLM_LANES - 1) / LLM_LANES)
#pragma HLS INTERFACE m_axi port = state offset = slave bundle = gmem1 depth = ((LLM_DEPTH_STATE + LLM_LANES_ATT - 1) / LLM_LANES_ATT)
#pragma HLS INTERFACE m_axi port = tokens offset = slave bundle = gmem2 depth = LLM_DEPTH_TOKENS
#if LLM_PORTS >= 2
#pragma HLS INTERFACE m_axi port = weights1 offset = slave bundle = gmem3 depth = ((LLM_DEPTH_WEIGHTS + LLM_LANES - 1) / LLM_LANES)
#pragma HLS INTERFACE s_axilite port = weights1 bundle = control
#endif
#if LLM_PORTS >= 3
#pragma HLS INTERFACE m_axi port = weights2 offset = slave bundle = gmem4 depth = ((LLM_DEPTH_WEIGHTS + LLM_LANES - 1) / LLM_LANES)
#pragma HLS INTERFACE s_axilite port = weights2 bundle = control
#endif
#if LLM_PORTS >= 4
#pragma HLS INTERFACE m_axi port = weights3 offset = slave bundle = gmem5 depth = ((LLM_DEPTH_WEIGHTS + LLM_LANES - 1) / LLM_LANES)
#pragma HLS INTERFACE s_axilite port = weights3 bundle = control
#endif
#if LLM_PORTS >= 5
#pragma HLS INTERFACE m_axi port = weights4 offset = slave bundle = gmem6 depth = ((LLM_DEPTH_WEIGHTS + LLM_LANES - 1) / LLM_LANES)
#pragma HLS INTERFACE s_axilite port = weights4 bundle = control
#endif
#if LLM_PORTS >= 6
#pragma HLS INTERFACE m_axi port = weights5 offset = slave bundle = gmem7 depth = ((LLM_DEPTH_WEIGHTS + LLM_LANES - 1) / LLM_LANES)
#pragma HLS INTERFACE s_axilite port = weights5 bundle = control
#endif
#if LLM_PORTS >= 7
#pragma HLS INTERFACE m_axi port = weights6 offset = slave bundle = gmem8 depth = ((LLM_DEPTH_WEIGHTS + LLM_LANES - 1) / LLM_LANES)
#pragma HLS INTERFACE s_axilite port = weights6 bundle = control
#endif
#if LLM_PORTS >= 8
#pragma HLS INTERFACE m_axi port = weights7 offset = slave bundle = gmem9 depth = ((LLM_DEPTH_WEIGHTS + LLM_LANES - 1) / LLM_LANES)
#pragma HLS INTERFACE s_axilite port = weights7 bundle = control
#endif
#if LLM_PORTS >= 9
#pragma HLS INTERFACE m_axi port = weights8 offset = slave bundle = gmem10 depth = ((LLM_DEPTH_WEIGHTS + LLM_LANES - 1) / LLM_LANES)
#pragma HLS INTERFACE s_axilite port = weights8 bundle = control
#endif
#if LLM_PORTS >= 10
#pragma HLS INTERFACE m_axi port = weights9 offset = slave bundle = gmem11 depth = ((LLM_DEPTH_WEIGHTS + LLM_LANES - 1) / LLM_LANES)
#pragma HLS INTERFACE s_axilite port = weights9 bundle = control
#endif
#if LLM_PORTS >= 11
#pragma HLS INTERFACE m_axi port = weights10 offset = slave bundle = gmem12 depth = ((LLM_DEPTH_WEIGHTS + LLM_LANES - 1) / LLM_LANES)
#pragma HLS INTERFACE s_axilite port = weights10 bundle = control
#endif
#if LLM_PORTS >= 12
#pragma HLS INTERFACE m_axi port = weights11 offset = slave bundle = gmem13 depth = ((LLM_DEPTH_WEIGHTS + LLM_LANES - 1) / LLM_LANES)
#pragma HLS INTERFACE s_axilite port = weights11 bundle = control
#endif
#if LLM_PORTS >= 13
#pragma HLS INTERFACE m_axi port = weights12 offset = slave bundle = gmem14 depth = ((LLM_DEPTH_WEIGHTS + LLM_LANES - 1) / LLM_LANES)
#pragma HLS INTERFACE s_axilite port = weights12 bundle = control
#endif
#if LLM_PORTS >= 14
#pragma HLS INTERFACE m_axi port = weights13 offset = slave bundle = gmem15 depth = ((LLM_DEPTH_WEIGHTS + LLM_LANES - 1) / LLM_LANES)
#pragma HLS INTERFACE s_axilite port = weights13 bundle = control
#endif
#if LLM_PORTS >= 15
#pragma HLS INTERFACE m_axi port = weights14 offset = slave bundle = gmem16 depth = ((LLM_DEPTH_WEIGHTS + LLM_LANES - 1) / LLM_LANES)
#pragma HLS INTERFACE s_axilite port = weights14 bundle = control
#endif
#if LLM_PORTS >= 16
#pragma HLS INTERFACE m_axi port = weights15 offset = slave bundle = gmem17 depth = ((LLM_DEPTH_WEIGHTS + LLM_LANES - 1) / LLM_LANES)
#pragma HLS INTERFACE s_axilite port = weights15 bundle = control
#endif
#if LLM_PORTS >= 17
#pragma HLS INTERFACE m_axi port = weights16 offset = slave bundle = gmem18 depth = ((LLM_DEPTH_WEIGHTS + LLM_LANES - 1) / LLM_LANES)
#pragma HLS INTERFACE s_axilite port = weights16 bundle = control
#endif
#if LLM_PORTS >= 18
#pragma HLS INTERFACE m_axi port = weights17 offset = slave bundle = gmem19 depth = ((LLM_DEPTH_WEIGHTS + LLM_LANES - 1) / LLM_LANES)
#pragma HLS INTERFACE s_axilite port = weights17 bundle = control
#endif
#if LLM_PORTS >= 19
#pragma HLS INTERFACE m_axi port = weights18 offset = slave bundle = gmem20 depth = ((LLM_DEPTH_WEIGHTS + LLM_LANES - 1) / LLM_LANES)
#pragma HLS INTERFACE s_axilite port = weights18 bundle = control
#endif
#if LLM_PORTS >= 20
#pragma HLS INTERFACE m_axi port = weights19 offset = slave bundle = gmem21 depth = ((LLM_DEPTH_WEIGHTS + LLM_LANES - 1) / LLM_LANES)
#pragma HLS INTERFACE s_axilite port = weights19 bundle = control
#endif
#if LLM_PORTS >= 21
#pragma HLS INTERFACE m_axi port = weights20 offset = slave bundle = gmem22 depth = ((LLM_DEPTH_WEIGHTS + LLM_LANES - 1) / LLM_LANES)
#pragma HLS INTERFACE s_axilite port = weights20 bundle = control
#endif
#if LLM_PORTS >= 22
#pragma HLS INTERFACE m_axi port = weights21 offset = slave bundle = gmem23 depth = ((LLM_DEPTH_WEIGHTS + LLM_LANES - 1) / LLM_LANES)
#pragma HLS INTERFACE s_axilite port = weights21 bundle = control
#endif
#if LLM_PORTS >= 23
#pragma HLS INTERFACE m_axi port = weights22 offset = slave bundle = gmem24 depth = ((LLM_DEPTH_WEIGHTS + LLM_LANES - 1) / LLM_LANES)
#pragma HLS INTERFACE s_axilite port = weights22 bundle = control
#endif
#if LLM_PORTS >= 24
#pragma HLS INTERFACE m_axi port = weights23 offset = slave bundle = gmem25 depth = ((LLM_DEPTH_WEIGHTS + LLM_LANES - 1) / LLM_LANES)
#pragma HLS INTERFACE s_axilite port = weights23 bundle = control
#endif
#if LLM_PORTS >= 25
#pragma HLS INTERFACE m_axi port = weights24 offset = slave bundle = gmem26 depth = ((LLM_DEPTH_WEIGHTS + LLM_LANES - 1) / LLM_LANES)
#pragma HLS INTERFACE s_axilite port = weights24 bundle = control
#endif
#if LLM_PORTS >= 26
#pragma HLS INTERFACE m_axi port = weights25 offset = slave bundle = gmem27 depth = ((LLM_DEPTH_WEIGHTS + LLM_LANES - 1) / LLM_LANES)
#pragma HLS INTERFACE s_axilite port = weights25 bundle = control
#endif
#if LLM_PORTS >= 27
#pragma HLS INTERFACE m_axi port = weights26 offset = slave bundle = gmem28 depth = ((LLM_DEPTH_WEIGHTS + LLM_LANES - 1) / LLM_LANES)
#pragma HLS INTERFACE s_axilite port = weights26 bundle = control
#endif
#pragma HLS INTERFACE s_axilite port = weights bundle = control
#pragma HLS INTERFACE s_axilite port = state bundle = control
#pragma HLS INTERFACE s_axilite port = tokens bundle = control
#pragma HLS INTERFACE s_axilite port = n_prompt bundle = control
#pragma HLS INTERFACE s_axilite port = n_steps bundle = control
#pragma HLS INTERFACE s_axilite port = start_pos bundle = control
#pragma HLS INTERFACE s_axilite port = stop0 bundle = control
#pragma HLS INTERFACE s_axilite port = stop1 bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control

#if LLM_PORTS < 2
    const llm_wvec_t *weights1 = weights;
#endif
#if LLM_PORTS < 3
    const llm_wvec_t *weights2 = weights;
#endif
#if LLM_PORTS < 4
    const llm_wvec_t *weights3 = weights;
#endif
#if LLM_PORTS < 5
    const llm_wvec_t *weights4 = weights;
#endif
#if LLM_PORTS < 6
    const llm_wvec_t *weights5 = weights;
#endif
#if LLM_PORTS < 7
    const llm_wvec_t *weights6 = weights;
#endif
#if LLM_PORTS < 8
    const llm_wvec_t *weights7 = weights;
#endif
#if LLM_PORTS < 9
    const llm_wvec_t *weights8 = weights;
#endif
#if LLM_PORTS < 10
    const llm_wvec_t *weights9 = weights;
#endif
#if LLM_PORTS < 11
    const llm_wvec_t *weights10 = weights;
#endif
#if LLM_PORTS < 12
    const llm_wvec_t *weights11 = weights;
#endif
#if LLM_PORTS < 13
    const llm_wvec_t *weights12 = weights;
#endif
#if LLM_PORTS < 14
    const llm_wvec_t *weights13 = weights;
#endif
#if LLM_PORTS < 15
    const llm_wvec_t *weights14 = weights;
#endif
#if LLM_PORTS < 16
    const llm_wvec_t *weights15 = weights;
#endif
#if LLM_PORTS < 17
    const llm_wvec_t *weights16 = weights;
#endif
#if LLM_PORTS < 18
    const llm_wvec_t *weights17 = weights;
#endif
#if LLM_PORTS < 19
    const llm_wvec_t *weights18 = weights;
#endif
#if LLM_PORTS < 20
    const llm_wvec_t *weights19 = weights;
#endif
#if LLM_PORTS < 21
    const llm_wvec_t *weights20 = weights;
#endif
#if LLM_PORTS < 22
    const llm_wvec_t *weights21 = weights;
#endif
#if LLM_PORTS < 23
    const llm_wvec_t *weights22 = weights;
#endif
#if LLM_PORTS < 24
    const llm_wvec_t *weights23 = weights;
#endif
#if LLM_PORTS < 25
    const llm_wvec_t *weights24 = weights;
#endif
#if LLM_PORTS < 26
    const llm_wvec_t *weights25 = weights;
#endif
#if LLM_PORTS < 27
    const llm_wvec_t *weights26 = weights;
#endif
    Cfg c;
    int rc = load_cfg(weights, c);
    if (rc) return rc;
    const int total = n_prompt + n_steps;
    if (n_prompt < 1 || n_steps < 0 || total - 1 > c.seq || start_pos < 0 || start_pos > n_prompt - 1) return -7;

    int generated = 0;
    for (int pos = start_pos; pos < total - 1; pos++) { /* positions below start_pos are already in the K/V cache (state) from an earlier run on the same tokens */
        int tok = (pos < n_prompt) ? tokens[pos] : generated;
        int next = forward(c, weights, weights1, weights2, weights3, weights4, weights5, weights6, weights7, weights8, weights9, weights10, weights11, weights12, weights13, weights14, weights15, weights16, weights17, weights18, weights19, weights20, weights21, weights22, weights23, weights24, weights25, weights26, state, tok, pos);
        if (pos >= n_prompt - 1) {
            tokens[pos + 1] = next;
            generated = next;
            if ((next == stop0 || next == stop1)) break; /* a generated stop token ends the reply early; it is stored but has no K/V */
        }
    }
    return 0;
}
