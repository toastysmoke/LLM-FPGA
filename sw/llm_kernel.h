/* Kernel-shaped LLM inference: the part that will be synthesized with Vitis HLS.
 *
 * Memory map (all external memory - DDR4 on a development board, HBM on an Alveo):
 *   weights : the flat weight file (ref/layout.py), read-only, 64-byte aligned
 *   state   : float32. [ logits (vocab, rounded up to 16) | K cache | V cache ]
 *             K cache / V cache are each [n_layers][seq_len][kv_dim]
 *   tokens  : int32[seq_len + 1]. Host writes the prompt in tokens[0..n_prompt); the kernel writes each generated
 *             token after it. Total length n_prompt + n_steps must be <= seq_len + 1.
 *
 * Everything else lives in small on-chip arrays sized by the LLM_MAX_* limits below (override with -D).
 * start_pos: the first position the kernel computes (0 = a fresh run). The K/V entries of positions 0 .. start_pos-1 must already be in the
 * state area, written by an earlier run on the SAME tokens (0 <= start_pos <= n_prompt-1); this is how a chat continues without re-reading
 * the earlier conversation.
 * Return value: 0 = ok, negative = error (see llm_kernel.cpp).
 */
#ifndef LLM_KERNEL_H
#define LLM_KERNEL_H

#include <stdint.h>

#ifndef LLM_MAX_DIM
#define LLM_MAX_DIM 288      /* covers stories260K and stories15M */
#endif
#ifndef LLM_MAX_HIDDEN
#define LLM_MAX_HIDDEN 768
#endif
#ifndef LLM_MAX_SEQ
#define LLM_MAX_SEQ 512
#endif
#ifndef LLM_MAX_HEADS
#define LLM_MAX_HEADS 16
#endif
#ifndef LLM_MAX_HEAD_SIZE
#define LLM_MAX_HEAD_SIZE 64
#endif

/* How many weights the dot products multiply per clock cycle (fixed when the hardware is built).
 * 1 = one at a time, exactly as before. */
#ifndef LLM_LANES
#define LLM_LANES 1
#endif
/* Running totals each lane rotates between. A floating-point add takes about 4 clock cycles, so one running total can only take a new
 * value every 4 cycles; rotating between 4 or more lets the hardware start a new add every cycle. 1 = a single total, as before. */
#ifndef LLM_ACC_DEPTH
#define LLM_ACC_DEPTH 1
#endif
/* Weight ports: how many memory ports the weights are read through, each LLM_LANES lanes wide. The matrix rows are shared out between the
 * ports (each port streams its own block of rows, in parallel), so the matrix unit has LLM_PORTS * LLM_LANES lanes in all and reads
 * LLM_PORTS * LLM_LANES floats per clock cycle. 1 = the single weight port, as before. Every extra port is an extra argument of llm_top
 * (weights1 ... weights26): the host gives each the address of the same weight file (in the memory that port is wired to). */
#ifndef LLM_PORTS
#define LLM_PORTS 1
#endif
#if LLM_PORTS < 1 || LLM_PORTS > 27
#error "LLM_PORTS must be 1 to 27"
#endif

/* The same two settings for the attention dot products (query x keys), which are a separate, smaller unit. They default to the matrix
 * settings above; kernel_plan.py sets them from the model's ratio of attention work to matrix work. */
#ifndef LLM_LANES_ATT
#define LLM_LANES_ATT LLM_LANES
#endif
/* Running totals for the sums inside the RMSNorm (sum of squares) and the softmax (sum of exponentials). One total is the old order of
 * additions, bit for bit, and takes 3 clock cycles per element because each add waits for the one before; 4 or more lets the loop take one
 * element per cycle (the sums then differ from the old ones in the last digits). */
#ifndef LLM_VEC_SUMS
#define LLM_VEC_SUMS 1
#endif
#if LLM_VEC_SUMS < 1
#error "LLM_VEC_SUMS must be at least 1"
#endif
#ifndef LLM_ACC_DEPTH_ATT
#define LLM_ACC_DEPTH_ATT LLM_ACC_DEPTH
#endif

#if LLM_LANES < 1 || (16 % LLM_LANES) != 0
#error "LLM_LANES must be 1, 2, 4, 8 or 16 (it has to divide the 16-float section alignment of the weight file)"
#endif
#if LLM_LANES_ATT < 1 || (16 % LLM_LANES_ATT) != 0 || LLM_ACC_DEPTH_ATT < 1
#error "LLM_LANES_ATT must be 1, 2, 4, 8 or 16"
#endif

/* The weights port is LLM_LANES floats wide (32 * LLM_LANES bits), so a row of weights arrives at LLM_LANES per clock cycle.
 * The weight file is read in units of this type; the host passes the same address as before. */
/* The state port (logits and the K/V cache) is LLM_LANES_ATT floats wide for the same reason: the attention unit reads a key or value
 * slice LLM_LANES_ATT floats per clock cycle. The state area starts on a 64-byte boundary.
 *
 * In hardware synthesis a wide word is one bit vector (ap_uint), so each access is a single read or write of the whole word; on the PC it
 * is a plain struct of floats (define LLM_WIDE_AP to use the bit vector there too, which needs the Vitis include folder). Read and write
 * the floats of a word with llm_get / llm_set. */
#if defined(__SYNTHESIS__) || defined(LLM_WIDE_AP)
#include <ap_int.h>
typedef ap_uint<32 * LLM_LANES> llm_wvec_t;
typedef ap_uint<32 * LLM_LANES_ATT> llm_svec_t;
template <typename W> static inline float llm_get(const W &t, int l) {
    union { uint32_t u; float f; } c;
    c.u = (uint32_t)t.range(32 * l + 31, 32 * l);
    return c.f;
}
template <typename W> static inline void llm_set(W &t, int l, float f) {
    union { uint32_t u; float f; } c;
    c.f = f;
    t.range(32 * l + 31, 32 * l) = c.u;
}
#else
typedef struct { float v[LLM_LANES]; } llm_wvec_t;
typedef struct { float v[LLM_LANES_ATT]; } llm_svec_t;
template <typename W> static inline float llm_get(const W &t, int l) { return t.v[l]; }
template <typename W> static inline void llm_set(W &t, int l, float f) { t.v[l] = f; }
#endif

#ifdef __cplusplus
extern "C" {
#endif

#if LLM_PORTS == 1
#define LLM_PORT_ARGS
#define LLM_PORT_PASS(w)
#elif LLM_PORTS == 2
#define LLM_PORT_ARGS , const llm_wvec_t *weights1
#define LLM_PORT_PASS(w) , (w)
#elif LLM_PORTS == 3
#define LLM_PORT_ARGS , const llm_wvec_t *weights1, const llm_wvec_t *weights2
#define LLM_PORT_PASS(w) , (w), (w)
#elif LLM_PORTS == 4
#define LLM_PORT_ARGS , const llm_wvec_t *weights1, const llm_wvec_t *weights2, const llm_wvec_t *weights3
#define LLM_PORT_PASS(w) , (w), (w), (w)
#elif LLM_PORTS == 5
#define LLM_PORT_ARGS , const llm_wvec_t *weights1, const llm_wvec_t *weights2, const llm_wvec_t *weights3, const llm_wvec_t *weights4
#define LLM_PORT_PASS(w) , (w), (w), (w), (w)
#elif LLM_PORTS == 6
#define LLM_PORT_ARGS , const llm_wvec_t *weights1, const llm_wvec_t *weights2, const llm_wvec_t *weights3, const llm_wvec_t *weights4, const llm_wvec_t *weights5
#define LLM_PORT_PASS(w) , (w), (w), (w), (w), (w)
#elif LLM_PORTS == 7
#define LLM_PORT_ARGS , const llm_wvec_t *weights1, const llm_wvec_t *weights2, const llm_wvec_t *weights3, const llm_wvec_t *weights4, const llm_wvec_t *weights5, const llm_wvec_t *weights6
#define LLM_PORT_PASS(w) , (w), (w), (w), (w), (w), (w)
#elif LLM_PORTS == 8
#define LLM_PORT_ARGS , const llm_wvec_t *weights1, const llm_wvec_t *weights2, const llm_wvec_t *weights3, const llm_wvec_t *weights4, const llm_wvec_t *weights5, const llm_wvec_t *weights6, const llm_wvec_t *weights7
#define LLM_PORT_PASS(w) , (w), (w), (w), (w), (w), (w), (w)
#elif LLM_PORTS == 9
#define LLM_PORT_ARGS , const llm_wvec_t *weights1, const llm_wvec_t *weights2, const llm_wvec_t *weights3, const llm_wvec_t *weights4, const llm_wvec_t *weights5, const llm_wvec_t *weights6, const llm_wvec_t *weights7, const llm_wvec_t *weights8
#define LLM_PORT_PASS(w) , (w), (w), (w), (w), (w), (w), (w), (w)
#elif LLM_PORTS == 10
#define LLM_PORT_ARGS , const llm_wvec_t *weights1, const llm_wvec_t *weights2, const llm_wvec_t *weights3, const llm_wvec_t *weights4, const llm_wvec_t *weights5, const llm_wvec_t *weights6, const llm_wvec_t *weights7, const llm_wvec_t *weights8, const llm_wvec_t *weights9
#define LLM_PORT_PASS(w) , (w), (w), (w), (w), (w), (w), (w), (w), (w)
#elif LLM_PORTS == 11
#define LLM_PORT_ARGS , const llm_wvec_t *weights1, const llm_wvec_t *weights2, const llm_wvec_t *weights3, const llm_wvec_t *weights4, const llm_wvec_t *weights5, const llm_wvec_t *weights6, const llm_wvec_t *weights7, const llm_wvec_t *weights8, const llm_wvec_t *weights9, const llm_wvec_t *weights10
#define LLM_PORT_PASS(w) , (w), (w), (w), (w), (w), (w), (w), (w), (w), (w)
#elif LLM_PORTS == 12
#define LLM_PORT_ARGS , const llm_wvec_t *weights1, const llm_wvec_t *weights2, const llm_wvec_t *weights3, const llm_wvec_t *weights4, const llm_wvec_t *weights5, const llm_wvec_t *weights6, const llm_wvec_t *weights7, const llm_wvec_t *weights8, const llm_wvec_t *weights9, const llm_wvec_t *weights10, const llm_wvec_t *weights11
#define LLM_PORT_PASS(w) , (w), (w), (w), (w), (w), (w), (w), (w), (w), (w), (w)
#elif LLM_PORTS == 13
#define LLM_PORT_ARGS , const llm_wvec_t *weights1, const llm_wvec_t *weights2, const llm_wvec_t *weights3, const llm_wvec_t *weights4, const llm_wvec_t *weights5, const llm_wvec_t *weights6, const llm_wvec_t *weights7, const llm_wvec_t *weights8, const llm_wvec_t *weights9, const llm_wvec_t *weights10, const llm_wvec_t *weights11, const llm_wvec_t *weights12
#define LLM_PORT_PASS(w) , (w), (w), (w), (w), (w), (w), (w), (w), (w), (w), (w), (w)
#elif LLM_PORTS == 14
#define LLM_PORT_ARGS , const llm_wvec_t *weights1, const llm_wvec_t *weights2, const llm_wvec_t *weights3, const llm_wvec_t *weights4, const llm_wvec_t *weights5, const llm_wvec_t *weights6, const llm_wvec_t *weights7, const llm_wvec_t *weights8, const llm_wvec_t *weights9, const llm_wvec_t *weights10, const llm_wvec_t *weights11, const llm_wvec_t *weights12, const llm_wvec_t *weights13
#define LLM_PORT_PASS(w) , (w), (w), (w), (w), (w), (w), (w), (w), (w), (w), (w), (w), (w)
#elif LLM_PORTS == 15
#define LLM_PORT_ARGS , const llm_wvec_t *weights1, const llm_wvec_t *weights2, const llm_wvec_t *weights3, const llm_wvec_t *weights4, const llm_wvec_t *weights5, const llm_wvec_t *weights6, const llm_wvec_t *weights7, const llm_wvec_t *weights8, const llm_wvec_t *weights9, const llm_wvec_t *weights10, const llm_wvec_t *weights11, const llm_wvec_t *weights12, const llm_wvec_t *weights13, const llm_wvec_t *weights14
#define LLM_PORT_PASS(w) , (w), (w), (w), (w), (w), (w), (w), (w), (w), (w), (w), (w), (w), (w)
#elif LLM_PORTS == 16
#define LLM_PORT_ARGS , const llm_wvec_t *weights1, const llm_wvec_t *weights2, const llm_wvec_t *weights3, const llm_wvec_t *weights4, const llm_wvec_t *weights5, const llm_wvec_t *weights6, const llm_wvec_t *weights7, const llm_wvec_t *weights8, const llm_wvec_t *weights9, const llm_wvec_t *weights10, const llm_wvec_t *weights11, const llm_wvec_t *weights12, const llm_wvec_t *weights13, const llm_wvec_t *weights14, const llm_wvec_t *weights15
#define LLM_PORT_PASS(w) , (w), (w), (w), (w), (w), (w), (w), (w), (w), (w), (w), (w), (w), (w), (w)
#elif LLM_PORTS == 17
#define LLM_PORT_ARGS , const llm_wvec_t *weights1, const llm_wvec_t *weights2, const llm_wvec_t *weights3, const llm_wvec_t *weights4, const llm_wvec_t *weights5, const llm_wvec_t *weights6, const llm_wvec_t *weights7, const llm_wvec_t *weights8, const llm_wvec_t *weights9, const llm_wvec_t *weights10, const llm_wvec_t *weights11, const llm_wvec_t *weights12, const llm_wvec_t *weights13, const llm_wvec_t *weights14, const llm_wvec_t *weights15, const llm_wvec_t *weights16
#define LLM_PORT_PASS(w) , (w), (w), (w), (w), (w), (w), (w), (w), (w), (w), (w), (w), (w), (w), (w), (w)
#elif LLM_PORTS == 18
#define LLM_PORT_ARGS , const llm_wvec_t *weights1, const llm_wvec_t *weights2, const llm_wvec_t *weights3, const llm_wvec_t *weights4, const llm_wvec_t *weights5, const llm_wvec_t *weights6, const llm_wvec_t *weights7, const llm_wvec_t *weights8, const llm_wvec_t *weights9, const llm_wvec_t *weights10, const llm_wvec_t *weights11, const llm_wvec_t *weights12, const llm_wvec_t *weights13, const llm_wvec_t *weights14, const llm_wvec_t *weights15, const llm_wvec_t *weights16, const llm_wvec_t *weights17
#define LLM_PORT_PASS(w) , (w), (w), (w), (w), (w), (w), (w), (w), (w), (w), (w), (w), (w), (w), (w), (w), (w)
#elif LLM_PORTS == 19
#define LLM_PORT_ARGS , const llm_wvec_t *weights1, const llm_wvec_t *weights2, const llm_wvec_t *weights3, const llm_wvec_t *weights4, const llm_wvec_t *weights5, const llm_wvec_t *weights6, const llm_wvec_t *weights7, const llm_wvec_t *weights8, const llm_wvec_t *weights9, const llm_wvec_t *weights10, const llm_wvec_t *weights11, const llm_wvec_t *weights12, const llm_wvec_t *weights13, const llm_wvec_t *weights14, const llm_wvec_t *weights15, const llm_wvec_t *weights16, const llm_wvec_t *weights17, const llm_wvec_t *weights18
#define LLM_PORT_PASS(w) , (w), (w), (w), (w), (w), (w), (w), (w), (w), (w), (w), (w), (w), (w), (w), (w), (w), (w)
#elif LLM_PORTS == 20
#define LLM_PORT_ARGS , const llm_wvec_t *weights1, const llm_wvec_t *weights2, const llm_wvec_t *weights3, const llm_wvec_t *weights4, const llm_wvec_t *weights5, const llm_wvec_t *weights6, const llm_wvec_t *weights7, const llm_wvec_t *weights8, const llm_wvec_t *weights9, const llm_wvec_t *weights10, const llm_wvec_t *weights11, const llm_wvec_t *weights12, const llm_wvec_t *weights13, const llm_wvec_t *weights14, const llm_wvec_t *weights15, const llm_wvec_t *weights16, const llm_wvec_t *weights17, const llm_wvec_t *weights18, const llm_wvec_t *weights19
#define LLM_PORT_PASS(w) , (w), (w), (w), (w), (w), (w), (w), (w), (w), (w), (w), (w), (w), (w), (w), (w), (w), (w), (w)
#elif LLM_PORTS == 21
#define LLM_PORT_ARGS , const llm_wvec_t *weights1, const llm_wvec_t *weights2, const llm_wvec_t *weights3, const llm_wvec_t *weights4, const llm_wvec_t *weights5, const llm_wvec_t *weights6, const llm_wvec_t *weights7, const llm_wvec_t *weights8, const llm_wvec_t *weights9, const llm_wvec_t *weights10, const llm_wvec_t *weights11, const llm_wvec_t *weights12, const llm_wvec_t *weights13, const llm_wvec_t *weights14, const llm_wvec_t *weights15, const llm_wvec_t *weights16, const llm_wvec_t *weights17, const llm_wvec_t *weights18, const llm_wvec_t *weights19, const llm_wvec_t *weights20
#define LLM_PORT_PASS(w) , (w), (w), (w), (w), (w), (w), (w), (w), (w), (w), (w), (w), (w), (w), (w), (w), (w), (w), (w), (w)
#elif LLM_PORTS == 22
#define LLM_PORT_ARGS , const llm_wvec_t *weights1, const llm_wvec_t *weights2, const llm_wvec_t *weights3, const llm_wvec_t *weights4, const llm_wvec_t *weights5, const llm_wvec_t *weights6, const llm_wvec_t *weights7, const llm_wvec_t *weights8, const llm_wvec_t *weights9, const llm_wvec_t *weights10, const llm_wvec_t *weights11, const llm_wvec_t *weights12, const llm_wvec_t *weights13, const llm_wvec_t *weights14, const llm_wvec_t *weights15, const llm_wvec_t *weights16, const llm_wvec_t *weights17, const llm_wvec_t *weights18, const llm_wvec_t *weights19, const llm_wvec_t *weights20, const llm_wvec_t *weights21
#define LLM_PORT_PASS(w) , (w), (w), (w), (w), (w), (w), (w), (w), (w), (w), (w), (w), (w), (w), (w), (w), (w), (w), (w), (w), (w)
#elif LLM_PORTS == 23
#define LLM_PORT_ARGS , const llm_wvec_t *weights1, const llm_wvec_t *weights2, const llm_wvec_t *weights3, const llm_wvec_t *weights4, const llm_wvec_t *weights5, const llm_wvec_t *weights6, const llm_wvec_t *weights7, const llm_wvec_t *weights8, const llm_wvec_t *weights9, const llm_wvec_t *weights10, const llm_wvec_t *weights11, const llm_wvec_t *weights12, const llm_wvec_t *weights13, const llm_wvec_t *weights14, const llm_wvec_t *weights15, const llm_wvec_t *weights16, const llm_wvec_t *weights17, const llm_wvec_t *weights18, const llm_wvec_t *weights19, const llm_wvec_t *weights20, const llm_wvec_t *weights21, const llm_wvec_t *weights22
#define LLM_PORT_PASS(w) , (w), (w), (w), (w), (w), (w), (w), (w), (w), (w), (w), (w), (w), (w), (w), (w), (w), (w), (w), (w), (w), (w)
#elif LLM_PORTS == 24
#define LLM_PORT_ARGS , const llm_wvec_t *weights1, const llm_wvec_t *weights2, const llm_wvec_t *weights3, const llm_wvec_t *weights4, const llm_wvec_t *weights5, const llm_wvec_t *weights6, const llm_wvec_t *weights7, const llm_wvec_t *weights8, const llm_wvec_t *weights9, const llm_wvec_t *weights10, const llm_wvec_t *weights11, const llm_wvec_t *weights12, const llm_wvec_t *weights13, const llm_wvec_t *weights14, const llm_wvec_t *weights15, const llm_wvec_t *weights16, const llm_wvec_t *weights17, const llm_wvec_t *weights18, const llm_wvec_t *weights19, const llm_wvec_t *weights20, const llm_wvec_t *weights21, const llm_wvec_t *weights22, const llm_wvec_t *weights23
#define LLM_PORT_PASS(w) , (w), (w), (w), (w), (w), (w), (w), (w), (w), (w), (w), (w), (w), (w), (w), (w), (w), (w), (w), (w), (w), (w), (w)
#elif LLM_PORTS == 25
#define LLM_PORT_ARGS , const llm_wvec_t *weights1, const llm_wvec_t *weights2, const llm_wvec_t *weights3, const llm_wvec_t *weights4, const llm_wvec_t *weights5, const llm_wvec_t *weights6, const llm_wvec_t *weights7, const llm_wvec_t *weights8, const llm_wvec_t *weights9, const llm_wvec_t *weights10, const llm_wvec_t *weights11, const llm_wvec_t *weights12, const llm_wvec_t *weights13, const llm_wvec_t *weights14, const llm_wvec_t *weights15, const llm_wvec_t *weights16, const llm_wvec_t *weights17, const llm_wvec_t *weights18, const llm_wvec_t *weights19, const llm_wvec_t *weights20, const llm_wvec_t *weights21, const llm_wvec_t *weights22, const llm_wvec_t *weights23, const llm_wvec_t *weights24
#define LLM_PORT_PASS(w) , (w), (w), (w), (w), (w), (w), (w), (w), (w), (w), (w), (w), (w), (w), (w), (w), (w), (w), (w), (w), (w), (w), (w), (w)
#elif LLM_PORTS == 26
#define LLM_PORT_ARGS , const llm_wvec_t *weights1, const llm_wvec_t *weights2, const llm_wvec_t *weights3, const llm_wvec_t *weights4, const llm_wvec_t *weights5, const llm_wvec_t *weights6, const llm_wvec_t *weights7, const llm_wvec_t *weights8, const llm_wvec_t *weights9, const llm_wvec_t *weights10, const llm_wvec_t *weights11, const llm_wvec_t *weights12, const llm_wvec_t *weights13, const llm_wvec_t *weights14, const llm_wvec_t *weights15, const llm_wvec_t *weights16, const llm_wvec_t *weights17, const llm_wvec_t *weights18, const llm_wvec_t *weights19, const llm_wvec_t *weights20, const llm_wvec_t *weights21, const llm_wvec_t *weights22, const llm_wvec_t *weights23, const llm_wvec_t *weights24, const llm_wvec_t *weights25
#define LLM_PORT_PASS(w) , (w), (w), (w), (w), (w), (w), (w), (w), (w), (w), (w), (w), (w), (w), (w), (w), (w), (w), (w), (w), (w), (w), (w), (w), (w)
#else
#define LLM_PORT_ARGS , const llm_wvec_t *weights1, const llm_wvec_t *weights2, const llm_wvec_t *weights3, const llm_wvec_t *weights4, const llm_wvec_t *weights5, const llm_wvec_t *weights6, const llm_wvec_t *weights7, const llm_wvec_t *weights8, const llm_wvec_t *weights9, const llm_wvec_t *weights10, const llm_wvec_t *weights11, const llm_wvec_t *weights12, const llm_wvec_t *weights13, const llm_wvec_t *weights14, const llm_wvec_t *weights15, const llm_wvec_t *weights16, const llm_wvec_t *weights17, const llm_wvec_t *weights18, const llm_wvec_t *weights19, const llm_wvec_t *weights20, const llm_wvec_t *weights21, const llm_wvec_t *weights22, const llm_wvec_t *weights23, const llm_wvec_t *weights24, const llm_wvec_t *weights25, const llm_wvec_t *weights26
#define LLM_PORT_PASS(w) , (w), (w), (w), (w), (w), (w), (w), (w), (w), (w), (w), (w), (w), (w), (w), (w), (w), (w), (w), (w), (w), (w), (w), (w), (w), (w)
#endif

int llm_top(const llm_wvec_t *weights, llm_svec_t *state, int *tokens, int n_prompt, int n_steps LLM_PORT_ARGS, int start_pos, int stop0, int stop1);

#ifdef __cplusplus
}
#endif

/* Host-side helpers: sizes of the state and token regions the kernel expects. */
static inline uint32_t llm_state_floats(int vocab, int n_layers, int seq_len, int kv_dim) {
    return (uint32_t)((vocab + 15) / 16 * 16) + 2u * (uint32_t)n_layers * (uint32_t)seq_len * (uint32_t)kv_dim;
}
static inline uint32_t llm_token_ints(int seq_len) { return (uint32_t)seq_len + 1u; }

#endif
