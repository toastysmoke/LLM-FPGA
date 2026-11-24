# How the kernel works

This is a guided tour of `sw/llm_kernel.cpp` (the HLS kernel `llm_top`). The file itself is heavily commented; this page gives the order to
read it in and the ideas that connect the parts. `sw/llm_kernel.h` documents the memory map and every build setting.

## 1. One call, one conversation turn

The host calls `llm_top(weights..., state, tokens, n_prompt, n_steps, start_pos, stop0, stop1)`.

- `tokens` holds the prompt; the kernel writes each generated token after it.
- `state` is the K/V cache plus the logits of the last position. It stays in HBM between calls.
- `start_pos` lets a chat continue: positions below it are already in the K/V cache, so only new tokens are computed.
- `stop0`, `stop1` are end-of-turn token ids; generation stops at the first one.
- The return value is 0, or a negative error code (listed at the top of `llm_kernel.cpp`).

`llm_top` loops over positions and calls `forward()` once per position. It is greedy (argmax), so output is deterministic.

## 2. `forward()`: one position through the model

For each of the 16 layers: RMSNorm, then the Q, K and V matrix-vector products, RoPE, writing K and V into the cache, attention over all
earlier positions, the output projection plus residual, RMSNorm, the SwiGLU feed-forward (gate, up, down) plus residual. After the last
layer: final RMSNorm and the output head, which scores the whole vocabulary and keeps the largest logit (first one wins ties).

Almost all the time and almost all the memory traffic is in the matrix-vector products. Attention reads the K/V cache once per position,
for all heads together (one burst instead of one read per head).

## 3. The matrix unit (`matvec`, `matvec_flow`, `row_stream`, `row_sum`, `merge_out`)

This is the part that decides the speed. One piece of hardware is shared by every matrix product in turn (`forward` limits `matvec` to one
instance), selected by the `src` and `dst` numbers.

```
               input vector (on chip, copied to every port)
                      |
   +------------------+-------------------+----- ... ----+
   |                  |                   |              |
 port 0             port 1              port 2   ...   port 26      each: a 256-bit m_axi read of its own HBM channel
 row_stream         row_stream          row_stream                  8 lanes: 8 weights x 8 inputs per clock
   | totals           | totals            | totals
 row_sum            row_sum             row_sum                     adds the partial totals of a finished row
   |                  |                   |
   +--------------> merge_out <-----------+                         stores rows while ports stream the next ones
                      |
              result vector (on chip)
```

- **`row_stream`** reads one wide word per clock (`LLM_LANES` weights), multiplies by the matching inputs and adds into `LLM_ACC_DEPTH`
  running totals per lane. A float add takes several cycles, so rotating between totals keeps the pipeline full (one new add per clock).
  When a row ends it hands its totals on and starts the next row immediately.
- **`row_sum`** (two small stages, `rs_slots` and `rs_final`) adds the totals of a row in a fixed tree order.
- **`merge_out`** takes the finished rows of every port in the order they are produced and applies the destination's operation:
  plain store (Q, K, V, hidden, head), residual add (output and down projections), or the SiLU gate (up projection).
- **Rows are split between ports** in consecutive blocks (`rows_of`). A row is always summed inside one port in the same order, so the
  result does not depend on how many ports there are. That is why the C simulation is bit-identical for 1 to 27 ports.

Weights are laid out per port by `sw/split_ports.py`, so each port reads only its own file, from its own HBM channel.

## 4. Why the code has 27 weight pointers

Vitis HLS needs one separate pointer argument per `m_axi` interface, and each interface is wired to one HBM channel in
`hls_u50/u50_link_llama1b_p27.cfg`. So `llm_top`, `forward`, `matvec` and `matvec_flow` carry `weights` plus `weights1` to `weights26`.
The repetition is deliberate, not copy-paste drift: the `LLM_PORTS` setting and the `#if LLM_PORTS >= N` blocks switch ports on and off,
and below 27 the unused pointers alias `weights`. In the C simulation the same arguments are simply pointers into the per-port files.

## 5. Settings that change the hardware

| Setting | Meaning | Final build |
|---|---|---|
| `LLM_PORTS` | weight ports (HBM channels read in parallel) | 27 |
| `LLM_LANES` | weights multiplied per port per clock | 8 |
| `LLM_ACC_DEPTH` | running totals per lane | 8 |
| `LLM_LANES_ATT`, `LLM_ACC_DEPTH_ATT` | the same for the attention unit | 16, 8 |
| `LLM_MAX_*` | sizes of the on-chip buffers (largest model the build accepts) | Llama-3.2-1B |

## 6. How it is tested

- `sw/tb_kernel.cpp` plays the host: it loads a weight file, runs prompt plus generation, writes every step's logits, then checks that a
  resumed run (`start_pos`) gives the same tokens and logits, and that a stop token ends the reply.
- `sw/test_kernel.py` builds that testbench and compares tokens and logits with the numpy model in `ref/`.
- `hls_u50/csim_check.sh` runs the 27-port testbench on a split model so logits can be compared byte for byte between kernel versions.
