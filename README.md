# FPGA Inference Engine: Llama-3.2-1B on an Alveo U50 (fp32, weights in HBM)

A Vitis HLS kernel that runs the whole forward pass of **Llama-3.2-1B** (fp32, no quantization) on an **Alveo U50**. The FPGA holds only compute
and small on-chip buffers. Weights, the K/V cache, logits and tokens sit in HBM. The host (XRT over PCIe) loads the weights once, sends token ids,
starts the kernel and reads token ids back. The tokenizer and chat template run on a PC; the card only ever sees token numbers.

**Result:** 31.88 positions/s (one position = one kernel pass over one token, prompt or generated) at a 207 MHz kernel clock, with output
tokens identical to the reference on the card. That is 1.74x the first U50 build.

## Design in one page

For a guided tour of the code, read [docs/KERNEL.md](docs/KERNEL.md) first, then `sw/llm_kernel.h` (memory map and settings) and `sw/llm_kernel.cpp`.

- **One kernel, `llm_top`** (`sw/llm_kernel.cpp`): embedding, RMSNorm, RoPE, grouped-query attention with a K/V cache in HBM, SwiGLU MLP and
  the output head, all as streaming matrix-vector products.
- **27 weight ports x 8 lanes** (216 fp32 multiply-adds per clock). Each port is a 256-bit `m_axi` interface bringing 8 weights per clock from
  its own HBM pseudo channel. Matrix rows are split between ports; a row's sum stays inside one port, so the number of ports never changes the
  numbers (the C simulation is bit-identical for any port count). 27 is the maximum: HBM has 32 channels (port 0 uses 5, ports 1..26 one each,
  state and tokens share the last).
- **Same source everywhere.** `sw/llm_kernel.cpp` is built as plain C++ for the PC testbench, for Vitis HLS, and for the card.
- **Resume and stop.** The K/V cache stays in HBM between calls (`start_pos`), so a chat only computes the new tokens, and the kernel stops at the
  end-of-turn token.

## Results (same 45-position run: 6 prompt + 40 generated tokens; tokens identical in every build)

| Build | Ports x lanes | Kernel clock | Positions/s | Speedup |
|---|---|---|---|---|
| start | 12 x 16 | 190 MHz | 18.31 | 1.00x |
| more ports | 24 x 8 | 198 MHz | 24.06 | 1.31x |
| + link strategy NetDelay_high | 24 x 8 | 196 MHz | 27.23 | 1.49x |
| + registered stage | 24 x 8 | 207 MHz | 28.63 | 1.56x |
| + registered control | 24 x 8 | 212 MHz | 29.31 | 1.60x |
| **final** | **27 x 8** | **207 MHz** | **31.88** | **1.74x** |

Gains came from fewer cycles per token (about 8.2M down to 7.4M: more parallel ports, wide input broadcast, a merge stage that overlaps copy-out
with streaming) and a faster clock (timing-driven link strategy, registers on the worst paths). Negative results (a retiming strategy, a
flattened merge stage) were measured and dropped. The logic is nearly full (SLR0 99.8% CLB), so the remaining lever is placement.

## Repository layout

| Path | What |
|---|---|
| `sw/llm_kernel.cpp`, `llm_kernel.h` | the HLS kernel (`LLM_PORTS`, `LLM_LANES` set the shape) |
| `sw/tb_kernel.cpp`, `llm.c`, `llm.h` | C testbench: plain-C reference, resume and stop-token checks, logits dump |
| `sw/host_xrt.cpp` | XRT host test: loads the xclbin and weights, runs, compares to `--expect` |
| `sw/host_serve.cpp` | XRT host server for chat: loads weights once, answers one request per line |
| `sw/chat.py`, `llama_text.py` | PC side: tokenizer + chat template, drives the host over ssh |
| `sw/split_ports.py` | cuts a flat weight file into one file per port |
| `sw/test_kernel.py` | builds the kernel as C++ and checks tokens and logits against `ref/` |
| `ref/` | numpy reference model and the flat weight format (`layout.py`) |
| `hf_to_flat.py` | Hugging Face checkpoint to the flat fp32 file |
| `kernel_plan.py`, `fit_check.py`, `model_to_fpga_spec.py`, `hls_check.py` | planning and helper tools |
| `hls_u50/` | HLS config, link config (HBM channel map) and build scripts for the 27-port kernel (the source is `sw/llm_kernel.cpp`) |
| `boards/` | board descriptions used by the planning tools |
| `tests/` | helper-script tests (`python tests/run_all.py --fast`) |
| `docs/KERNEL.md` | guided tour of the kernel: dataflow, the matrix unit, why there are 27 weight pointers, settings, tests |

## Build and run

Needs: Vitis 2026.1 (HLS and `v++`), XRT and the U50 platform `xilinx_u50_gen3x16_xdma_5_202210_1` on the machine with the card, Python 3.10+ with
`numpy` and `tokenizers`, and a C++17 compiler.

What needs what:
- Python and numpy only: `hf_to_flat.py`, `fit_check.py`, `kernel_plan.py`, `model_to_fpga_spec.py`.
- A C++ compiler only: `sw/test_kernel.py` and the testbench.
- Vitis HLS 2026.1 with a licence (no card needed): `hls_check.py` and `hls_u50/run_p27.sh`. The link step also needs the U50 platform files.
- The card, XRT and the card machine: `host_xrt`, `host_serve`, `chat.py`.

1. **Check the kernel on the PC (no FPGA needed):**
   ```
   python sw/test_kernel.py
   python tests/run_all.py --fast
   ```
2. **Get the model** (not included, the weights are Meta's): download `meta-llama/Llama-3.2-1B-Instruct` from Hugging Face, then
   ```
   python hf_to_flat.py ./Llama-3.2-1B-Instruct Llama-3.2-1B-Instruct.flat
   python sw/split_ports.py Llama-3.2-1B-Instruct.flat --ports 27 --chunk 8192 --out llamainst27
   ```
3. **HLS** (PC): `bash hls_u50/run_p27.sh` produces the `.xo` in `hls_u50/build_llama1b_p27/`.
4. **Link** (about 5 h, needs about 30 GB RAM): copy the `.xo` and `hls_u50/u50_link_llama1b_p27.cfg` to the build machine, then
   ```
   v++ -l -t hw --platform xilinx_u50_gen3x16_xdma_5_202210_1 --config u50_link_llama1b_p27.cfg -o llm_top.xclbin llm_top.xo
   ```
   The cfg maps port 0 to `HBM[0:4]`, ports 1..26 to `HBM[5]..HBM[30]`, and state and tokens to `HBM[31]`, and selects the `Performance_NetDelay_high` strategy.
5. **Test on the card** (host built from `sw/host_xrt.cpp`):
   ```
   g++ -std=c++17 -O2 -DLLM_PORTS=27 -DLLM_LANES=8 -Isw -I/opt/xilinx/xrt/include sw/host_xrt.cpp sw/llm.c -L/opt/xilinx/xrt/lib -lxrt_coreutil -lpthread -o host_xrt27
   ./host_xrt27 --xclbin llm_top.xclbin --split llamainst27 --ports 27 --steps 40 --prompt "128000 14924 512 10267 2874 284"
   ```
   `hls_u50/post_link.sh` runs this test twice after a link and writes `card.log`.
6. **Chat** (from a PC that can ssh to the card machine):
   ```
   python sw/chat.py --ssh USER@HOST --template llama3 --tokenizer Llama-3.2-1B-Instruct/tokenizer.json \
       --xclbin '~/llm_top.xclbin' --host '~/host_serve27' --split '~/llamainst27' --ports 27
   ```
   Commands inside the chat: `/reset`, `/system TEXT`, `/quit`.

The tools never flash or reset the card; the xclbin is loaded through XRT only.

## Correctness

- `sw/test_kernel.py`: tokens identical and logit error below 1e-5 against the numpy reference on three random model shapes (multi-head,
  grouped-query, separate classifier); oversized models and too-long sequences are rejected with error codes.
- `sw/tb_kernel.cpp` also checks resume (`start_pos`) and stop-token behaviour.
- Between kernel versions the C-simulation logits are compared byte for byte (`hls_u50/csim_check.sh`).
- On the card, generated tokens are compared with `--expect`, twice per build.

## Limits and next steps

fp32 is bandwidth-bound: the ceiling on this card is about 35 to 40 positions/s. The next steps are floorplanning (the HBM sits next to SLR0,
which is full) and lower-precision weights (ternary needs a model trained for it).
