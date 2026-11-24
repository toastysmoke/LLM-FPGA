# Progress Report

Phakin Dhamsirimongkol (Prue)
2026-10-20 to 2026-11-24

# Note
The work in this period was done locally on my own computer and on the remote build server for the U50, so it was not committed to this repository as it went. This is the first report and covers the whole period.

## Activities and accomplishments

### 1. First target: Alinx AXKU5 board (2026-10-20 to about 2026-10-27)
- Set up the HLS toolchain and wrote a Llama-style inference kernel (`llm_top`) in C++ for Vitis HLS, with a Python reference and tests. On 2026-10-20 it matched the reference on a small real model (stories260K: identical tokens, max logit difference 1.5e-5).
- Read the board manual and the DDR4 pin table, cross-checked the pins against Vivado's package data, and built the Vivado designs: a BRAM bring-up design (50 MHz, timing met) and DDR4 designs with a JTAG-to-AXI host path.
- Programmed the board and debugged DDR4 calibration. It failed at one byte lane (the data-mask line DM3, pin A22). I contacted Alinx and their own demo bitstream failed the same way, which points to a hardware fault on that board (board version 2.1). I built a 16-bit DDR4 variant (1 GB), it calibration and a 21-test memory test passed.
- Found and fixed a slow JTAG-to-AXI setting (AXI4-Lite instead of full AXI4), which made model loading about 100 times faster once rebuilt.
- First kernel run on the board produced wrong tokens. I isolated the cause with C/RTL co-simulation to an HLS mis-synthesis of the in-place final normalisation, fixed it in the kernel, rebuilt, and reprogrammed.
- 2026-10-23: LLM inference worked on the FPGA. stories260K on the AXKU5 gave tokens identical to the reference (120 steps, about 30 tokens/s).
- 2026-10-27: Ran a bigger model on the same bitstream, stories15M (60.8 MB, 6 layers) with identical tokens for 200 steps (about 0.77 tokens/s at 50 MHz).
- Wrote `model_to_fpga_spec.py` (2026-10-25 and 2026-10-26): it reads a model's architecture and produces the compute units, an SRAM-versus-external-memory plan, per-token traffic and a sizing estimate. Tested on the TinyLlama and Llama-3.2-1B headers.

### 2. Second target: Alveo U55C (late October to 2026-11-13)
- Built a U55C test design in Vivado (PCIe XDMA x8 Gen3 into the 16 GB HBM) and got the block design to validate. The synthesis was started and nothing was programmed.
- The U55C server was a shared lab machine running someone else's design, so I stopped there.

### 3. Final target: Alveo U50 (from 2026-11-13)
- Moved to a dedicated U50 VM and the Vitis flow (HLS on my PC, link on the VM, run through XRT).
- Built the Llama-3.2-1B fp32 kernel with no quantization and raised speed from 18.31 to 31.88 positions/s (1.74x) on the same 45-position test (6 prompt + 40 generated tokens):
  - 24 ports x 8 lanes instead of 12 x 16: 24.06
  - best link strategy: 27.23
  - registered pipeline stages on the worst timing paths: up to 29.31 at 212 MHz
  - 27 ports x 8 lanes (all of the card's HBM channels): 31.88 at 207 MHz
- Verified correctness: the 40 generated tokens are identical to the reference on the card, and the C simulation logits match bit for bit between builds.
- Prepared the demo-milestone submission: a public repository (README, kernel walkthrough, build scripts, tests) and the demo slides, including a recorded chat with the card.
- Measured the card's power with its board sensor: about 23 W idle and about 37.6 W while running 255 positions (26.09 positions/s).
- Started a floorplanning experiment (moving three memory blocks to the second chip region) to try to raise the clock.
- Wrote a benchmark notebook for a same-workload comparison against a Colab T4 GPU. It has not been run.

## Problems and issues

- **AXKU5 hardware fault:** the DDR4 DM3 line failed calibration even with the vendor's own demo. Worked around it with a 16-bit memory (half the bandwidth and 1 GB), and the board could never hold a 1B-parameter model.
- **HLS bug:** a correct C simulation produced wrong hardware output until co-simulation showed the in-place normalisation was mis-synthesised.
- **Slow early kernel:** at 50 MHz and 4.5 cycles per weight, even the 15M model ran at under 1 token/s, so larger models needed a different board.
- **U50 limits:** the chip is nearly full (about 99.8% of the first region's logic), so the clock settled at 207 MHz instead of 300 MHz. Place-and-route results vary by about 5% between runs and each link takes about 5.5 hours.
- **Longer context slows the kernel:** 31.88 positions/s at 45 positions and 26.09 at 255, because the attention unit is small.