#!/bin/bash
# usage: csim_check.sh DIR SPLIT_PREFIX [STEPS]   DIR holds llm_kernel.cpp/.h, tb_kernel.cpp, llm.c/.h, portable_alloc.h
# Builds the C testbench for 27 ports x 8 lanes and runs stories15M split in 27 ports (sw/split_ports.py --ports 27 --chunk 768).
# Writes DIR/out.txt (tokens) and DIR/out.logits; compare out.logits byte for byte between kernel versions.
set -e
cd "$1"
F="-DLLM_MAX_DIM=288 -DLLM_MAX_HIDDEN=768 -DLLM_MAX_SEQ=256 -DLLM_MAX_HEADS=6 -DLLM_MAX_HEAD_SIZE=48 -DLLM_PORTS=27 -DLLM_LANES=8 -DLLM_ACC_DEPTH=8 -DLLM_LANES_ATT=16 -DLLM_ACC_DEPTH_ATT=8 -DLLM_VEC_SUMS=8"
gcc -O2 -c llm.c -o l.o
g++ -std=c++17 -O2 $F -I. -c llm_kernel.cpp -o k.o
g++ -std=c++17 -O2 $F -I. -c tb_kernel.cpp -o t.o
g++ k.o t.o l.o -o tb
./tb split:$2 ${3:-20} out.logits 1 5 9 > out.txt 2>&1
tail -3 out.txt
