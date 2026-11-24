#!/bin/bash
# HLS build of the 27-port x 8-lane kernel (C synthesis + package to an .xo). Needs v++/vitis-run on PATH (Vitis 2026.1). Times it.
cd "$(dirname "$0")"
D=build_llama1b_p27
echo "START $(date '+%F %T')" > $D.time.log
v++ -c --mode hls --config hls_config_llama1b_p27.cfg --work_dir $D > $D.csynth.log 2>&1
echo "csynth exit=$? $(date '+%T')" >> $D.time.log
vitis-run --mode hls --package --config hls_config_llama1b_p27.cfg --work_dir $D > $D.package.log 2>&1
echo "package exit=$? $(date '+%T')" >> $D.time.log
echo "END $(date '+%F %T')" >> $D.time.log
