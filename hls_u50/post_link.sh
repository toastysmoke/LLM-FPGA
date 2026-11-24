#!/bin/bash
# usage: post_link.sh BUILD_DIR   (on the card machine; BUILD_DIR holds time.log, link.log and llm_top.xclbin)
# Waits for the link to end, then runs the host test twice and writes BUILD_DIR/card.log.
# Override with env vars: HOST (host_xrt binary), MODEL (.flat), SPLIT (per-port weight prefix), PORTS, PROMPT.
D=$1
HOST=${HOST:-$HOME/llm-host/host_xrt27}; MODEL=${MODEL:-$HOME/run/Llama-3.2-1B.flat}
SPLIT=${SPLIT:-$HOME/run/split/llamainst27}; PORTS=${PORTS:-27}; PROMPT=${PROMPT:-"128000 14924 512 10267 2874 284"}
until grep -q "^END" $D/time.log 2>/dev/null; do sleep 60; done
{ cat $D/time.log; grep -E "AUTO-FREQ-SCALING-0[47]|Kernel \(DATA\)" $D/link.log | tail -3
  if [ -f $D/llm_top.xclbin ]; then
    source /opt/xilinx/xrt/setup.sh >/dev/null 2>&1
    for i in 1 2; do $HOST --xclbin $D/llm_top.xclbin --model $MODEL --split $SPLIT --ports $PORTS --steps 40 --prompt "$PROMPT" 2>&1 | tail -2; done
  else echo NO-XCLBIN; fi; } > $D/card.log 2>&1
echo DONE >> $D/card.log
