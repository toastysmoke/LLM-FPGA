#!/usr/bin/env python3
"""llama_text.py - text in, text out through the U50 card.

  python sw/llama_text.py "The capital of France is" --steps 20

It encodes the prompt on this PC (Llama 3 tokenizer from the model folder), runs host_xrt on the U50 VM over ssh (the VM has no tokenizer),
reads the token ids back and decodes them. The card only ever sees token numbers. Needs: pip install tokenizers; an ssh login to the machine that has the card (--ssh).
It never flashes or resets anything: it just starts host_xrt, the same command as in sw/host_xrt.cpp's header.
"""
import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

from tokenizers import Tokenizer

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def main():
    """Encode the prompt here, run the card host once over ssh, and decode the generated token ids."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prompt", help="the text to continue")
    ap.add_argument("--steps", type=int, default=20, help="how many tokens to generate (the model is greedy: no sampling)")
    ap.add_argument("--tokenizer", default=str(ROOT / "models" / "Llama-3.2-1B" / "tokenizer.json"))
    ap.add_argument("--ssh", required=True, help="ssh host of the VM that has the card (user@host or an ssh config alias)")
    ap.add_argument("--key", default=None, help="ssh private key file (optional; default: your ssh config)")
    ap.add_argument("--xclbin", required=True, help="xclbin path on the VM")
    ap.add_argument("--ports", type=int, default=27)
    ap.add_argument("--split", required=True, help="prefix of the per-port weight files (sw/split_ports.py), the fast layout; with --share use the old xclbin and --model instead")
    ap.add_argument("--host", required=True, help="host_xrt binary on the VM, built from sw/host_xrt.cpp with -DLLM_PORTS=<ports>")
    ap.add_argument("--timeout", type=int, default=900)
    a = ap.parse_args()

    tok = Tokenizer.from_file(a.tokenizer)
    ids = tok.encode(a.prompt).ids            # the tokenizer puts <|begin_of_text|> (128000) first
    prompt = " ".join(map(str, ids))
    print(f"prompt tokens ({len(ids)}): {prompt}")

    remote = (f"source /opt/xilinx/xrt/setup.sh >/dev/null 2>&1; cd ~/run; timeout {a.timeout} {a.host} --xclbin {a.xclbin} --split {a.split} "
              f"--steps {a.steps} --prompt \"{prompt}\" --ports {a.ports}")
    t0 = time.time()
    r = subprocess.run(["ssh"] + (["-i", a.key] if a.key else []) + [a.ssh, remote], capture_output=True, text=True)
    out = r.stdout + r.stderr
    if r.returncode != 0:
        print(out)
        print(f"host_xrt failed (exit {r.returncode}); nothing was reset or flashed.")
        sys.exit(1)
    lines = [l for l in out.splitlines() if re.fullmatch(r"-?\d+( -?\d+)*", l.strip())]
    if not lines:
        print(out)
        sys.exit("could not find the token line in the host output")
    toks = [int(x) for x in lines[-1].split()]
    new = toks[len(ids):]
    for l in out.splitlines():
        if l.startswith("kernel time"):
            print(l)
    print(f"generated tokens: {new}")
    print("\n--- text ---")
    print(tok.decode(toks, skip_special_tokens=True))
    print(f"--- ({len(new)} tokens in {time.time() - t0:.1f} s wall, including loading the 4.9 GB of weights) ---")


if __name__ == "__main__":
    main()
