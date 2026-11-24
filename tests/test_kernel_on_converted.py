"""Run CONVERTED Hugging Face models through the real kernel code (sw/llm_kernel.cpp, built for the PC) and compare its tokens and
logits with the numpy reference and with the numpy Hugging Face implementation.   usage: python tests/test_kernel_on_converted.py
Needs g++ and gcc; if they are not on PATH the Vitis MinGW compiler is looked for. Skipped (exit 0) when none is found."""
import glob
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "tests"), str(ROOT / "sw"), str(ROOT / "ref")]

if not shutil.which("g++"):
    for d in glob.glob(r"C:\AMDDesignTools\*\Vitis\tps\mingw\*\win64.o\nt\bin") + glob.glob("/tools/Xilinx/*/Vitis/tps/lnx64/gcc-*/bin"):
        if (Path(d) / "g++.exe").exists() or (Path(d) / "g++").exists():
            os.environ["PATH"] = d + os.pathsep + os.environ["PATH"]
            break
if not (shutil.which("g++") and shutil.which("gcc")):
    print("SKIP  no C++ compiler found")
    sys.exit(0)

import hf_to_flat  # noqa: E402
import reference  # noqa: E402
import test_kernel  # noqa: E402
from hf_helpers import random_model, write_hf, hf_logits  # noqa: E402

tmp = Path(tempfile.mkdtemp(prefix="kernel_conv_"))
exe = test_kernel.build(str(tmp))
results = []
for seed, shared, dtype in ((11, False, "BF16"), (12, True, "F16"), (13, False, "F32")):
    cfg, t = random_model(seed, shared=shared)
    flat = tmp / f"m{seed}.flat"
    hf_to_flat.convert(write_hf(tmp / f"m{seed}", cfg, t, dtype), flat, log=lambda m: None)
    prompt, steps = [3, 17, 5], 8
    lg = tmp / f"m{seed}.logits"
    out = subprocess.run([exe, str(flat), str(steps), str(lg), *map(str, prompt)], check=True, capture_output=True, text=True).stdout.splitlines()[0].split()
    k_tokens = list(map(int, out))
    c, w = reference.load_flat(str(flat))
    r_tokens, r_logits = reference.generate(c, w, prompt, steps)
    k_logits = np.fromfile(lg, dtype="<f4").reshape(len(r_logits), cfg["vocab_size"])
    d_ref = float(np.abs(k_logits - np.array(r_logits)).max())
    d_hf = float(np.abs(k_logits - hf_logits(cfg, t, k_tokens[:len(r_logits)], dtype=dtype)).max())
    ok = k_tokens == r_tokens and d_ref < 1e-3 and d_hf < 1e-3
    results.append(ok)
    print(("PASS  " if ok else "FAIL  ") + f"kernel on a converted {dtype} model ({'tied' if shared else 'separate'} head): tokens equal the reference: "
          f"{k_tokens == r_tokens}; max|kernel - reference| {d_ref:.1e}; max|kernel - Hugging Face convention| {d_hf:.1e}")
print(f"\n{sum(results)} of {len(results)} checks passed")
shutil.rmtree(tmp, ignore_errors=True)
sys.exit(0 if all(results) else 1)
