"""Tests for the LLM_LANES setting of the kernel (several weights multiplied at once), on the PC.   usage: python tests/test_lanes.py
 - LANES = 1 must give exactly the same numbers, bit for bit, as the kernel before the setting existed (backups/llm_kernel_before_lanes.*)
 - LANES = 2, 4, 8, 16, with 1 to 8 running totals each, must give the same tokens as the numpy reference and logits within 1e-3 (the additions happen in a
   different order, so the last digits differ a little). The weights port is LANES floats wide, so dim and hidden must be multiples of LANES, and the state (K/V cache) port is
   LANES_ATT floats wide, so the head size must be a multiple of LANES_ATT: otherwise the kernel returns -8 (stories260K has hidden 172, so it
   runs with 1, 2, 4 lanes and is refused with 8 and 16).
Needs g++ and gcc (the Vitis MinGW compiler is looked for if they are not on PATH). Skipped when none is found."""
import glob
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "tests"), str(ROOT / "ref")]

if not shutil.which("g++"):
    for d in glob.glob(r"C:\AMDDesignTools\*\Vitis\tps\mingw\*\win64.o\nt\bin"):
        if (Path(d) / "g++.exe").exists():
            os.environ["PATH"] = d + os.pathsep + os.environ["PATH"]
            break
if not (shutil.which("g++") and shutil.which("gcc")):
    print("SKIP  no C++ compiler found")
    sys.exit(0)

import hf_to_flat  # noqa: E402
import reference  # noqa: E402
from hf_helpers import random_model, write_hf  # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="lanes_"))
results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(("PASS  " if ok else "FAIL  ") + name + ("" if ok else f"\n      {detail}"))


def build(srcdir, exe, lanes=None, depth=None, lanes_att=None, depth_att=None, extra=(), ports=None):
    """compile the kernel + testbench found in srcdir"""
    subprocess.run(["gcc", "-O2", "-c", str(srcdir / "llm.c"), "-o", str(srcdir / "llm.o")], check=True, capture_output=True)
    flags = ([f"-DLLM_LANES={lanes}"] if lanes else []) + ([f"-DLLM_ACC_DEPTH={depth}"] if depth else [])         + ([f"-DLLM_LANES_ATT={lanes_att}"] if lanes_att else []) + ([f"-DLLM_ACC_DEPTH_ATT={depth_att}"] if depth_att else []) + list(extra) + ([f"-DLLM_PORTS={ports}"] if ports else [])
    subprocess.run(["g++", "-O2", "-std=c++14", "-Wno-unknown-pragmas", *flags, "-o", str(exe), str(srcdir / "llm_kernel.cpp"),
                    str(srcdir / "tb_kernel.cpp"), str(srcdir / "llm.o"), "-lm"], check=True, capture_output=True)


def run(exe, flat, steps, prompt, tag):
    lg = TMP / f"{tag}.logits"
    out = subprocess.run([str(exe), str(flat), str(steps), str(lg), *map(str, prompt)], check=True, capture_output=True, text=True).stdout.split()
    return list(map(int, out)), lg


# the kernel as it is now, and as it was before the lanes setting
new_dir, old_dir = TMP / "new", TMP / "old"
for d in (new_dir, old_dir):
    d.mkdir()
    for f in (ROOT / "sw").glob("*"):
        if f.suffix in (".c", ".cpp", ".h"):
            shutil.copy(f, d / f.name)
shutil.copy(ROOT / "backups" / "llm_kernel_before_lanes.cpp", old_dir / "llm_kernel.cpp")
shutil.copy(ROOT / "backups" / "llm_kernel_before_lanes.h", old_dir / "llm_kernel.h")
shutil.copy(ROOT / "backups" / "tb_kernel_before_widetype.cpp", old_dir / "tb_kernel.cpp")   # the test bench that matches the old float* port
build(old_dir, TMP / "old.exe")

# two models: the real stories260K, and a small converted model whose sizes are not multiples of the lane counts
models = []
f260 = ROOT / "models" / "stories260K.flat"
if f260.exists():
    models.append(("stories260K", f260, [1], 24))
cfg, t = random_model(31)
rnd = TMP / "random.flat"
hf_to_flat.convert(write_hf(TMP / "rand_hf", cfg, t), rnd, log=lambda m: None)
models.append(("random (dim 32)", rnd, [3, 17, 5], 10))

old_out = {}
for name, flat, prompt, steps in models:
    old_out[name] = run(TMP / "old.exe", flat, steps, prompt, "old_" + name.split()[0])

# (matrix lanes, matrix running totals, attention lanes, attention running totals); None = same as the matrix setting
for lanes, depth, la, da in ((1, 1, None, None), (2, 1, None, None), (4, 1, None, None), (8, 1, None, None), (16, 1, None, None), (1, 4, None, None),
                             (4, 4, None, None), (8, 4, None, None), (2, 5, None, None), (8, 8, None, None),
                             (4, 4, 1, 1), (1, 1, 4, 4), (8, 4, 2, 4), (2, 2, 16, 4), (16, 4, 4, 4)):
    exe = TMP / f"lanes{lanes}_{depth}_{la}_{da}.exe"
    build(new_dir, exe, lanes, depth, la, da)
    for name, flat, prompt, steps in models:
        c, w = reference.load_flat(str(flat))
        hs = c["dim"] // c["n_heads"]
        if c["dim"] % lanes or c["hidden_dim"] % lanes or hs % (la or lanes):
            p = subprocess.run([str(exe), str(flat), str(steps), str(TMP / "x.logits"), *map(str, prompt)], capture_output=True, text=True)
            check(f"lanes {lanes:>2}, running totals {depth}, attention {la or lanes}/{da or depth}, {name}: refused with -8 (dim {c['dim']}, hidden {c['hidden_dim']} or head size {hs} not multiples of the lanes)", p.returncode != 0 and "returned -8" in p.stderr, p.stderr)
            continue
        tokens, lg = run(exe, flat, steps, prompt, f"l{lanes}_{depth}_{la}_{da}_" + name.split()[0])
        r_tokens, r_logits = reference.generate(c, w, prompt, steps)
        k_logits = np.fromfile(lg, dtype="<f4").reshape(len(r_logits), c["vocab_size"])
        diff = float(np.abs(k_logits - np.array(r_logits)).max())
        check(f"lanes {lanes:>2}, running totals {depth}, attention {la or lanes}/{da or depth}, {name}: tokens equal the reference; max logit difference {diff:.1e}", tokens == r_tokens and diff < 1e-3, f"tokens equal: {tokens == r_tokens}, diff {diff}")
        if lanes == 1 and depth == 1 and la is None:
            old_tokens, old_lg = old_out[name]
            same = Path(lg).read_bytes() == Path(old_lg).read_bytes() and tokens == old_tokens
            check(f"lanes  1, running totals 1, {name}: identical, bit for bit, to the kernel from before the setting", same)

# several weight ports: each row is summed by one port in the same order as with a single port, so for the same lanes per port the
# results must be identical, bit for bit, whatever the number of ports. Rows are not a multiple of the ports (2, 3, 4) on purpose.
for lanes, depth, la, da in ((1, 1, None, None), (2, 4, 2, 4), (4, 4, 1, 1)):
    base_exe = TMP / f"port1_{lanes}_{depth}_{la}_{da}.exe"
    build(new_dir, base_exe, lanes, depth, la, da)
    for ports in (2, 3, 4):
        exe = TMP / f"port{ports}_{lanes}_{depth}_{la}_{da}.exe"
        build(new_dir, exe, lanes, depth, la, da, ports=ports)
        for name, flat, prompt, steps in models:
            c, w = reference.load_flat(str(flat))
            hs = c["dim"] // c["n_heads"]
            if c["dim"] % lanes or c["hidden_dim"] % lanes or hs % (la or lanes):
                continue
            tag = f"p{ports}_{lanes}_{depth}_{la}_{da}_" + name.split()[0]
            tokens, lg = run(exe, flat, steps, prompt, tag)
            b_tokens, b_lg = run(base_exe, flat, steps, prompt, "b" + tag)
            check(f"{ports} weight ports, {lanes} lanes each, {name}: identical, bit for bit, to one port", tokens == b_tokens and Path(lg).read_bytes() == Path(b_lg).read_bytes())

# 4 or 8 running totals in the RMSNorm and softmax sums: the sums differ from the plain order in the last digits only
for sums in (4, 8):
    for lanes, depth, la, da in ((1, 1, None, None), (4, 4, 2, 2)):
        exe = TMP / f"vs{sums}_{lanes}_{depth}_{la}_{da}.exe"
        build(new_dir, exe, lanes, depth, la, da, extra=(f"-DLLM_VEC_SUMS={sums}",))
        for name, flat, prompt, steps in models:
            c, w = reference.load_flat(str(flat))
            hs = c["dim"] // c["n_heads"]
            if c["dim"] % lanes or c["hidden_dim"] % lanes or hs % (la or lanes):
                continue
            tokens, lg = run(exe, flat, steps, prompt, f"vs{sums}_{lanes}_{depth}_{la}_{da}_" + name.split()[0])
            r_tokens, r_logits = reference.generate(c, w, prompt, steps)
            k_logits = np.fromfile(lg, dtype="<f4").reshape(len(r_logits), c["vocab_size"])
            diff = float(np.abs(k_logits - np.array(r_logits)).max())
            check(f"{sums} running totals in the norm and softmax sums, lanes {lanes}, {name}: tokens equal the reference; max logit difference {diff:.1e}",
                  tokens == r_tokens and diff < 1e-3)

# the same kernel with the hardware word type (one bit vector per wide word, as in synthesis); needs the Vitis include folder
vitis_inc = next(iter(glob.glob(r"C:\AMDDesignTools\*\Vitis\include")), None)
if vitis_inc and (Path(vitis_inc) / "ap_int.h").exists():
    for lanes, depth, la, da in ((1, 1, None, None), (4, 4, 1, 1), (8, 4, 4, 4), (2, 2, 2, 2)):
        exe = TMP / f"ap{lanes}_{depth}_{la}_{da}.exe"
        build(new_dir, exe, lanes, depth, la, da, extra=("-DLLM_WIDE_AP", f"-I{vitis_inc}"))
        for name, flat, prompt, steps in models:
            c, w = reference.load_flat(str(flat))
            hs = c["dim"] // c["n_heads"]
            if c["dim"] % lanes or c["hidden_dim"] % lanes or hs % (la or lanes):
                continue
            tokens, lg = run(exe, flat, steps, prompt, f"ap{lanes}_{depth}_{la}_{da}_" + name.split()[0])
            r_tokens, r_logits = reference.generate(c, w, prompt, steps)
            k_logits = np.fromfile(lg, dtype="<f4").reshape(len(r_logits), c["vocab_size"])
            diff = float(np.abs(k_logits - np.array(r_logits)).max())
            check(f"bit-vector words, lanes {lanes}, totals {depth}, attention {la or lanes}/{da or depth}, {name}: tokens equal the reference; "
                  f"max logit difference {diff:.1e}", tokens == r_tokens and diff < 1e-3)
            if lanes == 1 and depth == 1:
                same = Path(lg).read_bytes() == Path(old_out[name][1]).read_bytes()
                check(f"bit-vector words, lanes 1, {name}: identical, bit for bit, to the original kernel", same)
else:
    print("SKIP  Vitis include folder not found: the bit-vector word type was not tested")

print(f"\n{sum(results)} of {len(results)} checks passed")
shutil.rmtree(TMP, ignore_errors=True)
sys.exit(0 if all(results) else 1)
