"""Build the kernel-shaped C++ and check it against the numpy reference (tokens + every step's logits)."""
import os
import subprocess
import sys
import tempfile
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REF = os.path.join(HERE, "..", "ref")
sys.path.insert(0, REF)
from reference import load_flat, generate  # noqa: E402
from test_all import CONFIGS  # noqa: E402


def build(tmp):
    """Compile llm.c and the kernel with the testbench into one executable (plain g++, no Vitis needed)."""
    obj = os.path.join(tmp, "llm.o")
    exe = os.path.join(tmp, "tb_kernel")
    subprocess.run(["gcc", "-O2", "-Wall", "-Wextra", "-c", os.path.join(HERE, "llm.c"), "-o", obj], check=True)
    subprocess.run(["g++", "-O2", "-std=c++14", "-Wall", "-Wextra", "-Wno-unknown-pragmas", "-o", exe,
                    os.path.join(HERE, "llm_kernel.cpp"), os.path.join(HERE, "tb_kernel.cpp"), obj, "-lm"],
                   check=True)
    return exe


def main():
    """For each test model: generate and convert it, run the testbench, and compare tokens and every step's logits with the numpy reference; also check that oversized models and too-long sequences are rejected."""
    tmp = tempfile.mkdtemp()
    exe = build(tmp)
    for name, c in CONFIGS.items():
        src, flat = os.path.join(tmp, name + ".bin"), os.path.join(tmp, name + ".flat")
        gen = ["make_tiny_model.py", src, "--dim", c["dim"], "--hidden", c["hidden"], "--layers", c["layers"],
               "--heads", c["heads"], "--kv-heads", c["kv_heads"], "--vocab", c["vocab"],
               "--seq-len", c["seq_len"], "--shared", c["shared"]]
        subprocess.run([sys.executable, *map(str, gen)], check=True, cwd=REF, stdout=subprocess.DEVNULL)
        subprocess.run([sys.executable, "convert.py", src, flat], check=True, cwd=REF, stdout=subprocess.DEVNULL)

        prompt, steps = [1, 5, 9], 12
        lgfile = os.path.join(tmp, name + ".logits")
        out = subprocess.run([exe, flat, str(steps), lgfile, *map(str, prompt)],
                             check=True, capture_output=True, text=True).stdout.splitlines()[0].split()
        k_tokens = list(map(int, out))
        cfg, w = load_flat(flat)
        r_tokens, r_logits = generate(cfg, w, prompt, steps)
        k_logits = np.fromfile(lgfile, dtype="<f4").reshape(len(r_logits), cfg["vocab_size"])
        err = np.abs(k_logits - np.array(r_logits)).max()
        assert k_tokens == r_tokens, f"{name}: tokens differ\n kernel {k_tokens}\n ref    {r_tokens}"
        assert err < 1e-3, f"{name}: max logit error {err}"
        print(f"{name:8s} ok  max|logit diff|={err:.2e}  tokens={k_tokens}")

    # error paths: config bigger than the on-chip limits, and a too-long sequence
    big = os.path.join(tmp, "big.bin")
    bigflat = os.path.join(tmp, "big.flat")
    subprocess.run([sys.executable, "make_tiny_model.py", big, "--dim", "320", "--hidden", "800", "--layers", "1",
                    "--heads", "5", "--kv-heads", "5", "--vocab", "64", "--seq-len", "16"],
                   check=True, cwd=REF, stdout=subprocess.DEVNULL)
    subprocess.run([sys.executable, "convert.py", big, bigflat], check=True, cwd=REF, stdout=subprocess.DEVNULL)
    r = subprocess.run([exe, bigflat, "2", os.path.join(tmp, "x.logits"), "1"], capture_output=True, text=True)
    assert r.returncode != 0 and "-4" in r.stderr, r.stderr
    print("oversized model rejected with -4")
    r = subprocess.run([exe, os.path.join(tmp, "mha.flat"), "100", os.path.join(tmp, "x.logits"), "1"],
                       capture_output=True, text=True)
    assert r.returncode != 0 and "-7" in r.stderr, r.stderr
    print("too-long sequence rejected with -7")
    print("kernel matches the reference")


if __name__ == "__main__":
    main()
