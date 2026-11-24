"""Build the C core and check it against the numpy reference on the tiny test models."""
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


def main():
    """Build the C core and compare its tokens and logits with the numpy reference on each test model."""
    tmp = tempfile.mkdtemp()
    exe = os.path.join(tmp, "llm_pc")
    subprocess.run(["gcc", "-O2", "-Wall", "-Wextra", "-o", exe, os.path.join(HERE, "llm.c"),
                    os.path.join(HERE, "main_pc.c"), "-lm"], check=True)
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
                             check=True, capture_output=True, text=True).stdout.split()
        c_tokens = list(map(int, out))
        cfg, w = load_flat(flat)
        r_tokens, r_logits = generate(cfg, w, prompt, steps)
        c_logits = np.fromfile(lgfile, dtype="<f4").reshape(len(r_logits), cfg["vocab_size"])
        err = np.abs(c_logits - np.array(r_logits)).max()
        assert c_tokens == r_tokens, f"{name}: tokens differ\n C  {c_tokens}\n ref {r_tokens}"
        assert err < 1e-3, f"{name}: max logit error {err}"
        print(f"{name:8s} ok  max|logit diff|={err:.2e}  tokens={c_tokens}")
    print("C core matches the reference")


if __name__ == "__main__":
    main()
