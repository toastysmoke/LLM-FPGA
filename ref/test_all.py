"""End-to-end check of the generator -> converter -> reference chain.

For several configs (multi-head, grouped-query, separate classifier) it
  1. makes a random llama2.c-format model,
  2. converts it to the flat file,
  3. checks the flat file layout (aligned, no overlaps, header matches),
  4. runs greedy generation from both files and requires identical tokens and logits,
  5. sanity-checks the model against a straightforward independent attention computation.
"""
import os
import subprocess
import sys
import tempfile
import numpy as np
import layout
from reference import load_llama2c, load_flat, generate, Model

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIGS = {
    "mha":    dict(dim=64, hidden=172, layers=2, heads=4, kv_heads=4, vocab=256, seq_len=64, shared=1),
    "gqa":    dict(dim=64, hidden=172, layers=3, heads=8, kv_heads=2, vocab=300, seq_len=48, shared=1),
    "noshare": dict(dim=48, hidden=128, layers=2, heads=6, kv_heads=3, vocab=200, seq_len=32, shared=0),
}


def run(*args):
    """Run a Python script in ref/ quietly; raises if it exits non-zero."""
    subprocess.run([sys.executable, *args], check=True, cwd=HERE, stdout=subprocess.DEVNULL)


def check_layout(cfg, o):
    """Check the section layout of one configuration: every section 64-byte aligned, in order, not overlapping, and inside the file."""
    spans = [("tok_emb", o["tok_emb"], layout.nbytes((cfg["vocab_size"], cfg["dim"])))]
    for l in range(cfg["n_layers"]):
        base = o["layers_start"] + l * o["layer_stride"]
        for name, shape in layout.layer_tensors(cfg):
            spans.append((f"L{l}.{name}", base + o["layer_rel"][name], layout.nbytes(shape)))
    hs = layout.head_size(cfg)
    rope = layout.nbytes((cfg["seq_len"], hs // 2))
    spans += [("final_norm", o["final_norm"], layout.nbytes((cfg["dim"],))),
              ("rope_cos", o["rope_cos"], rope), ("rope_sin", o["rope_sin"], rope)]
    if not cfg["shared"]:
        spans.append(("wcls", o["wcls"], layout.nbytes((cfg["vocab_size"], cfg["dim"]))))
    spans.sort(key=lambda s: s[1])
    prev_end = layout.HEADER_BYTES
    for name, off, n in spans:
        assert off % layout.ALIGN == 0, f"{name} not aligned"
        assert off >= prev_end, f"{name} overlaps previous section"
        prev_end = off + n
    assert prev_end <= o["total"]


def independent_attention_check(cfg, w):
    """Re-do layer 0 for one position with explicit loops over heads/positions; compare to Model."""
    pos_count = 4
    m = Model(cfg, w)
    toks = [3, 7, 11, 2]
    for p, t in enumerate(toks):
        m.forward(t, p)
    # recompute the cached K/V for layer 0 by hand and compare
    hs, kvd = layout.head_size(cfg), layout.kv_dim(cfg)
    for p, t in enumerate(toks):
        x = w["tok_emb"][t]
        xb = x * w["rms_att"][0] / np.sqrt(np.mean(x * x) + 1e-5)
        k = w["wk"][0] @ xb
        for i in range(0, kvd, 2):
            hd = i % hs
            a, b = k[i], k[i + 1]
            c_, s_ = w["rope_cos"][p, hd // 2], w["rope_sin"][p, hd // 2]
            assert np.allclose([a * c_ - b * s_, a * s_ + b * c_], m.k[0, p, i:i + 2], atol=1e-5)
        assert np.allclose(w["wv"][0] @ xb, m.v[0, p], atol=1e-5)


def main():
    """Generate each test model, convert it, and compare the converted file with the original through the numpy reference."""
    tmp = tempfile.mkdtemp()
    for name, c in CONFIGS.items():
        src, flat, gold = (os.path.join(tmp, f"{name}.{e}") for e in ("bin", "flat", "json"))
        args = ["make_tiny_model.py", src, "--dim", c["dim"], "--hidden", c["hidden"], "--layers", c["layers"],
                "--heads", c["heads"], "--kv-heads", c["kv_heads"], "--vocab", c["vocab"],
                "--seq-len", c["seq_len"], "--shared", c["shared"]]
        run(*map(str, args))
        run("convert.py", src, flat, "--hex", flat + ".hex")

        cfg_a, w_a = load_llama2c(src)
        cfg_b, w_b = load_flat(flat)
        assert cfg_a == cfg_b
        _, o = layout.unpack_header(open(flat, "rb").read(layout.HEADER_BYTES))
        check_layout(cfg_b, o)
        for k in w_a:
            assert np.array_equal(w_a[k], w_b[k]), f"{name}: tensor {k} differs after conversion"

        prompt = [1, 5, 9]
        t_a, l_a = generate(cfg_a, w_a, prompt, 10)
        t_b, l_b = generate(cfg_b, w_b, prompt, 10)
        assert t_a == t_b
        assert all(np.array_equal(x, y) for x, y in zip(l_a, l_b))
        assert all(np.isfinite(x).all() for x in l_b)
        independent_attention_check(cfg_b, w_b)
        hexlines = open(flat + ".hex").read().split()
        assert len(hexlines) * 4 == o["total"]
        print(f"{name:8s} ok  tokens={t_b}  layer_stride={o['layer_stride']}  file={o['total']} bytes")
    print("all tests passed")


if __name__ == "__main__":
    main()
