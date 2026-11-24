"""Golden reference: numpy fp32 Llama forward pass that mirrors llama2.c's run.c.

Loads either a llama2.c checkpoint or the flat FPGA weight file, so the two can be compared.

  python reference.py model.bin|model.flat --prompt 1 5 9 --steps 8 [--golden out.json]
"""
import argparse
import json
import struct
import numpy as np
import layout


# ---------------------------------------------------------------- loaders
def load_llama2c(path):
    """Read a llama2.c version-0 fp32 checkpoint into (config, dict of numpy tensors)."""
    with open(path, "rb") as f:
        dim, hid, nl, nh, nkv, vocab, seq = struct.unpack("<7i", f.read(28))
        shared = vocab > 0
        vocab = abs(vocab)
        c = dict(dim=dim, hidden_dim=hid, n_layers=nl, n_heads=nh, n_kv_heads=nkv,
                 vocab_size=vocab, seq_len=seq, shared=shared)
        hs, kvd = dim // nh, dim * nkv // nh

        def rd(*shape):
            n = int(np.prod(shape))
            return np.frombuffer(f.read(4 * n), dtype="<f4").reshape(shape).copy()

        w = {}
        w["tok_emb"] = rd(vocab, dim)
        w["rms_att"] = rd(nl, dim)
        w["wq"] = rd(nl, dim, dim)
        w["wk"] = rd(nl, kvd, dim)
        w["wv"] = rd(nl, kvd, dim)
        w["wo"] = rd(nl, dim, dim)
        w["rms_ffn"] = rd(nl, dim)
        w["w1"] = rd(nl, hid, dim)
        w["w2"] = rd(nl, dim, hid)
        w["w3"] = rd(nl, hid, dim)
        w["rms_final"] = rd(dim)
        w["rope_cos"] = rd(seq, hs // 2)
        w["rope_sin"] = rd(seq, hs // 2)
        w["wcls"] = w["tok_emb"] if shared else rd(vocab, dim)
        assert f.read(1) == b"", "trailing bytes in checkpoint"
    return c, w


def load_flat(path):
    """Read the flat FPGA weight file into the same (config, tensors) form as load_llama2c."""
    raw = open(path, "rb").read()
    c, o = layout.unpack_header(raw)
    assert len(raw) == o["total"]

    def rd(off, *shape):
        n = int(np.prod(shape))
        return np.frombuffer(raw, dtype="<f4", count=n, offset=off).reshape(shape).copy()

    nl = c["n_layers"]
    w = {"tok_emb": rd(o["tok_emb"], c["vocab_size"], c["dim"])}
    for name, shape in layout.layer_tensors(c):
        w[name] = np.stack([
            rd(o["layers_start"] + l * o["layer_stride"] + o["layer_rel"][name], *shape) for l in range(nl)])
    hs = layout.head_size(c)
    w["rms_final"] = rd(o["final_norm"], c["dim"])
    w["rope_cos"] = rd(o["rope_cos"], c["seq_len"], hs // 2)
    w["rope_sin"] = rd(o["rope_sin"], c["seq_len"], hs // 2)
    w["wcls"] = w["tok_emb"] if c["shared"] else rd(o["wcls"], c["vocab_size"], c["dim"])
    return c, w


def load(path):
    """Load either format; the flat file is recognised by its magic bytes, anything else is read as a llama2.c checkpoint."""
    with open(path, "rb") as f:
        magic = f.read(4)
    return load_flat(path) if magic == layout.MAGIC else load_llama2c(path)


# ---------------------------------------------------------------- model
def rmsnorm(x, wt):
    """RMSNorm of one vector: w * x / sqrt(mean(x^2) + 1e-5)."""
    ss = np.float32(np.sum(x * x, dtype=np.float32) / np.float32(x.size) + np.float32(1e-5))
    return (wt * (np.float32(1.0) / np.sqrt(ss)) * x).astype(np.float32)


def softmax(x):
    """Numerically stable softmax of a vector."""
    e = np.exp(x - np.max(x)).astype(np.float32)
    return e / np.sum(e, dtype=np.float32)


def rope(v, cos, sin, hs):
    """llama2.c rotation: consecutive pairs (2i, 2i+1) within each head."""
    v = v.reshape(-1, hs // 2, 2)
    a, b = v[..., 0], v[..., 1]
    out = np.empty_like(v)
    out[..., 0] = a * cos - b * sin
    out[..., 1] = a * sin + b * cos
    return out.reshape(-1)


class Model:
    def __init__(self, cfg, w):
        self.c, self.w = cfg, w
        kvd = layout.kv_dim(cfg)
        self.k = np.zeros((cfg["n_layers"], cfg["seq_len"], kvd), np.float32)
        self.v = np.zeros_like(self.k)

    def forward(self, token, pos):
        c, w = self.c, self.w
        hs, kvd = layout.head_size(c), layout.kv_dim(c)
        kv_mul = c["n_heads"] // c["n_kv_heads"]
        x = w["tok_emb"][token].copy()
        cos, sin = w["rope_cos"][pos], w["rope_sin"][pos]
        for l in range(c["n_layers"]):
            xb = rmsnorm(x, w["rms_att"][l])
            q = w["wq"][l] @ xb
            k = w["wk"][l] @ xb
            v = w["wv"][l] @ xb
            q = rope(q, cos, sin, hs)
            k = rope(k, cos, sin, hs)
            self.k[l, pos], self.v[l, pos] = k, v
            att_out = np.empty(c["dim"], np.float32)
            for h in range(c["n_heads"]):
                g = h // kv_mul
                qh = q[h * hs:(h + 1) * hs]
                kh = self.k[l, :pos + 1, g * hs:(g + 1) * hs]
                vh = self.v[l, :pos + 1, g * hs:(g + 1) * hs]
                s = softmax((kh @ qh) / np.float32(np.sqrt(hs)))
                att_out[h * hs:(h + 1) * hs] = s @ vh
            x = x + w["wo"][l] @ att_out
            xb = rmsnorm(x, w["rms_ffn"][l])
            h1, h3 = w["w1"][l] @ xb, w["w3"][l] @ xb
            h1 = h1 * (np.float32(1.0) / (np.float32(1.0) + np.exp(-h1))) * h3   # SiLU(h1) * h3
            x = x + w["w2"][l] @ h1
        x = rmsnorm(x, w["rms_final"])
        return (w["wcls"] @ x).astype(np.float32)


def generate(cfg, w, prompt, steps):
    """Greedy decode. Returns (tokens, logits list) - logits[i] is the output after feeding tokens[i]."""
    m = Model(cfg, w)
    toks, logits = list(prompt), []
    for pos in range(len(prompt) + steps - 1):
        lg = m.forward(toks[pos], pos)
        logits.append(lg)
        if pos >= len(prompt) - 1:
            toks.append(int(np.argmax(lg)))
    return toks, logits


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("--prompt", type=int, nargs="+", default=[1, 5, 9])
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--golden")
    a = p.parse_args()
    cfg, w = load(a.model)
    toks, logits = generate(cfg, w, a.prompt, a.steps)
    print("tokens:", toks)
    if a.golden:
        with open(a.golden, "w") as f:
            json.dump({"prompt": a.prompt, "tokens": toks,
                       "logits": [lg.tolist() for lg in logits]}, f)
        print("wrote", a.golden)
