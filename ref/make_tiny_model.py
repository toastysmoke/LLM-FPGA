"""Generate a random-weight model in llama2.c checkpoint format (version 0, fp32).

Used to test the RTL on a model small enough to simulate in seconds, including grouped-query
attention (--kv-heads smaller than --heads) and a separate classifier (--shared 0).
"""
import argparse
import struct
import numpy as np


def main():
    """Command line: write a random llama2.c-format model of any size (for tests)."""
    p = argparse.ArgumentParser()
    p.add_argument("out")
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--hidden", type=int, default=172)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--kv-heads", type=int, default=4)
    p.add_argument("--vocab", type=int, default=256)
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--shared", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    assert a.dim % a.heads == 0 and a.heads % a.kv_heads == 0
    hs = a.dim // a.heads
    assert hs % 2 == 0
    kvd = a.dim * a.kv_heads // a.heads
    rng = np.random.default_rng(a.seed)

    def mat(*shape, fan_in):
        return (rng.standard_normal(shape) / np.sqrt(fan_in)).astype(np.float32)

    def norm(*shape):
        return (1.0 + 0.1 * rng.standard_normal(shape)).astype(np.float32)

    L, d, h = a.layers, a.dim, a.hidden
    tok = (0.5 * rng.standard_normal((a.vocab, d))).astype(np.float32)
    rms_att = norm(L, d)
    wq = mat(L, d, d, fan_in=d)
    wk = mat(L, kvd, d, fan_in=d)
    wv = mat(L, kvd, d, fan_in=d)
    wo = mat(L, d, d, fan_in=d)
    rms_ffn = norm(L, d)
    w1 = mat(L, h, d, fan_in=d)
    w2 = mat(L, d, h, fan_in=h)
    w3 = mat(L, h, d, fan_in=d)
    rms_final = norm(d)

    freqs = 1.0 / (10000.0 ** (np.arange(0, hs, 2) / hs))
    ang = np.outer(np.arange(a.seq_len), freqs)
    cos = np.cos(ang).astype(np.float32)
    sin = np.sin(ang).astype(np.float32)

    with open(a.out, "wb") as f:
        vocab_field = a.vocab if a.shared else -a.vocab
        f.write(struct.pack("<7i", d, h, L, a.heads, a.kv_heads, vocab_field, a.seq_len))
        for t in (tok, rms_att, wq, wk, wv, wo, rms_ffn, w1, w2, w3, rms_final, cos, sin):
            f.write(t.tobytes())
        if not a.shared:
            f.write(mat(a.vocab, d, fan_in=d).tobytes())
    print(f"wrote {a.out}: dim={d} hidden={h} layers={L} heads={a.heads} kv_heads={a.kv_heads} "
          f"vocab={a.vocab} seq={a.seq_len} shared={a.shared}")


if __name__ == "__main__":
    main()
