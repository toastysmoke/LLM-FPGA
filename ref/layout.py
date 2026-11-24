"""Flat weight-file layout (version 1). Single source of truth for converter, reference and (later) RTL.

File = 256-byte header, then sections, each aligned to 64 bytes. All tensors fp32, row-major,
matrices stored [out, in] (y = W @ x), same as llama2.c.

  header | token_embedding | layer 0 | layer 1 | ... | final_norm | rope_cos | rope_sin | wcls (optional)

Each layer has a fixed stride and the tensors inside appear in this order, so a layer streams
sequentially from HBM:  rms_att, wq, wk, wv, wo, rms_ffn, w1, w2, w3
"""
import struct

ALIGN = 64
HEADER_BYTES = 256
MAGIC = b"LLMF"
VERSION = 1
DTYPE_FP32 = 0
ELEM = 4

HDR_PARAMS = "<4sI9I"   # magic, version, dtype, dim, hidden, layers, heads, kv_heads, vocab, seq_len, shared
HDR_OFFSETS = "<8Q"     # tok_emb, layers_start, layer_stride, final_norm, rope_cos, rope_sin, wcls, total


def align(n):
    """Round a byte count up to the 64-byte alignment used for every section and tensor."""
    return (n + ALIGN - 1) // ALIGN * ALIGN


def head_size(c):
    """Size of one attention head (dim / n_heads)."""
    return c["dim"] // c["n_heads"]


def kv_dim(c):
    """Width of the K or V vector of one position (dim * n_kv_heads / n_heads); smaller than dim for grouped-query attention."""
    return c["dim"] * c["n_kv_heads"] // c["n_heads"]


def layer_tensors(c):
    """The nine tensors of one layer in file order, as (name, shape): rms_att wq wk wv wo rms_ffn w1 w2 w3."""
    d, h, k = c["dim"], c["hidden_dim"], kv_dim(c)
    return [
        ("rms_att", (d,)),
        ("wq", (d, d)),
        ("wk", (k, d)),
        ("wv", (k, d)),
        ("wo", (d, d)),
        ("rms_ffn", (d,)),
        ("w1", (h, d)),
        ("w2", (d, h)),
        ("w3", (h, d)),
    ]


def nbytes(shape):
    """Size in bytes of a tensor shape in fp32."""
    n = ELEM
    for s in shape:
        n *= s
    return n


def compute(c):
    """Return dict of absolute byte offsets plus per-layer relative tensor offsets."""
    o = {}
    o["tok_emb"] = HEADER_BYTES
    o["layers_start"] = align(o["tok_emb"] + nbytes((c["vocab_size"], c["dim"])))

    rel, pos = {}, 0
    for name, shape in layer_tensors(c):
        rel[name] = pos
        pos = align(pos + nbytes(shape))
    o["layer_stride"] = pos
    o["layer_rel"] = rel

    o["final_norm"] = align(o["layers_start"] + c["n_layers"] * o["layer_stride"])
    rope_bytes = nbytes((c["seq_len"], head_size(c) // 2))
    o["rope_cos"] = align(o["final_norm"] + nbytes((c["dim"],)))
    o["rope_sin"] = align(o["rope_cos"] + rope_bytes)
    end = o["rope_sin"] + rope_bytes
    if c["shared"]:
        o["wcls"] = 0
    else:
        o["wcls"] = align(end)
        end = o["wcls"] + nbytes((c["vocab_size"], c["dim"]))
    o["total"] = align(end)
    return o


def pack_header(c, o):
    """Build the 256-byte header (magic, version, dtype, sizes, section offsets) from a config and the computed offsets."""
    hdr = struct.pack(HDR_PARAMS, MAGIC, VERSION, DTYPE_FP32, c["dim"], c["hidden_dim"], c["n_layers"],
                      c["n_heads"], c["n_kv_heads"], c["vocab_size"], c["seq_len"], int(c["shared"]))
    hdr += struct.pack(HDR_OFFSETS, o["tok_emb"], o["layers_start"], o["layer_stride"], o["final_norm"],
                       o["rope_cos"], o["rope_sin"], o["wcls"], o["total"])
    return hdr.ljust(HEADER_BYTES, b"\0")


def unpack_header(buf):
    """Parse a 256-byte header back into (config, offsets). Asserts the magic, version and dtype, and that the stored section offsets equal the ones computed from the config."""
    n1 = struct.calcsize(HDR_PARAMS)
    magic, ver, dtype, dim, hid, nl, nh, nkv, vocab, seq, shared = struct.unpack_from(HDR_PARAMS, buf, 0)
    assert magic == MAGIC, "bad magic"
    assert ver == VERSION and dtype == DTYPE_FP32
    c = dict(dim=dim, hidden_dim=hid, n_layers=nl, n_heads=nh, n_kv_heads=nkv,
             vocab_size=vocab, seq_len=seq, shared=bool(shared))
    names = ["tok_emb", "layers_start", "layer_stride", "final_norm", "rope_cos", "rope_sin", "wcls", "total"]
    vals = struct.unpack_from(HDR_OFFSETS, buf, n1)
    stored = dict(zip(names, vals))
    # recompute from the config and make sure the file agrees
    o = compute(c)
    for k in names:
        assert o[k] == stored[k], f"header offset mismatch for {k}: {o[k]} vs {stored[k]}"
    return c, o
