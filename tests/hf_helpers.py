"""Shared helpers for the hf_to_flat tests: build Hugging Face style test models and run a numpy implementation of the
Hugging Face Llama convention (rotate_half) to compare against."""
import json
import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ref"))
import layout  # noqa: E402
import reference  # noqa: E402


def quant(arr, dtype):
    """the fp32 values a tensor holds after being stored in `dtype`"""
    arr = np.ascontiguousarray(arr, dtype="<f4")
    if dtype == "BF16":
        return (((arr.view("<u4") >> 16).astype(np.uint32)) << 16).view("<f4")
    if dtype == "F16":
        return arr.astype("<f2").astype("<f4")
    return arr


def write_safetensors(path, tensors, dtype="F32", override=None):
    hdr, blobs, off = {}, [], 0
    for name, arr in tensors.items():
        dt = (override or {}).get(name, dtype)
        arr = np.ascontiguousarray(arr, dtype="<f4")
        if dt == "F32":
            raw = arr.tobytes()
        elif dt == "F16":
            raw = arr.astype("<f2").tobytes()
        elif dt == "BF16":
            raw = (arr.view("<u4") >> 16).astype("<u2").tobytes()
        else:
            raw = arr.astype("<i1").tobytes()
        hdr[name] = {"dtype": dt, "shape": list(arr.shape), "data_offsets": [off, off + len(raw)]}
        blobs.append(raw)
        off += len(raw)
    h = json.dumps(hdr).encode()
    Path(path).write_bytes(struct.pack("<Q", len(h)) + h + b"".join(blobs))


def write_hf(folder, cfg, tensors, dtype="F32", shards=1, extra_cfg=None, override=None):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    c = {"model_type": "llama", "hidden_size": cfg["dim"], "intermediate_size": cfg["hidden_dim"], "num_hidden_layers": cfg["n_layers"],
         "num_attention_heads": cfg["n_heads"], "num_key_value_heads": cfg["n_kv_heads"], "vocab_size": cfg["vocab_size"],
         "max_position_embeddings": cfg["seq_len"], "rms_norm_eps": 1e-5, "hidden_act": "silu", "rope_theta": 10000.0,
         "tie_word_embeddings": bool(cfg["shared"])}
    c.update(extra_cfg or {})
    (folder / "config.json").write_text(json.dumps(c), encoding="utf-8")
    names = list(tensors)
    for i in range(shards):
        part = {n: tensors[n] for n in names[i::shards]}
        write_safetensors(folder / (f"model-{i + 1:05d}.safetensors" if shards > 1 else "model.safetensors"), part, dtype, override)
    return folder


def to_meta_to_hf(w, n_heads):
    """llama2.c (adjacent pairs) -> Hugging Face (halves): the inverse of hf_to_flat.interleave"""
    d1, d2 = w.shape
    return w.reshape(n_heads, d1 // n_heads // 2, 2, d2).transpose(0, 2, 1, 3).reshape(d1, d2)


def hf_tensors_from_llama2c(cfg, w, prefix="model."):
    t = {prefix + "embed_tokens.weight": w["tok_emb"]}
    for l in range(cfg["n_layers"]):
        p = f"{prefix}layers.{l}."
        t[p + "input_layernorm.weight"] = w["rms_att"][l]
        t[p + "self_attn.q_proj.weight"] = to_meta_to_hf(w["wq"][l], cfg["n_heads"])
        t[p + "self_attn.k_proj.weight"] = to_meta_to_hf(w["wk"][l], cfg["n_kv_heads"])
        t[p + "self_attn.v_proj.weight"] = w["wv"][l]
        t[p + "self_attn.o_proj.weight"] = w["wo"][l]
        t[p + "post_attention_layernorm.weight"] = w["rms_ffn"][l]
        t[p + "mlp.gate_proj.weight"] = w["w1"][l]
        t[p + "mlp.down_proj.weight"] = w["w2"][l]
        t[p + "mlp.up_proj.weight"] = w["w3"][l]
    t[prefix + "norm.weight"] = w["rms_final"]
    if not cfg["shared"]:
        t["lm_head.weight"] = w["wcls"]
    return t


def random_model(seed=0, shared=False, prefix="model."):
    rng = np.random.default_rng(seed)
    cfg = dict(dim=32, hidden_dim=48, n_layers=2, n_heads=4, n_kv_heads=2, vocab_size=64, seq_len=16, shared=shared)
    d, h, k = cfg["dim"], cfg["hidden_dim"], layout.kv_dim(cfg)
    f = lambda *s: (0.25 * rng.standard_normal(s)).astype(np.float32)
    nrm = lambda n: (1.0 + 0.1 * rng.standard_normal(n)).astype(np.float32)
    t = {prefix + "embed_tokens.weight": f(cfg["vocab_size"], d)}
    for l in range(cfg["n_layers"]):
        p = f"{prefix}layers.{l}."
        t.update({p + "input_layernorm.weight": nrm(d), p + "self_attn.q_proj.weight": f(d, d), p + "self_attn.k_proj.weight": f(k, d),
                  p + "self_attn.v_proj.weight": f(k, d), p + "self_attn.o_proj.weight": f(d, d),
                  p + "post_attention_layernorm.weight": nrm(d), p + "mlp.gate_proj.weight": f(h, d),
                  p + "mlp.down_proj.weight": f(d, h), p + "mlp.up_proj.weight": f(h, d)})
    t[prefix + "norm.weight"] = nrm(d)
    if not shared:
        t["lm_head.weight"] = f(cfg["vocab_size"], d)
    return cfg, t


def hf_logits(cfg, t, tokens, prefix="model.", dtype="F32", theta=10000.0):
    """numpy implementation of the Hugging Face Llama forward (rotate_half convention), float64. Returns [T, vocab]."""
    q_ = lambda n: quant(t[n], dtype).astype(np.float64)
    nh, nkv, dim = cfg["n_heads"], cfg["n_kv_heads"], cfg["dim"]
    hs = dim // nh
    T = len(tokens)
    rms = lambda x, w: x / np.sqrt((x * x).mean(-1, keepdims=True) + 1e-5) * w
    inv = 1.0 / theta ** (np.arange(0, hs, 2) / hs)
    ang = np.outer(np.arange(T), inv)
    cos, sin = np.cos(np.concatenate([ang, ang], -1)), np.sin(np.concatenate([ang, ang], -1))
    rot = lambda x: np.concatenate([-x[..., hs // 2:], x[..., : hs // 2]], -1)
    x = q_(prefix + "embed_tokens.weight")[tokens]
    for l in range(cfg["n_layers"]):
        p = f"{prefix}layers.{l}."
        h = rms(x, q_(p + "input_layernorm.weight"))
        q = (h @ q_(p + "self_attn.q_proj.weight").T).reshape(T, nh, hs)
        k = (h @ q_(p + "self_attn.k_proj.weight").T).reshape(T, nkv, hs)
        v = (h @ q_(p + "self_attn.v_proj.weight").T).reshape(T, nkv, hs)
        q = q * cos[:, None, :] + rot(q) * sin[:, None, :]
        k = k * cos[:, None, :] + rot(k) * sin[:, None, :]
        att = np.zeros((T, nh, hs))
        for hh in range(nh):
            g = hh // (nh // nkv)
            s = q[:, hh] @ k[:, g].T / np.sqrt(hs)
            s = np.where(np.tril(np.ones((T, T), bool)), s, -1e30)
            e = np.exp(s - s.max(-1, keepdims=True))
            att[:, hh] = (e / e.sum(-1, keepdims=True)) @ v[:, g]
        x = x + att.reshape(T, dim) @ q_(p + "self_attn.o_proj.weight").T
        h = rms(x, q_(p + "post_attention_layernorm.weight"))
        gate, up = h @ q_(p + "mlp.gate_proj.weight").T, h @ q_(p + "mlp.up_proj.weight").T
        x = x + (gate / (1 + np.exp(-gate)) * up) @ q_(p + "mlp.down_proj.weight").T
    x = rms(x, q_(prefix + "norm.weight"))
    head = "lm_head.weight" if "lm_head.weight" in t else prefix + "embed_tokens.weight"
    return x @ q_(head).T


def our_logits(flat, tokens):
    c, w = reference.load_flat(flat)
    return np.array(reference.generate(c, w, tokens, 1)[1], dtype=np.float64)
