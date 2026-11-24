"""Build synthetic Hugging Face model folders (config.json + model.safetensors) for testing model_to_fpga_spec.py.
The script only reads the safetensors HEADER (names, shapes, dtypes), so by default no tensor data is written (the file is a
header-only, technically truncated safetensors). Use with_data=True for tiny models when --pack is tested.

usage: python make_fake_hf.py <kind> <out_dir>      kinds: tiny, tinyllama, qknorm, parallel
"""
import json
import struct
import sys
from pathlib import Path

BYTES = {"F32": 4, "F16": 2, "BF16": 2}


def write_safetensors(path, tensors, dtype="BF16", with_data=False):
    """tensors: list of (name, shape). Writes the header (and zero data if requested)."""
    hdr, off = {}, 0
    for name, shape in tensors:
        n = 1
        for s in shape:
            n *= s
        nb = n * BYTES[dtype]
        hdr[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [off, off + nb]}
        off += nb
    raw = json.dumps(hdr).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(raw)))
        f.write(raw)
        if with_data:
            f.write(bytes(off))


def llama_tensors(layers, hidden, heads, kv_heads, ffn, vocab, qk_norm=False, tied=True, norms_per_layer=2):
    hd = hidden // heads
    t = [("model.embed_tokens.weight", (vocab, hidden))]
    for l in range(layers):
        p = f"model.layers.{l}."
        t.append((p + "input_layernorm.weight", (hidden,)))
        t += [(p + "self_attn.q_proj.weight", (heads * hd, hidden)),
              (p + "self_attn.k_proj.weight", (kv_heads * hd, hidden)),
              (p + "self_attn.v_proj.weight", (kv_heads * hd, hidden)),
              (p + "self_attn.o_proj.weight", (hidden, heads * hd))]
        if qk_norm:
            t += [(p + "self_attn.q_norm.weight", (hd,)), (p + "self_attn.k_norm.weight", (hd,))]
        if norms_per_layer == 2:
            t.append((p + "post_attention_layernorm.weight", (hidden,)))
        t += [(p + "mlp.gate_proj.weight", (ffn, hidden)),
              (p + "mlp.up_proj.weight", (ffn, hidden)),
              (p + "mlp.down_proj.weight", (hidden, ffn))]
    t.append(("model.norm.weight", (hidden,)))
    if not tied:
        t.append(("lm_head.weight", (vocab, hidden)))
    return t


def parallel_tensors(layers, hidden, heads, kv_heads, ffn, vocab):
    """One norm per layer, ungated FFN (up/down): the block a user would describe with a parallel-block recipe."""
    hd = hidden // heads
    t = [("model.embed_tokens.weight", (vocab, hidden))]
    for l in range(layers):
        p = f"model.layers.{l}."
        t.append((p + "input_layernorm.weight", (hidden,)))
        t += [(p + "self_attn.q_proj.weight", (heads * hd, hidden)),
              (p + "self_attn.k_proj.weight", (kv_heads * hd, hidden)),
              (p + "self_attn.v_proj.weight", (kv_heads * hd, hidden)),
              (p + "self_attn.o_proj.weight", (hidden, heads * hd)),
              (p + "mlp.up_proj.weight", (ffn, hidden)),
              (p + "mlp.down_proj.weight", (hidden, ffn))]
    t.append(("model.norm.weight", (hidden,)))
    return t


def config(model_type, layers, hidden, heads, kv_heads, ffn, vocab, ctx, act="silu", tied=True):
    return {"model_type": model_type, "num_hidden_layers": layers, "hidden_size": hidden, "num_attention_heads": heads,
            "num_key_value_heads": kv_heads, "intermediate_size": ffn, "vocab_size": vocab, "max_position_embeddings": ctx,
            "rms_norm_eps": 1e-5, "hidden_act": act, "rope_theta": 10000.0, "tie_word_embeddings": tied,
            "torch_dtype": "bfloat16"}


def build(kind, out, with_data=False):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if kind == "tiny":
        dims = dict(layers=2, hidden=64, heads=4, kv_heads=2, ffn=128, vocab=256)
        ctx, tensors, cfg = 128, llama_tensors(**dims), config("llama", ctx=128, **dims)
    elif kind == "tinyllama":                      # TinyLlama-1.1B shapes
        dims = dict(layers=22, hidden=2048, heads=32, kv_heads=4, ffn=5632, vocab=32000)
        tensors, cfg = llama_tensors(tied=False, **dims), config("llama", ctx=2048, tied=False, **dims)
    elif kind == "qknorm":
        dims = dict(layers=4, hidden=256, heads=8, kv_heads=4, ffn=512, vocab=1000)
        tensors, cfg = llama_tensors(qk_norm=True, **dims), config("qwen3", ctx=512, **dims)
    elif kind == "parallel":
        dims = dict(layers=3, hidden=128, heads=4, kv_heads=4, ffn=512, vocab=500)
        tensors, cfg = parallel_tensors(**dims), config("custom_parallel", ctx=256, act="gelu", **dims)
    else:
        raise SystemExit(f"unknown kind {kind}")
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    write_safetensors(out / "model.safetensors", tensors, with_data=with_data)
    return out


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    print(build(sys.argv[1], sys.argv[2]))
