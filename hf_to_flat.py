#!/usr/bin/env python3
"""
hf_to_flat.py - convert a Hugging Face Llama-style model folder into the flat FPGA weight file (layout in ref/layout.py).

  python hf_to_flat.py ./my-model out.flat [--seq-len N] [--board boards/u50.json] [--kernel-limits dim=512,seq=1024]

Input : a folder with config.json and one or more *.safetensors files (F32, F16 or BF16 weights).
Output: out.flat, all tensors in fp32, in the order the kernel streams them (the same file ref/convert.py makes from a
        llama2.c checkpoint). Needs numpy. Tensors are copied one at a time, so a large model does not have to fit in memory.

What the conversion does besides renaming and widening to fp32:
  * Hugging Face stores the query and key matrices with the rotary pairs split in half (rotate_half); the kernel (like
    llama2.c) rotates adjacent pairs. The rows of wq and wk are reordered to match. Without this the model runs but is wrong.
  * The rotary cos/sin tables are not in a Hugging Face model; they are computed from rope_theta.
  * The context length (seq_len) is max_position_embeddings unless --seq-len gives a smaller one. It sets the K/V cache size.

Before writing it prints a FIT CHECK (fit_check.py): does the model fit the kernel's built-in size limits, and the board's memory
(--board)? It is information only: the file is written either way, so it can be used with a rebuilt kernel.

Refused with a message (the kernel cannot run them): attention or MLP biases, query/key norms, mixture-of-experts, a
non-SiLU activation, rope scaling or partial rotary, a head size different from hidden/heads, sliding-window attention
shorter than the context, quantized weights.
Warns: rms_norm_eps other than 1e-5 (the kernel uses 1e-5 for every model).
"""
import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "ref"))
import layout  # noqa: E402
import fit_check  # noqa: E402

DTYPES = {"F32": np.dtype("<f4"), "F16": np.dtype("<f2"), "BF16": np.dtype("<u2")}
LAYER_MAP = {"rms_att": "input_layernorm.weight", "wq": "self_attn.q_proj.weight", "wk": "self_attn.k_proj.weight",
             "wv": "self_attn.v_proj.weight", "wo": "self_attn.o_proj.weight", "rms_ffn": "post_attention_layernorm.weight",
             "w1": "mlp.gate_proj.weight", "w2": "mlp.down_proj.weight", "w3": "mlp.up_proj.weight"}
IGNORED = ("rotary_emb.inv_freq",)      # buffers some older checkpoints store; recomputed here


def fail(msg):
    """Stop with a one-line error message prefixed by the tool name."""
    raise SystemExit("hf_to_flat: " + msg)


def index_safetensors(folder):
    """tensor name -> where it is: only the headers are read."""
    files = sorted(folder.glob("*.safetensors"))
    if not files:
        fail(f"no .safetensors files in {folder}")
    idx = {}
    for f in files:
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            hdr = json.loads(fh.read(n))
        for name, m in hdr.items():
            if name != "__metadata__":
                b, e = m["data_offsets"]
                idx[name] = {"file": f, "dtype": m["dtype"], "shape": tuple(m["shape"]), "start": 8 + n + b, "end": 8 + n + e}
    return idx


def read_fp32(name, t):
    """Read one tensor from its safetensors file and return it as fp32, widening F16 or BF16 (F32 is used as is)."""
    dt = DTYPES.get(t["dtype"])
    if dt is None:
        fail(f"{name}: data type {t['dtype']} is not supported (use F32, F16 or BF16 weights)")
    count = int(np.prod(t["shape"]))
    a = np.fromfile(t["file"], dtype=dt, count=count, offset=t["start"])
    if a.size != count:
        fail(f"{name}: {t['file'].name} is shorter than its header says")
    a = (a.astype(np.uint32) << 16).view("<f4") if t["dtype"] == "BF16" else a.astype("<f4")      # bf16 = top half of an fp32
    return a.reshape(t["shape"])


def interleave(w, n_heads):
    """rows of a q/k matrix: Hugging Face order (each head's pairs split in two halves) -> adjacent pairs, as the kernel rotates."""
    d1, d2 = w.shape
    return w.reshape(n_heads, 2, d1 // n_heads // 2, d2).transpose(0, 2, 1, 3).reshape(d1, d2)


def llama3_scale(inv, scaling):
    """the Llama 3 / 3.1 / 3.2 rope scaling of the inverse frequencies (same rule as Hugging Face's _compute_llama3_parameters): frequencies whose
    wavelength is shorter than original_max_position / high_freq_factor stay as they are, those longer than original_max_position / low_freq_factor
    are divided by `factor`, and the ones in between are blended smoothly."""
    factor, low, high = float(scaling["factor"]), float(scaling["low_freq_factor"]), float(scaling["high_freq_factor"])
    old = float(scaling["original_max_position_embeddings"])
    low_wavelen, high_wavelen = old / low, old / high
    wavelen = np.float32(2.0 * np.pi) / inv
    out = np.where(wavelen > low_wavelen, inv / np.float32(factor), inv)
    smooth = (np.float32(old) / wavelen - np.float32(low)) / np.float32(high - low)
    blended = (np.float32(1.0) - smooth) * inv / np.float32(factor) + smooth * inv
    medium = ~(wavelen < high_wavelen) & ~(wavelen > low_wavelen)
    return np.where(medium, blended, out).astype(np.float32)


def rope_tables(theta, hs, seq, scaling=None):
    """cos/sin of position * theta^(-2i/hs), shape [seq, hs/2], fp32. `scaling`: the model's rope_scaling dict, only the llama3 kind is supported."""
    inv = np.float32(1.0) / (np.float32(theta) ** (np.arange(0, hs, 2, dtype=np.float32)[: hs // 2] / np.float32(hs)))
    if scaling:
        inv = llama3_scale(inv, scaling)
    ang = np.outer(np.arange(seq, dtype=np.float32), inv).astype(np.float32)
    return np.cos(ang).astype("<f4"), np.sin(ang).astype("<f4")


def read_config(folder, seq_len):
    """Read config.json and return the model sizes in flat-file header form, the RoPE settings, and lists of errors and warnings (unsupported features are reported, not guessed)."""
    path = folder / "config.json"
    if not path.exists():
        fail(f"no config.json in {folder}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    c = raw.get("text_config", raw)
    need = ["hidden_size", "num_hidden_layers", "num_attention_heads", "intermediate_size", "vocab_size", "max_position_embeddings"]
    missing = [k for k in need if not c.get(k)]
    if missing:
        fail(f"config.json is missing {missing}")
    heads, dim = c["num_attention_heads"], c["hidden_size"]
    kv = c.get("num_key_value_heads") or heads
    errors, warnings = [], []
    if dim % heads or heads % kv or (dim // heads) % 2:
        errors.append(f"hidden_size {dim} / heads {heads} / kv heads {kv} do not divide evenly into an even head size")
    if c.get("head_dim") and c["head_dim"] != dim // heads:
        errors.append(f"head_dim {c['head_dim']} differs from hidden_size / heads = {dim // heads}")
    if (c.get("hidden_act") or c.get("hidden_activation") or "silu") != "silu":
        errors.append(f"activation {c.get('hidden_act') or c.get('hidden_activation')} (the kernel implements SiLU only)")
    rs = c.get("rope_scaling") or (c.get("rope_parameters") or {}).get("rope_scaling")
    scaling = None
    if rs and (rs.get("rope_type") or rs.get("type")) not in (None, "default"):
        if (rs.get("rope_type") or rs.get("type")) == "llama3" and all(k in rs for k in ("factor", "low_freq_factor", "high_freq_factor", "original_max_position_embeddings")):
            scaling = rs      # Llama 3.x: the tables are computed with the scaled frequencies (llama3_scale)
        else:
            errors.append(f"rope scaling {rs} (only the llama3 kind is supported)")
    if c.get("partial_rotary_factor", 1.0) != 1.0:
        errors.append("partial rotary (partial_rotary_factor != 1)")
    if c.get("attention_bias") or c.get("mlp_bias"):
        errors.append("attention or MLP biases (the kernel has none)")
    if c.get("num_local_experts") or c.get("num_experts"):
        errors.append("mixture-of-experts")
    if c.get("quantization_config") or raw.get("quantization_config"):
        errors.append("quantized weights")
    seq = seq_len or c["max_position_embeddings"]
    if seq > c["max_position_embeddings"]:
        errors.append(f"--seq-len {seq} is longer than the model's max_position_embeddings {c['max_position_embeddings']}")
    if seq < 1:
        errors.append("--seq-len must be at least 1")
    if c.get("sliding_window") and seq > c["sliding_window"]:
        errors.append(f"sliding-window attention of {c['sliding_window']} is shorter than the context {seq}; use --seq-len {c['sliding_window']} or less")
    eps = c.get("rms_norm_eps")
    if eps is not None and abs(eps - 1e-5) > 1e-12:
        warnings.append(f"rms_norm_eps is {eps}; the kernel uses 1e-5, so results differ slightly from the original model")
    theta = c.get("rope_theta") or (c.get("rope_parameters") or {}).get("rope_theta") or 10000.0
    cfg = dict(dim=dim, hidden_dim=c["intermediate_size"], n_layers=c["num_hidden_layers"], n_heads=heads, n_kv_heads=kv,
               vocab_size=c["vocab_size"], seq_len=seq, shared=bool(c.get("tie_word_embeddings", False)))
    return cfg, theta, scaling, errors, warnings


def convert(folder, dst, seq_len=None, log=print, board=None, kernel_limits=None):
    """Convert a Hugging Face model folder into the flat file dst: validate the config, then copy the tensors one at a time (so a large model need not fit in memory), reordering the Q and K rows into the rotary layout the kernel expects."""
    folder, dst = Path(folder), Path(dst)
    if not folder.is_dir():
        fail(f"{folder} must be a Hugging Face model folder (config.json + *.safetensors)")
    cfg, theta, scaling, errors, warnings = read_config(folder, seq_len)
    idx = index_safetensors(folder)
    bad = sorted(n for n in idx if n.endswith(".bias") or "q_norm" in n or "k_norm" in n or "experts" in n)
    if bad:
        errors.append(f"tensors the kernel cannot use: {bad[:4]}{' ...' if len(bad) > 4 else ''}")
    if errors:
        fail("this model cannot be converted:\n  - " + "\n  - ".join(errors))
    prefix = "model." if "model.embed_tokens.weight" in idx else ("" if "embed_tokens.weight" in idx else None)
    if prefix is None:
        fail("no embed_tokens.weight in the safetensors files: not a Hugging Face Llama-style model?")
    if "lm_head.weight" not in idx:
        cfg["shared"] = True
    o = layout.compute(cfg)
    log(f"{cfg['n_layers']} layers, dim {cfg['dim']}, hidden {cfg['hidden_dim']}, {cfg['n_heads']} heads / {cfg['n_kv_heads']} kv, "
        f"vocab {cfg['vocab_size']}, seq_len {cfg['seq_len']}, output head {'shared with the embedding' if cfg['shared'] else 'separate'}; "
        f"output {o['total'] / 2**20:,.1f} MB (fp32)")
    for w in warnings:
        log("WARNING: " + w)
    ext = None
    if board:
        from model_to_fpga_spec import load_board
        ext = load_board(board)["ext_bytes"]
    for line in fit_check.report_lines(fit_check.check(cfg, o, fit_check.kernel_limits(override=kernel_limits), ext)):
        log(line)                       # printed before the (possibly large) file is written

    used = {"lm_head.weight"} if cfg["shared"] else set()      # a tied model's own lm_head copy is ignored, as Hugging Face does

    def get(name, shape):
        if name not in idx:
            fail(f"tensor {name} not found")
        t = idx[name]
        if t["shape"] != tuple(shape):
            fail(f"{name}: shape {t['shape']}, expected {tuple(shape)}")
        used.add(name)
        return read_fp32(name, t)

    with open(dst, "wb") as f:
        f.truncate(o["total"])
        f.seek(0)
        f.write(layout.pack_header(cfg, o))

        def put(off, arr):
            f.seek(off)
            f.write(np.ascontiguousarray(arr, dtype="<f4").tobytes())

        put(o["tok_emb"], get(prefix + "embed_tokens.weight", (cfg["vocab_size"], cfg["dim"])))
        for l in range(cfg["n_layers"]):
            base = o["layers_start"] + l * o["layer_stride"]
            for name, shape in layout.layer_tensors(cfg):
                arr = get(f"{prefix}layers.{l}.{LAYER_MAP[name]}", shape)
                if name == "wq":
                    arr = interleave(arr, cfg["n_heads"])
                elif name == "wk":
                    arr = interleave(arr, cfg["n_kv_heads"])
                put(base + o["layer_rel"][name], arr)
        put(o["final_norm"], get(prefix + "norm.weight", (cfg["dim"],)))
        cos, sin = rope_tables(theta, layout.head_size(cfg), cfg["seq_len"], scaling)
        put(o["rope_cos"], cos)
        put(o["rope_sin"], sin)
        if not cfg["shared"]:
            put(o["wcls"], get("lm_head.weight", (cfg["vocab_size"], cfg["dim"])))
    left = sorted(n for n in idx if n not in used and not n.endswith(IGNORED))
    if left:
        log(f"WARNING: {len(left)} tensor(s) in the model were not used: {left[:4]}{' ...' if len(left) > 4 else ''}")
    log(f"wrote {dst} ({o['total']} bytes)")
    return cfg, o


def main():
    """Command line wrapper around convert()."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model", help="Hugging Face model folder (config.json + *.safetensors)")
    p.add_argument("dst", help="output .flat file")
    p.add_argument("--seq-len", type=int, default=None, help="context length to store (default: the model's max_position_embeddings)")
    p.add_argument("--board", default=None, help="board JSON file: its ext_gb is used to check the model fits the board's memory")
    p.add_argument("--kernel-limits", default=None, help="limits the kernel was built with if not the header defaults, e.g. dim=512,seq=1024")
    a = p.parse_args()
    convert(a.model, a.dst, a.seq_len, board=a.board, kernel_limits=a.kernel_limits)


if __name__ == "__main__":
    main()
