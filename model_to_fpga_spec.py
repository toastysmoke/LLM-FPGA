#!/usr/bin/env python3
"""
model_to_fpga_spec.py - split a model into "build in the FPGA" vs "store in memory".

Input : a Hugging Face model folder (*.safetensors + config.json). Only the safetensors headers are read, not the weights.
Board : --board board.json = the specs of YOUR FPGA board (on-chip memory and external memory size).
        Example: boards/u50.json. Needed for the memory plan.
Model : dense Llama-style decoder models only (sequential pre-norm blocks, no mixture-of-experts). The tool warns when a model
        does not look like that (a different number of norms per layer, layers that differ, a router).
Output (in --out):
  fpga_spec.json    hardware side: compute units, special functions, precisions, sizes
  memory_map.json   memory side: every tensor with role, shape, dtype, bytes, placement, offset
  model_params.vh   Verilog localparams for the dimensions (include in your RTL / testbench)
  weights.bin       (only with --pack) raw tensor bytes, 64-byte aligned, at the offsets in memory_map.json

SRAM / external-memory planner (--board): decides what lives in on-chip SRAM (BRAM/URAM) and what in external
memory (DDR4/HBM). It plans four kinds of data: model tensors, KV cache (one segment per layer), run-time scratch buffers
(always on chip) and tables.
Extra output: memory_map.json gets "plan" and "segments" (space = sram | ext | split, offsets in each space), memory_plan.vh;
with --pack: weights_ext.bin (external image) and weights_sram_init.bin (preload data for SRAM) instead of weights.bin.

Usage:
  python model_to_fpga_spec.py ./Llama-3.2-1B -o out
  python model_to_fpga_spec.py ./Llama-3.2-1B -o out --board boards/u50.json --max-ctx 4096
  python model_to_fpga_spec.py ./stories --board boards/u50.json --priority norms,tables,kv,layers --pack
  python model_to_fpga_spec.py ./my-model -o out --board my_board.json --pack
Only the Python standard library is needed.
"""
import argparse, json, math, re, struct
from collections import Counter, defaultdict
from pathlib import Path

# ---------------------------------------------------------------- tensor classification
# first match wins, so specific patterns come first
ROLES = [
    ("qk_norm",     r"(q_norm|k_norm)"),
    ("router",      r"(ffn_gate_inp|block_sparse_moe\.gate\.|mlp\.gate\.weight|\.router\.)"),
    ("embed",       r"(embed_tokens|token_embd|\bwte\b|word_embeddings)"),
    ("lm_head",     r"(lm_head|^output\.weight)"),
    ("attn_qkv",    r"(qkv_proj|c_attn|query_key_value|attn_qkv)"),
    ("attn_q",      r"(q_proj|attn_q\.)"),
    ("attn_k",      r"(k_proj|attn_k\.)"),
    ("attn_v",      r"(v_proj|attn_v\.)"),
    ("attn_o",      r"(o_proj|attn_output|out_proj)"),
    ("ffn_gate_up", r"gate_up_proj"),
    ("ffn_gate",    r"(gate_proj|ffn_gate)"),
    ("ffn_up",      r"(up_proj|ffn_up|fc1)"),
    ("ffn_down",    r"(down_proj|ffn_down|fc2)"),
    ("norm",        r"(norm|ln_f|ln_\d)"),
]
MATRIX_ROLES = {"attn_q", "attn_k", "attn_v", "attn_qkv", "attn_o", "ffn_gate",
                "ffn_up", "ffn_gate_up", "ffn_down", "lm_head", "router"}
QUANT_AUX = {"qzeros", "scales", "g_idx", "weight_scale", "weight_scale_inv", "input_scale"}
LAYER_RE = re.compile(r"(?:layers|blk|h)\.(\d+)\.")
ST_BYTES = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "F8_E4M3": 1, "F8_E5M2": 1,
            "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1}


def classify(name):
    """Role of a tensor from its name (embedding, attention projection, norm, ...), using the ROLES patterns."""
    low = name.lower()
    for role, pat in ROLES:
        if re.search(pat, low):
            return role
    return "other"


# ---------------------------------------------------------------- loaders
def hp_from_hf(cfg):
    """Model hyper-parameters (layers, heads, sizes, RoPE, quantization) from a Hugging Face config.json."""
    c = cfg.get("text_config", cfg)
    heads, hidden = c.get("num_attention_heads"), c.get("hidden_size")
    hd = c.get("head_dim") or (hidden // heads if hidden and heads else None)
    rp = c.get("rope_parameters") or {}
    q = c.get("quantization_config") or cfg.get("quantization_config")
    norm = None
    if "rms_norm_eps" in c:
        norm = "RMSNorm"
    elif any(k in c for k in ("layer_norm_eps", "layer_norm_epsilon", "norm_epsilon")):
        norm = "LayerNorm"
    return {
        "model_type": c.get("model_type"),
        "layers": c.get("num_hidden_layers"), "hidden": hidden, "heads": heads,
        "kv_heads": c.get("num_key_value_heads", heads), "head_dim": hd,
        "ffn": c.get("intermediate_size"), "vocab": c.get("vocab_size"),
        "ctx": c.get("max_position_embeddings"),
        "norm_kind": norm,
        "norm_eps": c.get("rms_norm_eps") or c.get("layer_norm_eps") or c.get("layer_norm_epsilon"),
        "act": c.get("hidden_act") or c.get("hidden_activation"),
        "rope_theta": c.get("rope_theta") or rp.get("rope_theta"),
        "rope_dim": int(hd * c.get("partial_rotary_factor", 1.0)) if hd else None,
        "rope_scaling": c.get("rope_scaling") or rp.get("rope_scaling"),
        "sliding_window": c.get("sliding_window"),
        "experts": c.get("num_local_experts") or c.get("num_experts"),
        "experts_used": c.get("num_experts_per_tok"),
        "tie_embeddings": c.get("tie_word_embeddings"),
        "native_dtype": c.get("torch_dtype") or c.get("dtype"),
        "quant": ({"method": q.get("quant_method"), "bits": q.get("bits"),
                   "group_size": q.get("group_size")} if q else None),
    }


def load_safetensors(p):
    """p must be a Hugging Face model folder: *.safetensors + config.json (only the safetensors headers are read)."""
    if not p.is_dir():
        raise SystemExit(f"{p}: the model must be a Hugging Face folder containing *.safetensors and config.json")
    files, d = sorted(p.glob("*.safetensors")), p
    if not files:
        raise SystemExit(f"no .safetensors files in {p}")
    cfg_path = d / "config.json"
    if not cfg_path.exists():
        raise SystemExit(f"no config.json in {p}: the layer count, sizes and activation come from it")
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    tensors = []
    for f in files:
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            hdr = json.loads(fh.read(n))
        base = 8 + n
        for name, m in hdr.items():
            if name == "__metadata__":
                continue
            b, e = m["data_offsets"]
            tensors.append({"name": name, "shape": tuple(m["shape"]), "dtype": m["dtype"],
                            "bytes": e - b, "_src": (f, base + b, e - b)})
    return {"format": "safetensors", "hp": hp_from_hf(cfg), "tensors": tensors}


# ---------------------------------------------------------------- analysis
def analyze(info, act_bits):
    """Fill in each tensor's role and layer number and derive the hardware-side numbers from the headers."""
    hp, ts = info["hp"], info["tensors"]
    for t in ts:
        t["role"] = classify(t["name"])
        m = LAYER_RE.search(t["name"])
        t["layer"] = int(m.group(1)) if m else None
        t["is_bias"] = t["name"].endswith(".bias")
        t["quant_aux"] = t["name"].rsplit(".", 1)[-1] in QUANT_AUX
        t["packed"] = t["name"].endswith("qweight")  # GPTQ/AWQ: shape is the packed shape
        n = math.prod(t["shape"]) or 1
        if t["packed"] and hp["quant"] and hp["quant"].get("bits"):
            t["bits_per_weight"] = hp["quant"]["bits"]
        else:
            t["bits_per_weight"] = round(t["bytes"] * 8 / n, 3)
        if t["is_bias"] or t["role"] in ("norm", "qk_norm", "router"):
            t["placement"] = "onchip_candidate"
        elif t["role"] == "embed":
            t["placement"] = "external_lookup"
        elif t["quant_aux"]:
            t["placement"] = "external_stream_with_weights"
        else:
            t["placement"] = "external_stream"

    # --- matmul engines: one entry per (role, shape)
    mats = defaultdict(Counter)
    dtypes = defaultdict(Counter)
    for t in ts:
        if t["role"] in MATRIX_ROLES and not t["is_bias"] and not t["quant_aux"] and len(t["shape"]) >= 2:
            mats[t["role"]][tuple(t["shape"][-2:])] += 1
            dtypes[t["role"]][t["dtype"]] += 1
    mvm = [{"role": r, "out": s[0], "in": s[1], "count": c, "weight_dtypes": dict(dtypes[r])}
           for r, shapes in mats.items() for s, c in shapes.items()]
    max_in = max((u["in"] for u in mvm), default=None)
    max_out = max((u["out"] for u in mvm), default=None)

    roles = Counter(t["role"] for t in ts if not t["is_bias"] and not t["quant_aux"])
    has_norm_bias = any(t["is_bias"] and t["role"] == "norm" for t in ts)
    gated = bool(roles["ffn_gate"] or roles["ffn_gate_up"])
    notes = []

    norm_kind = hp["norm_kind"] or ("LayerNorm" if has_norm_bias else "RMSNorm")
    if not hp["norm_kind"]:
        notes.append("norm type guessed from tensors (bias present -> LayerNorm, else RMSNorm)")
    act = hp["act"]
    if act is None:
        act = "silu" if gated else "gelu"
        notes.append("activation not stored in file - assumed " + act + "; verify against the model's reference code")
    if any(t["packed"] for t in ts):
        notes.append("GPTQ/AWQ qweight shapes are the packed shapes, not logical (out,in)")

    # --- weight precision
    mt = [t for t in ts if t["role"] in MATRIX_ROLES | {"embed"} and not t["is_bias"] and not t["quant_aux"]]
    by_dtype = Counter()
    for t in mt:
        by_dtype[t["dtype"]] += t["bytes"]
    wbits = max((t["bits_per_weight"] for t in mt), default=None)
    prec = {"weight_dtypes_by_bytes": dict(by_dtype), "max_bits_per_weight": wbits,
            "trained_dtype": hp["native_dtype"], "quantization": hp["quant"],
            "activation_precision": "not stored in the model - your choice"}
    if wbits is not None and wbits <= 8.6 and max_in:
        w = math.ceil(wbits)
        prec["int_mac_accumulator_bits"] = w + act_bits + math.ceil(math.log2(max_in))
        prec["accumulator_assumes"] = f"{w}-bit weights x {act_bits}-bit activations summed over K={max_in} (before block/group scales)"
    elif wbits is not None:
        prec["float_note"] = "float weights: multipliers + accumulator width are your choice; FP32/wide fixed-point accumulate is the safe default"

    layers, kvh, hd = hp["layers"], hp["kv_heads"], hp["head_dim"]
    ctx = hp["ctx"]
    hw = {
        "dimensions": {k: hp[k] for k in ("layers", "hidden", "heads", "kv_heads", "head_dim", "ffn", "vocab", "ctx")},
        "matmul_engines": mvm,
        "max_matmul_in": max_in, "max_matmul_out": max_out,
        "norm": {"kind": norm_kind, "dim": hp["hidden"], "eps": hp["norm_eps"], "has_bias": has_norm_bias,
                 "per_layer_count": round((roles["norm"]) / layers, 2) if layers else None},
        "qk_norm": bool(roles["qk_norm"]),
        "activation": {"fn": act, "gated": gated,
                       "ffn_pattern": "gate/up -> act(gate)*up -> down" if gated else "up -> act -> down"},
        "rope": None if hp["rope_theta"] is None else {
            "theta": hp["rope_theta"], "rotary_dim": hp["rope_dim"], "scaling": hp["rope_scaling"],
            "full_table_entries_sin_cos": (ctx * hp["rope_dim"]) if ctx and hp["rope_dim"] else None},
        "attention": {"heads": hp["heads"], "kv_heads": kvh, "head_dim": hd,
                      "gqa_group": (hp["heads"] // kvh) if hp["heads"] and kvh else None,
                      "sliding_window": hp["sliding_window"],
                      "softmax_len_max": ctx,
                      "kv_cache_bytes_per_token_16bit": (2 * layers * kvh * hd * 2) if layers and kvh and hd else None},
        "embedding_lookup": {"vocab": hp["vocab"], "dim": hp["hidden"], "tied_with_lm_head": hp["tie_embeddings"]},
        "moe": None if not hp["experts"] else {"experts": hp["experts"], "used_per_token": hp["experts_used"],
                                              "needs_router_topk": True},
        "other_units": ["residual add", "final norm", "lm_head matmul + argmax/sampling"],
        "precision": prec,
        "notes": notes,
        "unclassified_tensors": sorted({t["name"] for t in ts if t["role"] == "other"})[:20],
    }
    return hw


def build_memory(info):
    """Lay the tensors out one after another in an external-memory image, in the order the kernel uses them."""
    ts = info["tensors"]
    order = sorted(ts, key=lambda t: (-1 if t["role"] == "embed" else (t["layer"] if t["layer"] is not None else 10**9), t["name"]))
    off = 0
    for t in order:
        t["offset"] = off
        off += (t["bytes"] + 63) // 64 * 64
    totals = Counter()
    for t in ts:
        totals[t["placement"]] += t["bytes"]
    total = sum(totals.values())
    summ = {"total_bytes": total, "total_MB": round(total / 2**20, 2),
            "by_placement_MB": {k: round(v / 2**20, 3) for k, v in totals.items()},
            "packed_file_bytes": off}
    gen = ["rope sin/cos table (generated from theta, not in the model file)",
           "activation lookup/approximation table", "exp table for softmax"]
    return order, summ, gen


# ---------------------------------------------------------------- SRAM / external-memory planner
# Usable on-chip bytes per block at 32/64-bit data width: BRAM36 = 1K x 36 -> 4096 B of data; URAM288 = 4K x 72 -> 32768 B.
BRAM36_BYTES, URAM_BYTES = 4096, 32768        # the board file says how many of each the board has


def load_board(path):
    """The user's board specs. Required: ext_gb and either sram_mb or bram36 (+ uram). Optional: name, and dsp / lut / ff / clock_mhz / mem_gbps / max_ports
    (used only by kernel_plan.py)."""
    p = Path(path)
    if not p.is_file():
        raise SystemExit(f"--board: file {p} not found")
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise SystemExit(f"--board {p}: not valid JSON ({e})")
    if not isinstance(raw, dict):
        raise SystemExit(f"--board {p}: expected a JSON object such as {{\"sram_mb\": 4, \"ext_gb\": 1}}")
    known = ["name", "sram_mb", "bram36", "uram", "ext_gb", "dsp", "lut", "ff", "clock_mhz", "mem_gbps", "max_ports",
             "resource_model", "weight_cycles", "clock_by_lanes"]   # the last three: kernel_plan.py calibration from builds measured on the board
    bad = sorted(k for k in raw if k not in known and not str(k).startswith("_"))   # keys starting with _ are comments
    if bad:
        raise SystemExit(f"--board {p}: unknown field(s) {bad}; known fields: {known}")

    def num(key, required=True, zero_ok=False):
        v = raw.get(key)
        if v is None:
            if required:
                raise SystemExit(f"--board {p}: missing '{key}'")
            return None
        if isinstance(v, bool) or not isinstance(v, (int, float)) or v < 0 or (v == 0 and not zero_ok):
            raise SystemExit(f"--board {p}: '{key}' must be a number{'' if zero_ok else ' greater than 0'}, got {v!r}")
        return v

    if "sram_mb" in raw and ("bram36" in raw or "uram" in raw):
        raise SystemExit(f"--board {p}: give either 'sram_mb' or 'bram36'/'uram', not both")
    if "sram_mb" in raw:
        sram = num("sram_mb") * 2**20
    elif "bram36" in raw:
        sram = num("bram36", zero_ok=True) * BRAM36_BYTES + (num("uram", required=False, zero_ok=True) or 0) * URAM_BYTES
        if sram <= 0:
            raise SystemExit(f"--board {p}: 'bram36'/'uram' give no on-chip memory")
    else:
        raise SystemExit(f"--board {p}: give the on-chip memory as 'sram_mb' or as 'bram36' (+ 'uram')")
    def pairs_ok(v):
        return isinstance(v, list) and len(v) >= 1 and all(isinstance(x, list) and len(x) == 2 and all(isinstance(y, (int, float)) and not isinstance(y, bool) and y > 0 for y in x) for x in v)

    if "weight_cycles" in raw and not (isinstance(raw["weight_cycles"], list) and len(raw["weight_cycles"]) == 2
                                       and all(isinstance(y, (int, float)) and not isinstance(y, bool) and y >= 0 for y in raw["weight_cycles"]) and raw["weight_cycles"][0] > 0):
        raise SystemExit(f"--board {p}: 'weight_cycles' must be [a, b] (clock cycles per weight word = a + b / lanes per port), a > 0, b >= 0")
    if "clock_by_lanes" in raw and not pairs_ok(raw["clock_by_lanes"]):
        raise SystemExit(f"--board {p}: 'clock_by_lanes' must be a list of [matrix lanes in all, kernel MHz] pairs, all greater than 0")
    if "resource_model" in raw:
        rm = raw["resource_model"]
        if not isinstance(rm, dict) or any(k not in ("DSP", "LUT", "FF", "margin", "bram_factor") for k in rm) or any(
                k in rm and not (isinstance(rm[k], list) and len(rm[k]) == 3 and all(isinstance(y, (int, float)) and not isinstance(y, bool) for y in rm[k])) for k in ("DSP", "LUT", "FF")):
            raise SystemExit(f"--board {p}: 'resource_model' may hold DSP / LUT / FF as [base, per matrix lane, per attention lane] and 'margin', 'bram_factor' as numbers")
    return {"name": str(raw.get("name") or p.stem), "file": str(p), "sram_bytes": int(sram),
            "resource_model": raw.get("resource_model"), "weight_cycles": raw.get("weight_cycles"), "clock_by_lanes": raw.get("clock_by_lanes"),
            "ext_bytes": int(num("ext_gb") * 2**30), "dsp": num("dsp", required=False), "clock_mhz": num("clock_mhz", required=False),
            "mem_gbps": num("mem_gbps", required=False), "max_ports": num("max_ports", required=False), "lut": num("lut", required=False), "ff": num("ff", required=False),
            "bram36": raw.get("bram36")}


CLASS_ORDER = ["scratch", "tables", "norms", "kv", "lm_head", "embed", "layers", "state"]
ALIGN = 64


def align_up(n, a=ALIGN):
    """Round n up to a multiple of a (default: the memory alignment)."""
    return (int(n) + a - 1) // a * a


def exec_rank(t):
    """Order inside a layer = order the kernel uses the tensors (not alphabetical)."""
    n, r = t["name"].lower(), t["role"]
    if r == "norm":
        return 0 if re.search(r"(input_layernorm|attn_norm|ln_1|input_norm|pre_attn|self_attn_layer_norm)", n) else 5
    base = {"qk_norm": 1.5, "attn_qkv": 1, "attn_q": 1, "attn_k": 2, "attn_v": 3, "attn_o": 4, "router": 5.5,
            "ffn_gate_up": 6, "ffn_gate": 6, "ffn_up": 7, "ffn_down": 8}.get(r, 9)
    return base + (0.1 if t["is_bias"] else 0)


def build_segments(info, hw, a):
    """Everything that needs memory at run time, as segments with size and per-token traffic."""
    hp, d = info["hp"], hw["dimensions"]
    missing = [k for k in ("layers", "hidden", "heads", "kv_heads", "head_dim", "ffn", "vocab", "ctx") if not d.get(k)]
    if missing:
        raise SystemExit(f"planner needs these hyperparameters (config.json): {missing}")
    L, H, nh, kvh, hd, F, V = (d[k] for k in ("layers", "hidden", "heads", "kv_heads", "head_dim", "ffn", "vocab"))
    max_ctx = a.max_ctx or min(d["ctx"], 2048)
    ab, kb = a.act_bytes, a.kv_bytes
    rot = (hw["rope"] or {}).get("rotary_dim") or 0
    tied = bool(hp["tie_embeddings"]) or not any(t["role"] == "lm_head" for t in info["tensors"])
    moe_frac = (hp["experts_used"] / hp["experts"]) if hp["experts"] and hp["experts_used"] else 1.0
    segs = []

    def seg(**k):
        k.setdefault("rows", None); k.setdefault("row_bytes", None); k.setdefault("layer", None)
        k.setdefault("reads", 0.0); k.setdefault("writes", 0.0); k.setdefault("src", None)
        k["bytes"] = int(k["bytes"]); segs.append(k)

    # --- model tensors (read from the model file)
    for t in info["tensors"]:
        r, nb = t["role"], t["bytes"]
        if t["is_bias"] or r in ("norm", "qk_norm", "router"):
            cls, rows = "norms", None
        elif r == "embed":
            cls = "embed"
        elif r == "lm_head":
            cls = "lm_head"
        else:
            cls = "layers"
        rows = None
        if len(t["shape"]) == 2 and not t["is_bias"] and not t["packed"] and not t["quant_aux"] and t["shape"][0] and nb % t["shape"][0] == 0:
            rows = t["shape"][0]
        frac = moe_frac if re.search(r"(experts?\.|_exps)", t["name"].lower()) else 1.0
        reads = nb * frac
        if r == "embed":                       # lookup reads one row; a tied table is also the output head -> whole table
            reads = nb if tied else nb / V
        seg(name=t["name"], kind="weight", cls=cls, bytes=nb, rows=rows, row_bytes=(nb // rows if rows else None),
            layer=t["layer"], reads=reads, src=t,
            key=((-1, 0) if r == "embed" else ((10**6 + 1, 0) if r == "lm_head" else (t["layer"] if t["layer"] is not None else 10**6, exec_rank(t)))))
    # --- run-time buffers: always on chip (scratch)
    rowbuf = hw["max_matmul_in"] or max(H, F)
    scratch = [("x", H), ("xb", H), ("xb2", H), ("q", nh * hd), ("k_cur", kvh * hd), ("v_cur", kvh * hd), ("attn_scores", max_ctx),
               ("ffn_gate_out", F), ("ffn_up_out", F), ("weight_row_buf", rowbuf), ("kv_slice_buf", hd), ("norm_weights", H),
               ("rope_cs_cur", rot)]
    for n, cnt in scratch:
        seg(name="scratch." + n, kind="scratch", cls="scratch", bytes=cnt * ab, key=(3 * 10**6, 0))
    # --- tables
    if rot:
        seg(name="table.rope_sin_cos", kind="table", cls="tables", bytes=max_ctx * rot * ab, rows=max_ctx, row_bytes=rot * ab,
            reads=rot * ab, key=(2 * 10**6, 0))
    seg(name="table.exp_and_activation_lut", kind="table", cls="tables", bytes=2 * 1024 * ab, key=(2 * 10**6, 1))
    # --- KV cache, one segment per layer; rows = tokens (first tokens go to SRAM when split)
    rb = 2 * kvh * hd * kb
    for l in range(L):
        seg(name=f"kv_cache.layer{l}", kind="kv", cls="kv", bytes=rb * max_ctx, rows=max_ctx, row_bytes=rb, layer=l,
            reads=rb * max_ctx, writes=rb, key=(2 * 10**6 + 1, l))
    # --- state the host reads / writes
    seg(name="state.logits", kind="state", cls="state", bytes=V * ab, writes=V * ab, key=(2 * 10**6 + 2, 0))
    seg(name="state.tokens", kind="state", cls="state", bytes=(max_ctx + 1) * 4, key=(2 * 10**6 + 2, 1))
    plan_in = {"max_ctx": max_ctx, "act_bytes": ab, "kv_bytes": kb, "tied_embedding_head": tied,
               "moe_read_fraction": moe_frac}
    return segs, plan_in


def place(segs, budget_bytes, priority, row_split):
    """Greedy placement. Returns (sram_used, notes). Sets seg['sram_bytes'], ['sram_rows'] on every segment."""
    for s in segs:
        s["sram_bytes"], s["sram_rows"] = 0, 0
    notes, remaining = [], budget_bytes
    scratch = [s for s in segs if s["cls"] == "scratch"]
    need = sum(align_up(s["bytes"]) for s in scratch)
    for s in scratch:
        s["sram_bytes"] = s["bytes"]
    remaining -= need
    if remaining < 0:
        notes.append(f"scratch buffers alone ({need} B) exceed the usable SRAM budget ({budget_bytes} B): the kernel will not fit")
        remaining = 0

    def density(s):                                  # external traffic removed per SRAM byte
        return (s["reads"] + s["writes"]) / s["bytes"] if s["bytes"] else 0.0

    rest = [s for s in segs if s["cls"] != "scratch"]
    if priority == "auto":
        rest.sort(key=lambda s: (-round(density(s), 2), CLASS_ORDER.index(s["cls"]), s["key"]))   # near-ties -> class order
    else:
        rank = {c: i for i, c in enumerate(priority)}
        rest.sort(key=lambda s: (rank.get(s["cls"], len(rank)), s["key"]))
    for s in rest:
        size = align_up(s["bytes"])
        if s["bytes"] == 0:
            continue
        if size <= remaining:
            s["sram_bytes"], s["sram_rows"] = s["bytes"], (s["rows"] or 0)
            remaining -= size
        elif row_split and s["rows"] and s["row_bytes"] and s["row_bytes"] <= remaining:
            n = min(s["rows"] - 1, remaining // s["row_bytes"])
            while n > 0 and align_up(n * s["row_bytes"]) > remaining:
                n -= 1
            # a split must be worth it: at least 4 KiB and 2% of the segment, otherwise leave the space for something else
            if n > 0 and n * s["row_bytes"] >= max(4096, 0.02 * s["bytes"]):
                s["sram_bytes"], s["sram_rows"] = n * s["row_bytes"], n
                remaining -= align_up(n * s["row_bytes"])
    return budget_bytes - remaining, notes


def finish_plan(segs, ddr_cap):
    """Assign offsets in the two address spaces (on-chip and external). Mutates segs."""
    # SRAM offsets in the order of placement priority are not needed: pack by class then key for a stable map
    off = 0
    for s in sorted([s for s in segs if s["sram_bytes"]], key=lambda s: (CLASS_ORDER.index(s["cls"]), s["key"])):
        s["sram_offset"] = off
        off += align_up(s["sram_bytes"])
    sram_used = off
    # external space: weights in execution order, then tables, KV, state
    off = 0
    for s in sorted(segs, key=lambda s: s["key"]):
        s["ddr_bytes"] = s["bytes"] - s["sram_bytes"]
        if s["ddr_bytes"] > 0 and s["cls"] != "scratch":
            s["ddr_offset"], s["ddr_row_start"] = off, s["sram_rows"]
            off += align_up(s["ddr_bytes"])
    ddr_used = off
    return {"sram_used_bytes": sram_used, "ext_used_bytes": ddr_used, "fits_external": ddr_used <= ddr_cap}


def make_plan(info, hw, a):
    """Decide which tensors live in on-chip memory and which in external memory for the given board, by priority, and return the plan and its segments."""
    board = a.board_spec
    sram_total, ddr_cap = board["sram_bytes"], board["ext_bytes"]
    segs, plan_in = build_segments(info, hw, a)
    prio = "auto" if a.priority == "auto" else [c.strip() for c in a.priority.split(",")]
    if prio != "auto":
        bad = [c for c in prio if c not in CLASS_ORDER]
        if bad:
            raise SystemExit(f"unknown class in --priority: {bad}; known: {CLASS_ORDER}")
    budget = int(sram_total * a.sram_util)
    _, notes = place(segs, budget, prio, not a.no_row_split)
    res = finish_plan(segs, ddr_cap)
    by_cls = defaultdict(lambda: {"total": 0, "sram": 0, "ext": 0})
    for s in segs:
        c = by_cls[s["cls"]]
        c["total"] += s["bytes"]; c["sram"] += s["sram_bytes"]; c["ext"] += s["bytes"] - s["sram_bytes"] if s["cls"] != "scratch" else 0
    plan = {"board": board["name"],
            "sram_total_bytes": int(sram_total), "sram_util": a.sram_util, "sram_usable_bytes": budget,
            "external_capacity_bytes": int(ddr_cap),
            "priority": prio, "row_split": not a.no_row_split,
            **plan_in, **res,
            "by_class_bytes": {c: v for c, v in sorted(by_cls.items(), key=lambda kv: CLASS_ORDER.index(kv[0]))}, "notes": notes}
    if not res["fits_external"]:
        plan["notes"].append("external data does not fit the external memory capacity")
    return segs, plan


# ---------------------------------------------------------------- output
def read_bytes(t):
    """Read one tensor's raw bytes from its source file."""
    f, start, n = t["_src"]
    with open(f, "rb") as fh:
        fh.seek(start)
        return fh.read(n)


def write_vh(path, hw):
    """Write the model dimensions as Verilog localparams (model_params.vh)."""
    d = hw["dimensions"]
    lines = ["// generated by model_to_fpga_spec.py", "`ifndef MODEL_PARAMS_VH", "`define MODEL_PARAMS_VH"]
    names = {"layers": "N_LAYERS", "hidden": "HIDDEN", "heads": "N_HEADS", "kv_heads": "N_KV_HEADS",
             "head_dim": "HEAD_DIM", "ffn": "FFN_DIM", "vocab": "VOCAB", "ctx": "MAX_CTX"}
    for k, n in names.items():
        if isinstance(d.get(k), int):
            lines.append(f"localparam {n} = {d[k]};")
    for n, v in (("MAX_IN", hw["max_matmul_in"]), ("MAX_OUT", hw["max_matmul_out"]),
                 ("ACC_BITS", hw["precision"].get("int_mac_accumulator_bits"))):
        if isinstance(v, int):
            lines.append(f"localparam {n} = {v};")
    lines.append("`endif")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def space_of(s):
    """Where a segment lives: 'sram', 'ext' or 'split' (partly on chip, partly external)."""
    if s["sram_bytes"] and s["sram_bytes"] < s["bytes"]:
        return "split"
    return "sram" if s["sram_bytes"] else "ext"


def clean_segments(segs):
    """Keep only the documented fields of each segment, in a stable order, for the JSON output."""
    keys = ("name", "kind", "cls", "layer", "bytes", "rows", "row_bytes", "sram_offset", "sram_bytes", "sram_rows",
            "ddr_offset", "ddr_bytes", "ddr_row_start")
    out = []
    for s in sorted(segs, key=lambda s: (CLASS_ORDER.index(s["cls"]), s["key"])):
        d = {k: s[k] for k in keys if k in s}
        d["space"] = space_of(s)
        if s["cls"] == "scratch":
            d.pop("ddr_bytes", None)
        out.append(d)
    return out


def write_plan_vh(path, plan, segs):
    """Write the memory plan (base addresses and sizes per class of data) as Verilog localparams (memory_plan.vh)."""
    def base(cls):
        offs = [s["ddr_offset"] for s in segs if s["cls"] == cls and "ddr_offset" in s]
        return min(offs) if offs else 0
    w = [s for s in segs if s["kind"] == "weight" and "ddr_offset" in s]
    wbytes = max((s["ddr_offset"] + s["ddr_bytes"] for s in w), default=0)
    lines = ["// generated by model_to_fpga_spec.py (memory plan: SRAM = on-chip BRAM/URAM, EXT = DDR4/HBM)",
             "`ifndef MEMORY_PLAN_VH", "`define MEMORY_PLAN_VH",
             f"localparam SRAM_USABLE_BYTES = {plan['sram_usable_bytes']};", f"localparam SRAM_USED_BYTES = {plan['sram_used_bytes']};",
             f"localparam EXT_USED_BYTES = {plan['ext_used_bytes']};", f"localparam PLAN_MAX_CTX = {plan['max_ctx']};",
             f"localparam EXT_BASE_WEIGHTS = 0;", f"localparam EXT_WEIGHTS_BYTES = {wbytes};",
             f"localparam EXT_BASE_TABLES = {base('tables')};", f"localparam EXT_BASE_KV = {base('kv')};",
             f"localparam EXT_BASE_STATE = {base('state')};"]
    for s in sorted([s for s in segs if s["kind"] == "kv"], key=lambda s: s["layer"]):
        lines.append(f"localparam KV_SRAM_TOKENS_L{s['layer']} = {s['sram_rows']};")
    lines.append("`endif")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def pack_plan(out, segs):
    """weights_ext.bin = image for external memory; weights_sram_init.bin = data to preload into SRAM (zeros where buffers live)."""
    ext_end = max((s["ddr_offset"] + s["ddr_bytes"] for s in segs if s["kind"] == "weight" and "ddr_offset" in s), default=0)
    sram_end = max((s["sram_offset"] + s["sram_bytes"] for s in segs if "sram_offset" in s), default=0)
    with open(out / "weights_ext.bin", "wb") as fe, open(out / "weights_sram_init.bin", "wb") as fs:
        for s in segs:
            if s["kind"] != "weight":
                continue
            data = read_bytes(s["src"])
            cut = s["sram_rows"] * s["row_bytes"] if s["rows"] and 0 < s["sram_bytes"] < s["bytes"] else s["sram_bytes"]
            if s["sram_bytes"]:
                fs.seek(s["sram_offset"]); fs.write(data[:cut])
            if s["bytes"] - cut > 0:
                fe.seek(s["ddr_offset"]); fe.write(data[cut:])
        fe.truncate(align_up(ext_end)); fs.truncate(align_up(sram_end))


def print_plan(plan, segs):
    """Print the memory plan as a human-readable table."""
    mb = lambda b: f"{b / 2**20:.3f} MB"
    print("\nMEMORY PLAN")
    print(f"  board {plan['board']} | SRAM {mb(plan['sram_total_bytes'])} x {plan['sram_util']:.0%} usable = {mb(plan['sram_usable_bytes'])}"
          f" | external {mb(plan['external_capacity_bytes'])}"
          f" | context {plan['max_ctx']} | priority {plan['priority']}")
    print(f"  {'class':<9}{'total':>12}{'on-chip':>12}{'external':>12}")
    for c, v in plan["by_class_bytes"].items():
        print(f"  {c:<9}{mb(v['total']):>12}{mb(v['sram']):>12}{mb(v['ext']):>12}")
    print(f"  SRAM used {mb(plan['sram_used_bytes'])} | external used {mb(plan['ext_used_bytes'])} (fits: {plan['fits_external']})")
    split = [s for s in segs if space_of(s) == "split"]
    for s in split:
        print(f"  split: {s['name']}: first {s['sram_rows']} of {s['rows']} rows on-chip, rest external")
    for n in plan["notes"]:
        print("  PLAN NOTE:", n)


# ---------------------------------------------------------------- the supported model family
# What ONE transformer block does, and in what order, is not stored in a model file. The tool supports one structure: the
# sequential pre-norm decoder block of the Llama family (dense, no mixture-of-experts). Which parts of it a model has (fused
# projections, gated or ungated feed-forward, query/key norm, rotary encoding) is read from its tensors and config.json.
def unsupported_warnings(info, hw):
    """Signs that the model is not a dense Llama-style decoder, so the counts below may be wrong for it."""
    w, stems, per = [], defaultdict(set), defaultdict(Counter)
    for t in info["tensors"]:
        if t["layer"] is None or t["quant_aux"]:
            continue
        if t["role"] == "norm":
            stems[t["layer"]].add(re.sub(r"\.(weight|bias)$", "", t["name"]))
        elif not t["is_bias"]:
            per[t["layer"]][(t["role"], tuple(t["shape"]))] += 1
    odd = sorted(l for l, v in stems.items() if len(v) != 2)
    if odd:
        w.append(f"layer(s) {odd[:5]} have {len(stems[odd[0]])} norm(s) instead of 2 (before attention, before the feed-forward block)")
    if per:
        first = min(per)
        diff = sorted(l for l in per if per[l] != per[first])
        if diff:
            w.append(f"layer(s) {diff[:5]} have different weights from layer {first}; every layer is assumed to be the same")
    if hw["moe"]:
        w.append("mixture-of-experts model: the router and expert selection are not modelled")
    return w


def main():
    """Command line: analyse the model, build the memory plan and write fpga_spec.json, the .vh files and memory_map.json."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("-o", "--out", default="model_architecture")
    ap.add_argument("--pack", action="store_true", help="also write the weight image(s) (weights.bin, or weights_ext.bin + weights_sram_init.bin with the planner)")
    ap.add_argument("--act-bits", type=int, default=8, help="activation bits for accumulator sizing (default 8)")
    ap.add_argument("--board", default=None, help="JSON file with the specs of your board (on-chip memory and external memory size); "
                    "turns on the memory plan. Example: boards/u50.json")
    g = ap.add_argument_group("SRAM / external-memory planner (runs when --board is given)")
    g.add_argument("--sram-util", type=float, default=0.85, help="fraction of SRAM the plan may use (the rest: kernel logic, routing; default 0.85)")
    g.add_argument("--max-ctx", type=int, default=None, help="context length to plan for (KV cache + attention buffers; default min(model ctx, 2048))")
    g.add_argument("--act-bytes", type=int, default=4, help="bytes per activation / scratch / table element (default 4 = fp32, as the current kernel)")
    g.add_argument("--kv-bytes", type=int, default=4, help="bytes per KV-cache element (default 4)")
    g.add_argument("--priority", default="auto",
                   help="'auto' (largest external traffic removed per SRAM byte first) or a comma list of classes, first = most wanted on-chip: "
                        + ",".join(CLASS_ORDER))
    g.add_argument("--no-row-split", action="store_true", help="never split one tensor (or KV cache) between SRAM and external memory")
    a = ap.parse_args()
    a.board_spec = load_board(a.board) if a.board else None
    want_plan = a.board_spec is not None

    p = Path(a.model)
    info = load_safetensors(p)
    hw = analyze(info, a.act_bits)
    layout_notes = unsupported_warnings(info, hw)
    order, mem, gen = build_memory(info)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    segs = plan = None
    if want_plan:
        segs, plan = make_plan(info, hw, a)

    (out / "fpga_spec.json").write_text(json.dumps({"hardware": hw, "generated_tables": gen}, indent=2, default=str), encoding="utf-8")
    clean = [{k: (list(v) if k == "shape" else v) for k, v in t.items() if not k.startswith("_")} for t in order]
    mm = {"summary": mem, "tensors": clean}
    if plan:
        mm["plan"], mm["segments"] = plan, clean_segments(segs)
    (out / "memory_map.json").write_text(json.dumps(mm, indent=2, default=str), encoding="utf-8")
    write_vh(out / "model_params.vh", hw)
    if plan:
        write_plan_vh(out / "memory_plan.vh", plan, segs)

    if a.pack and plan:
        pack_plan(out, segs)
    elif a.pack:
        with open(out / "weights.bin", "wb") as fh:
            for t in order:
                fh.seek(t["offset"])
                fh.write(read_bytes(t))
            fh.truncate(mem["packed_file_bytes"])

    # --- console summary
    d = hw["dimensions"]
    print(f"{info['format']} | {info['hp']['model_type']} | {d['layers']} layers, hidden {d['hidden']}, "
          f"{d['heads']}h/{d['kv_heads']}kv x {d['head_dim']}, ffn {d['ffn']}, vocab {d['vocab']}")
    print("\nBUILD IN FPGA")
    for u in sorted(hw["matmul_engines"], key=lambda u: u["role"]):
        print(f"  matmul {u['role']:<12} {u['out']} x {u['in']}  x{u['count']}  {u['weight_dtypes']}")
    print(f"  norm: {hw['norm']['kind']} (dim {hw['norm']['dim']}, bias={hw['norm']['has_bias']})  qk_norm={hw['qk_norm']}")
    print(f"  activation: {hw['activation']['fn']} ({'gated' if hw['activation']['gated'] else 'ungated'})")
    print(f"  rope: {hw['rope']}")
    print(f"  attention: {hw['attention']}")
    if hw["moe"]:
        print(f"  moe: {hw['moe']}")
    print(f"  precision: {hw['precision']}")
    print("\nLAYER STRUCTURE")
    print("  dense Llama-style decoder layer (sequential pre-norm block): the only layer structure this tool supports")
    for n in layout_notes:
        print("  WARNING: " + n + "; this tool supports dense Llama-style models only, so the result may be wrong for this model")
    print("\nSTORE IN MEMORY")
    print(f"  {mem['total_MB']} MB total | {mem['by_placement_MB']}")
    print("  generated (not in file): " + "; ".join(gen))
    for n in hw["notes"]:
        print("  note:", n)
    if hw["unclassified_tensors"]:
        print("  UNCLASSIFIED tensors (check these):", hw["unclassified_tensors"])
    if plan:
        print_plan(plan, segs)
    files = ["fpga_spec.json", "memory_map.json", "model_params.vh"] + (["memory_plan.vh"] if plan else [])
    if a.pack:
        files += ["weights_ext.bin", "weights_sram_init.bin"] if plan else ["weights.bin"]
    print(f"\nwrote {out}/" + ", ".join(files))


if __name__ == "__main__":
    main()
