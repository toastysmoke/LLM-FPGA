"""Tests for hf_to_flat.py (no downloads, no board).   usage: python tests/test_hf_to_flat.py

 1. round trip: rebuild a Hugging Face folder from our llama2.c stories checkpoints, convert it, and compare with the .flat files
    that already ran on the board (byte for byte; the rotary tables are compared separately because they are recomputed)
 2. independent check: a random Hugging Face model is run through a numpy implementation of the Hugging Face convention
    (rotate_half) and through our reference on the converted file; the logits must agree. A control run without the row
    reordering must FAIL this check, so the check can really detect the mistake it exists for.
 3. file layouts: shards, names without the 'model.' prefix, tied and separate output head, F32 / F16 / BF16, --seq-len
 4. refusals and warnings"""
import json
import struct
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "ref"))
sys.path.insert(0, str(ROOT / "tests"))
import hf_to_flat  # noqa: E402
import layout  # noqa: E402
import reference  # noqa: E402
from hf_helpers import (quant, write_safetensors, write_hf, to_meta_to_hf, hf_tensors_from_llama2c,  # noqa: E402
                        random_model, hf_logits, our_logits)

results = []
TMP = Path(tempfile.mkdtemp(prefix="hf2flat_"))


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(("PASS  " if ok else "FAIL  ") + name + ("" if ok else f"\n      {detail}"))


# ---------------------------------------------------------------- helpers
def convert(folder, name="out.flat", **kw):
    msgs = []
    dst = TMP / name
    hf_to_flat.convert(folder, dst, log=msgs.append, **kw)
    return dst, msgs


def refused(folder, expect, **kw):
    try:
        hf_to_flat.convert(folder, TMP / "refused.flat", log=lambda m: None, **kw)
    except SystemExit as e:
        return expect in str(e)
    return False


# ---------------------------------------------------------------- 1. round trip against the files that ran on the board
for model in ("stories260K", "stories15M"):
    src, flat = ROOT / "models" / f"{model}.bin", ROOT / "models" / f"{model}.flat"
    if not (src.exists() and flat.exists()):
        print(f"SKIP  round trip {model} (files not found)")
        continue
    cfg, w = reference.load_llama2c(str(src))
    folder = write_hf(TMP / f"hf_{model}", cfg, hf_tensors_from_llama2c(cfg, w))
    out, _ = convert(folder, f"{model}.flat")
    a, b = np.fromfile(out, np.uint8), np.fromfile(flat, np.uint8)
    o = layout.compute(cfg)
    rope_lo, rope_hi = o["rope_cos"], o["total"] if cfg["shared"] else o["wcls"]
    mask = np.ones(len(a), bool)
    mask[rope_lo:rope_hi] = False
    same_size = len(a) == len(b)
    check(f"round trip {model}: same size, and everything except the rotary tables is byte-identical",
          same_size and np.array_equal(a[mask], b[mask]), f"sizes {len(a)} vs {len(b)}; differing bytes outside the rope tables: {int((a[mask] != b[mask]).sum()) if same_size else '?'}")
    ca, cb = np.fromfile(out, "<f4"), np.fromfile(flat, "<f4")
    r = slice(rope_lo // 4, rope_hi // 4)
    err = float(np.abs(ca[r] - cb[r]).max()) if same_size else 1e9
    exact = same_size and np.array_equal(a, b)
    check(f"round trip {model}: rotary tables {'are byte-identical' if exact else f'match to {err:.1e} (max abs difference)'}", err < 1e-5,
          f"max abs difference {err}")

# ---------------------------------------------------------------- 2. independent Hugging Face convention check
tokens = [3, 17, 5, 40, 9, 22]
for dtype in ("F32", "F16", "BF16"):
    cfg, t = random_model(1)
    folder = write_hf(TMP / f"rand_{dtype}", cfg, t, dtype)
    out, _ = convert(folder, f"rand_{dtype}.flat")
    diff = float(np.abs(our_logits(out, tokens) - hf_logits(cfg, t, tokens, dtype=dtype)).max())
    check(f"logits match the Hugging Face convention ({dtype} weights, separate head, grouped-query attention): max diff {diff:.1e}", diff < 2e-4, f"max diff {diff}")
cfg, t = random_model(2, shared=True)
out, _ = convert(write_hf(TMP / "rand_tied", cfg, t), "rand_tied.flat")
diff = float(np.abs(our_logits(out, tokens) - hf_logits(cfg, t, tokens)).max())
check(f"logits match with the output head tied to the embedding: max diff {diff:.1e}", diff < 2e-4, f"max diff {diff}")
check("tied model writes the shared flag", reference.load_flat(out)[0]["shared"] is True)

# control: without the row reordering the same check must fail
cfg, t = random_model(1)
folder = write_hf(TMP / "rand_control", cfg, t)
saved = hf_to_flat.interleave
hf_to_flat.interleave = lambda w, n: w
try:
    out, _ = convert(folder, "control.flat")
finally:
    hf_to_flat.interleave = saved
diff = float(np.abs(our_logits(out, tokens) - hf_logits(cfg, t, tokens)).max())
check(f"control: skipping the row reordering is detected (max diff {diff:.2f})", diff > 1e-2)

# ---------------------------------------------------------------- 3. file layouts
cfg, t = random_model(3)
single, _ = convert(write_hf(TMP / "one", cfg, t), "one.flat")
sharded, _ = convert(write_hf(TMP / "three", cfg, t, shards=3), "three.flat")
check("a model split over 3 safetensors files gives the same output", single.read_bytes() == sharded.read_bytes())
cfg2, t2 = random_model(3, prefix="")
nopre, _ = convert(write_hf(TMP / "nopre", cfg2, t2), "nopre.flat")
check("names without the 'model.' prefix give the same output", single.read_bytes() == nopre.read_bytes())
short, _ = convert(write_hf(TMP / "seq", cfg, t), "short.flat", seq_len=8)
c_short, _ = layout.unpack_header(short.read_bytes())
check("--seq-len 8 stores a shorter context and rotary table", c_short["seq_len"] == 8 and short.stat().st_size < single.stat().st_size)
check("shorter context keeps the first rotary rows identical", np.array_equal(reference.load_flat(short)[1]["rope_cos"], reference.load_flat(single)[1]["rope_cos"][:8]))
cfg_t, t_t = random_model(4, shared=True)
t_t_head = dict(t_t)
t_t_head["lm_head.weight"] = t_t["model.embed_tokens.weight"]
out, msgs = convert(write_hf(TMP / "tiedhead", cfg_t, t_t_head), "tiedhead.flat")
check("a tied model that also stores lm_head ignores the copy (shared flag, no warning)", reference.load_flat(out)[0]["shared"] and not any("not used" in m for m in msgs), str(msgs))

# ---------------------------------------------------------------- 4. refusals and warnings
cfg, t = random_model(5)
def hf(name, tensors=None, **kw):
    return write_hf(TMP / name, cfg, tensors if tensors is not None else t, **kw)

check("refuses a bias tensor", refused(hf("bias", {**t, "model.layers.0.self_attn.q_proj.bias": np.zeros(32, np.float32)}), "cannot use"))
check("refuses query/key norm tensors", refused(hf("qkn", {**t, "model.layers.0.self_attn.q_norm.weight": np.ones(8, np.float32)}), "cannot use"))
check("refuses expert tensors", refused(hf("moe", {**t, "model.layers.0.mlp.experts.0.w.weight": np.ones(4, np.float32)}), "cannot use"))
check("refuses rope scaling", refused(hf("rs", extra_cfg={"rope_scaling": {"rope_type": "llama3", "factor": 8.0}}), "rope scaling"))
check("accepts rope_scaling of type default", not refused(hf("rsd", extra_cfg={"rope_scaling": {"rope_type": "default"}}), "rope scaling"))
check("refuses a GELU model", refused(hf("gelu", extra_cfg={"hidden_act": "gelu"}), "SiLU"))
check("refuses a different head_dim", refused(hf("hd", extra_cfg={"head_dim": 16}), "head_dim"))
check("refuses partial rotary", refused(hf("pr", extra_cfg={"partial_rotary_factor": 0.5}), "partial rotary"))
check("refuses attention_bias", refused(hf("ab", extra_cfg={"attention_bias": True}), "biases"))
check("refuses a mixture-of-experts config", refused(hf("moec", extra_cfg={"num_local_experts": 8}), "mixture-of-experts"))
check("refuses quantized weights", refused(hf("qc", extra_cfg={"quantization_config": {"quant_method": "gptq", "bits": 4}}), "quantized"))
check("refuses --seq-len longer than the model's context", refused(hf("sl"), "longer than", seq_len=64))
check("refuses sliding window shorter than the context", refused(hf("sw", extra_cfg={"sliding_window": 4}), "sliding-window"))
check("refuses a missing tensor", refused(hf("miss", {k: v for k, v in t.items() if "layers.1.mlp.up_proj" not in k}), "not found"))
check("refuses a wrong shape", refused(hf("shape", {**t, "model.layers.0.self_attn.v_proj.weight": np.zeros((5, 32), np.float32)}), "expected"))
check("refuses an unsupported data type (I8)", refused(hf("i8", override={"model.layers.0.self_attn.o_proj.weight": "I8"}), "not supported"))
(TMP / "nocfg").mkdir()
check("refuses a folder without config.json", refused(TMP / "nocfg", "no config.json") or refused(TMP / "nocfg", "no .safetensors"))
check("refuses a path that is not a folder", refused(TMP / "one" / "model.safetensors", "must be a Hugging Face model folder"))
_, msgs = convert(hf("eps", extra_cfg={"rms_norm_eps": 1e-6}), "eps.flat")
check("warns when rms_norm_eps is not 1e-5", any("rms_norm_eps" in m for m in msgs), str(msgs))
_, msgs = convert(hf("extra", {**t, "stray.weight": np.zeros(3, np.float32)}), "extra.flat")
check("warns about tensors that were not used", any("not used" in m for m in msgs), str(msgs))
_, msgs = convert(hf("invf", {**t, "model.layers.0.self_attn.rotary_emb.inv_freq": np.zeros(4, np.float32)}), "invf.flat")
check("ignores the rotary_emb.inv_freq buffer silently", not any("not used" in m for m in msgs), str(msgs))

print(f"\n{sum(results)} of {len(results)} checks passed")
import shutil  # noqa: E402
shutil.rmtree(TMP, ignore_errors=True)
sys.exit(0 if all(results) else 1)
