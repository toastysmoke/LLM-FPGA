"""Tests for fit_check.py and its use in hf_to_flat.py (no board needed).   usage: python tests/test_fit_check.py
The two models that really ran on the board (stories260K, stories15M) are the ground truth: they must pass, with the memory
addresses run_kernel.tcl printed for them."""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "tests"), str(ROOT / "ref")]
import fit_check  # noqa: E402
import hf_to_flat  # noqa: E402
import layout  # noqa: E402
from hf_helpers import random_model, write_hf  # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="fit_"))
BOARD = ROOT / "tests" / "fixtures" / "small_board.json"
results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(("PASS  " if ok else "FAIL  ") + name + ("" if ok else f"\n      {detail}"))


def cfg_of(dim, hidden, layers, heads, kv, vocab, seq, shared=True):
    c = dict(dim=dim, hidden_dim=hidden, n_layers=layers, n_heads=heads, n_kv_heads=kv, vocab_size=vocab, seq_len=seq, shared=shared)
    return c, layout.compute(c)


LIM = fit_check.kernel_limits()
GIB = 2 ** 30

# ---- the limits come from the kernel header
check("limits and lanes are read from sw/llm_kernel.h", LIM == {"LLM_MAX_DIM": 288, "LLM_MAX_HIDDEN": 768, "LLM_MAX_SEQ": 512, "LLM_MAX_HEADS": 16,
                                                      "LLM_MAX_HEAD_SIZE": 64, "LLM_LANES": 1, "LLM_LANES_ATT": 1}, str(LIM))
# the lanes of a build: read from its HLS config; a model whose sizes are not multiples is refused (the kernel would return -8)
cfgfile = TMP / "build.cfg"
cfgfile.write_text(chr(10).join(["part=x", "[hls]", "syn.cflags=-DLLM_PORTS=1 -DLLM_LANES=4 -DLLM_LANES_ATT=8 -DLLM_MAX_DIM=512", ""]), encoding="utf-8")
LB = fit_check.kernel_limits(config=cfgfile)
check("an HLS config's -D settings replace the header values (lanes 4, attention 8, dim 512)", LB["LLM_LANES"] == 4 and LB["LLM_LANES_ATT"] == 8 and LB["LLM_MAX_DIM"] == 512, str(LB))
c66, o66 = cfg_of(64, 66, 1, 4, 4, 64, 16)
r66 = fit_check.check(c66, o66, LB, None)
t66 = chr(10).join(fit_check.report_lines(r66))
check("hidden 66 with 4 lanes: not ok, and the report names the -8 error", not r66["kernel_ok"] and "returns -8" in t66, t66)
c15l, o15l = cfg_of(288, 768, 6, 6, 6, 32000, 256)
check("stories15M sizes with 4 lanes and 8 attention lanes: ok", fit_check.check(c15l, o15l, LB, None)["kernel_ok"])
check("lanes can be given with --kernel-limits too (lanes=2,lanes_att=16)", fit_check.kernel_limits(override="lanes=2,lanes_att=16")["LLM_LANES_ATT"] == 16)
check("--kernel-limits overrides single values", fit_check.kernel_limits(override="dim=512,seq=1024")["LLM_MAX_DIM"] == 512
      and fit_check.kernel_limits(override="dim=512,seq=1024")["LLM_MAX_HIDDEN"] == 768)
for bad in ("dim=abc", "banana=3", "dim"):
    try:
        fit_check.kernel_limits(override=bad)
        check(f"bad --kernel-limits '{bad}' is refused", False)
    except SystemExit:
        check(f"bad --kernel-limits '{bad}' is refused", True)

# ---- ground truth: the two models that ran on the board
for model, base, nstate, tok in (("stories260K", 0x01000000, 164352, 0x010A0800), ("stories15M", 0x04000000, 916736, 0x0437F400)):
    flat = ROOT / "models" / f"{model}.flat"
    if not flat.exists():
        print(f"SKIP  {model} (file not found)")
        continue
    c, o = layout.unpack_header(flat.read_bytes()[:256])
    r = fit_check.check(c, o, LIM, 1 * GIB)
    m = r["memory"]
    check(f"{model}: runs on the current kernel and fits the U50's 1 GiB", r["kernel_ok"] and r["fits_memory"] is True, str(r["rows"]))
    check(f"{model}: addresses equal what run_kernel.tcl printed on the board (state {base:#x}, nstate {nstate}, tokens {tok:#x})",
          (m["state_base"], m["state_floats"], m["tokens_addr"]) == (base, nstate, tok), str(m))

# ---- a TinyLlama-sized model: needs a rebuild AND does not fit
c, o = cfg_of(2048, 5632, 22, 32, 4, 32000, 2048, shared=False)
r = fit_check.check(c, o, LIM, 1 * GIB)
text = "\n".join(fit_check.report_lines(r))
check("TinyLlama-sized: exceeds dim, hidden, context and heads, but not head size",
      [x[0] for x in r["rows"] if not x[4]] == ["dim", "hidden", "context", "heads"], str(r["rows"]))
check("TinyLlama-sized: the rebuild values are listed", r["rebuild_values"] == {"LLM_MAX_DIM": 2048, "LLM_MAX_HIDDEN": 5632, "LLM_MAX_SEQ": 2048, "LLM_MAX_HEADS": 32})
check("TinyLlama-sized: does not fit 1 GiB", r["fits_memory"] is False)
r16 = fit_check.check(c, o, LIM, 16 * GIB)
check("TinyLlama-sized: weights over 4 GiB are no longer a problem on a 16 GiB board", r16["fits_memory"] is True)
check("TinyLlama-sized: the verdict says rebuild and does-not-fit", "needs a kernel rebuild" in text and "DOES NOT FIT" in text and "does not fit the board's memory either" in text, text)

# ---- a mid-sized model: needs a rebuild but fits the memory
c, o = cfg_of(512, 1376, 8, 8, 8, 32000, 512)
r = fit_check.check(c, o, LIM, 1 * GIB)
check("42M-sized: needs a rebuild (dim, hidden) and fits the memory", [x[0] for x in r["rows"] if not x[4]] == ["dim", "hidden"] and r["fits_memory"] is True)
check("42M-sized: the on-chip working buffers stay small (a rebuild is cheap in memory)", r["buffers_needed"] > r["buffers_now"] and r["buffers_needed"] < 200 * 1024,
      f"{r['buffers_now']} -> {r['buffers_needed']}")
r2 = fit_check.check(c, o, fit_check.kernel_limits(override="dim=512,hidden=1376"), 1 * GIB)
check("42M-sized: runs once the kernel is built with those limits", r2["kernel_ok"] and "runs on the current kernel build" in "\n".join(fit_check.report_lines(r2)))
check("without a board the memory comparison is skipped", fit_check.check(c, o, LIM, None)["fits_memory"] is None)

# ---- the largest-model line
if (ROOT / "models" / "stories15M.flat").exists():
    c15, o15 = layout.unpack_header((ROOT / "models" / "stories15M.flat").read_bytes()[:256])
    r = fit_check.check(c15, o15, LIM, 1 * GIB)
    text = "\n".join(fit_check.report_lines(r))
    check("1 GiB board: largest model is about 250 million parameters, not capped", 240e6 <= r["max_params"] <= 260e6
          and "largest model this board can hold: about 250 million parameters" in text and "capped" not in text, text)
    r = fit_check.check(c15, o15, LIM, 16 * GIB)
    text = "\n".join(fit_check.report_lines(r))
    check("16 GiB board: largest model is about 4 billion parameters, no 4 GiB cap", 3.8e9 <= r["max_params"] <= 4.2e9 and "capped" not in text and "4 GiB" not in text, text)
    r = fit_check.check(c15, o15, LIM, None)
    check("without a board there is no largest-model line", r["max_params"] is None and "largest model" not in "\n".join(fit_check.report_lines(r)))
    r = fit_check.check(c15, o15, LIM, 5 * 2 ** 20)
    check("a board with 5 MB holds no model of this shape", r["max_params"] == 0 and "none" in "\n".join(fit_check.report_lines(r)))
check("human-readable sizes", [fit_check.human_params(n) for n in (0, 500_000, 7_400_000, 252_000_000, 1_050_000_000)]
      == ["none", "less than 1 million", "about 7 million", "about 250 million", "about 1.1 billion"], str([fit_check.human_params(n) for n in (0, 500_000, 7_400_000, 252_000_000, 1_050_000_000)]))

# ---- command line
header_only = TMP / "big.flat"
cbig, obig = cfg_of(2048, 5632, 22, 32, 4, 32000, 2048, shared=False)
header_only.write_bytes(layout.pack_header(cbig, obig))
run = lambda *a: subprocess.run([sys.executable, str(ROOT / "fit_check.py"), *map(str, a)], capture_output=True, text=True)
p = run(header_only, "--board", BOARD)
check("fit_check.py exits 1 and says why for a model that does not fit", p.returncode == 1 and "needs a kernel rebuild" in p.stdout and "DOES NOT FIT" in p.stdout, p.stdout + p.stderr)
if (ROOT / "models" / "stories15M.flat").exists():
    p = run(ROOT / "models" / "stories15M.flat", "--board", BOARD)
    check("fit_check.py exits 0 for stories15M", p.returncode == 0 and "runs on the current kernel build" in p.stdout, p.stdout + p.stderr)
p = run(header_only, "--kernel-limits", "dim=2048,hidden=5632,seq=2048,heads=32")
check("fit_check.py without a board accepts --kernel-limits and skips the memory comparison", p.returncode == 0 and "give --board" in p.stdout, p.stdout + p.stderr)

# ---- inside hf_to_flat.py
cfg, t = random_model(7)
folder = write_hf(TMP / "m", cfg, t)
msgs = []
hf_to_flat.convert(folder, TMP / "m.flat", log=msgs.append)
check("hf_to_flat prints the fit check before writing", any("FIT CHECK" in m for m in msgs) and any("runs on the current kernel build" in m for m in msgs), str(msgs))
tiny_board = TMP / "tiny_board.json"
tiny_board.write_text(json.dumps({"bram36": 10, "ext_gb": 0.01}))
msgs = []
hf_to_flat.convert(folder, TMP / "m2.flat", log=msgs.append, board=str(tiny_board))
check("a too-small board is reported as not fitting, and the file is still written", any("DOES NOT FIT" in m for m in msgs) and (TMP / "m2.flat").exists(), str(msgs))
msgs = []
hf_to_flat.convert(folder, TMP / "m3.flat", log=msgs.append, kernel_limits="dim=8")
check("hf_to_flat --kernel-limits changes the verdict (dim 32 > 8)", any("EXCEEDS" in m for m in msgs) and any("needs a kernel rebuild with LLM_MAX_DIM=32" in m for m in msgs), str(msgs))

print(f"\n{sum(results)} of {len(results)} checks passed")
import shutil  # noqa: E402
shutil.rmtree(TMP, ignore_errors=True)
sys.exit(0 if all(results) else 1)
