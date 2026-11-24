"""Tests for the board file, the input checks, the removed options and the unsupported-model warning of
model_to_fpga_spec.py (synthetic models, no downloads).   usage: python tests/test_spec_script.py"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_fake_hf  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "model_to_fpga_spec.py"
BOARD = ROOT / "tests" / "fixtures" / "small_board.json"
WORK = ROOT / "tests" / "work"
shutil.rmtree(WORK, ignore_errors=True)
WORK.mkdir(parents=True)
results = []


def run(model, *extra, out="out"):
    r = subprocess.run([sys.executable, str(SCRIPT), str(model), "-o", str(WORK / out), *extra], capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr


def check(name, ok, detail=""):
    results.append(ok)
    print(("PASS  " if ok else "FAIL  ") + name + ("" if ok else f"\n      {detail}"))


def write(name, text):
    p = WORK / name
    p.write_text(text, encoding="utf-8")
    return p


tiny = make_fake_hf.build("tiny", WORK / "m_tiny")
par = make_fake_hf.build("parallel", WORK / "m_parallel")

# ---- 1. models that are not the supported Llama-style layout get a clear warning; Llama-style ones do not
code, out = run(tiny, "--board", str(BOARD))
check("Llama-style model: no warning", code == 0 and "WARNING" not in out and "only layer structure this tool supports" in out, out[-500:])
code, out = run(par, "--board", str(BOARD))
check("model with one norm per layer: warns", code == 0 and "norm(s) instead of 2" in out and "may be wrong" in out, out[-500:])

# ---- 2. the analysis options are gone
for opt in ("--layer-recipe x.py", "--sizing", "--schedule", "--sweep", "--target-tps 5", "--fclk-mhz 100", "--ext-eff 0.5", "--ctx-len 64"):
    check(f"removed option {opt.split()[0]} is refused", run(tiny, *opt.split())[0] != 0)
code, out = run(tiny, "--board", str(BOARD))
files = sorted(p.name for p in (WORK / "out").iterdir())
check("output files with a board", code == 0 and files == ["fpga_spec.json", "memory_map.json", "memory_plan.vh", "model_params.vh"], str(files))
code, out = run(tiny, out="out_no_board")
files = sorted(p.name for p in (WORK / "out_no_board").iterdir())
check("output files without a board", code == 0 and "model_params.vh" in files and "memory_plan.vh" not in files, str(files))

# ---- 3. board file errors
for name, content, expect in [
    ("missing ext_gb", {"sram_mb": 4}, "missing 'ext_gb'"),
    ("unknown field", {"sram_mb": 4, "ext_gb": 1, "banana": 1}, "unknown field"),
    ("removed field ext_gbs", {"sram_mb": 4, "ext_gb": 1, "ext_gbs": 3.2}, "unknown field"),
    ("removed field luts", {"sram_mb": 4, "ext_gb": 1, "luts": 100}, "unknown field"),
    ("no on-chip memory", {"ext_gb": 1}, "on-chip memory"),
    ("both sram forms", {"sram_mb": 4, "bram36": 10, "ext_gb": 1}, "not both"),
    ("negative number", {"sram_mb": -1, "ext_gb": 1}, "must be a number"),
    ("text instead of number", {"sram_mb": "lots", "ext_gb": 1}, "must be a number"),
]:
    p = write(f"b_{name.replace(' ', '_')}.json", json.dumps(content))
    code, out = run(par, "--board", str(p))
    check(f"board error: {name}", code != 0 and expect in out, f"exit {code}\n{out[-300:]}")
p = write("b_ok.json", json.dumps({"_comment": "keys starting with _ are ignored", "name": "my board", "sram_mb": 8, "ext_gb": 4}))
code, out = run(par, "--board", str(p))
check("a custom board is accepted and named in the output", code == 0 and "my board" in out, out[-500:])
p = write("b_min.json", json.dumps({"bram36": 100, "ext_gb": 2}))
code, out = run(par, "--board", str(p))
check("a minimal board (no name, no uram) works and uses the file name", code == 0 and "board b_min" in out, out[-300:])
p = write("b_null.json", json.dumps({"name": None, "bram36": 100, "uram": None, "ext_gb": 2}))
check("null for the optional fields works", run(par, "--board", str(p))[0] == 0)
check("board error: file not found", run(par, "--board", str(WORK / "nope.json"))[0] != 0)
for f in ("u50.json", "board_template.json"):
    code, out = run(tiny, "--board", str(ROOT / "boards" / f))
    check(f"the board file boards/{f} works", code == 0 and "MEMORY PLAN" in out, out[-300:])

# ---- 4. model input must be a Hugging Face folder with config.json
code, out = run(WORK / "m_tiny" / "model.safetensors")
check("input error: a single .safetensors file is refused", code != 0 and "Hugging Face folder" in out, out[-300:])
code, out = run(WORK / "model.gguf")
check("input error: a .gguf path is refused", code != 0 and "Hugging Face folder" in out, out[-300:])
nocfg = WORK / "m_nocfg"
nocfg.mkdir()
shutil.copy(tiny / "model.safetensors", nocfg / "model.safetensors")
code, out = run(nocfg)
check("input error: missing config.json", code != 0 and "no config.json" in out, out[-300:])
empty = WORK / "m_empty"
empty.mkdir()
code, out = run(empty)
check("input error: folder without safetensors", code != 0 and "no .safetensors" in out, out[-300:])

# ---- 5. --pack
data = make_fake_hf.build("tiny", WORK / "m_data", with_data=True)
code, out = run(data, "--pack", out="pack1")
sizes = {p.name: p.stat().st_size for p in (WORK / "pack1").iterdir()}
total = json.loads((WORK / "pack1" / "memory_map.json").read_text())["summary"]["total_bytes"] if code == 0 else -1
check("--pack without a board writes weights.bin of the right size", code == 0 and sizes.get("weights.bin") == total, f"{sizes} total {total}")
code, out = run(data, "--pack", "--board", str(BOARD), out="pack2")
names = {p.name for p in (WORK / "pack2").iterdir()}
check("--pack with a board writes the two images", code == 0 and {"weights_ext.bin", "weights_sram_init.bin"} <= names and "weights.bin" not in names, str(names))

print(f"\n{sum(results)} of {len(results)} checks passed")
shutil.rmtree(WORK, ignore_errors=True)
sys.exit(0 if all(results) else 1)
