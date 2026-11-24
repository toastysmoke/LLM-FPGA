"""Regression check for model_to_fpga_spec.py: run the current script on the synthetic models and compare what it PLACES with
tests/golden/* (made with the script BEFORE the board-file change and before the analysis options were removed).
usage: python tests/compare_golden.py [kind ...]      (default: all four kinds)

Compared exactly: fpga_spec.json, model_params.vh, memory_plan.vh, and in memory_map.json the tensor list, the summary, the
placement part of the plan and every segment's size / offset / on-chip-or-external choice.
Not compared (removed on purpose): per-token traffic, speed ceilings, the board's bandwidth and DSP count, schedule.json."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "model_to_fpga_spec.py"
OUT = ROOT / "tests" / "work_golden"
ARGS = ["--board", str(ROOT / "tests" / "fixtures" / "small_board.json"), "--max-ctx", "512"]
PLAN_KEYS = ["sram_total_bytes", "sram_util", "sram_usable_bytes", "external_capacity_bytes", "priority", "row_split", "max_ctx",
             "act_bytes", "kv_bytes", "tied_embedding_head", "moe_read_fraction", "sram_used_bytes", "ext_used_bytes",
             "fits_external", "by_class_bytes", "notes"]
SEG_KEYS = ["name", "kind", "cls", "layer", "bytes", "rows", "row_bytes", "sram_offset", "sram_bytes", "sram_rows", "ddr_offset",
            "ddr_bytes", "ddr_row_start", "space"]


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def compare(kind, gold, new):
    diffs = []
    if load(gold / "fpga_spec.json") != load(new / "fpga_spec.json"):
        diffs.append("fpga_spec.json")
    for f in ("model_params.vh", "memory_plan.vh"):
        if (gold / f).read_text(encoding="utf-8") != (new / f).read_text(encoding="utf-8"):
            diffs.append(f)
    g, n = load(gold / "memory_map.json"), load(new / "memory_map.json")
    for key in ("summary", "tensors"):
        if g[key] != n[key]:
            diffs.append(f"memory_map.json:{key}")
    for k in PLAN_KEYS:
        if g["plan"].get(k) != n["plan"].get(k):
            diffs.append(f"plan.{k}: {g['plan'].get(k)} -> {n['plan'].get(k)}")
    gs = [{k: s.get(k) for k in SEG_KEYS} for s in g["segments"]]
    ns = [{k: s.get(k) for k in SEG_KEYS} for s in n["segments"]]
    if gs != ns:
        diffs.append("segments")
    return diffs


def main():
    kinds = sys.argv[1:] or ["tiny", "tinyllama", "qknorm", "parallel"]
    bad = 0
    for k in kinds:
        out = OUT / k
        fake = ROOT / "tests" / "fake_models" / k
        if not fake.exists():
            subprocess.run([sys.executable, str(ROOT / "tests" / "make_fake_hf.py"), k, str(fake)], check=True, capture_output=True)
        r = subprocess.run([sys.executable, str(SCRIPT), str(fake), "-o", str(out)] + ARGS, capture_output=True, text=True)
        if r.returncode:
            print(f"{k}: script failed\n{r.stdout}\n{r.stderr}")
            bad += 1
            continue
        diffs = compare(k, ROOT / "tests" / "golden" / k, out)
        print(f"{k}: " + ("same placement and hardware description as before" if not diffs else "DIFFERENT -> " + "; ".join(diffs)))
        bad += bool(diffs)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
