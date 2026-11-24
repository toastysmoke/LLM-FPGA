"""Tests for kernel_plan.py (no board needed).   usage: python tests/test_kernel_plan.py
Checks the minimum and maximum, that more DSPs never make the estimate slower, the lane rules, the memory-bound case, the command line,
and that the settings it writes really build and run correctly on the PC."""
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "tests"), str(ROOT / "ref")]
import kernel_plan  # noqa: E402
import layout  # noqa: E402
import reference  # noqa: E402
from model_to_fpga_spec import load_board  # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="kplan_"))
BOARD_FILE = ROOT / "tests" / "fixtures" / "small_board.json"
BOARD = load_board(BOARD_FILE)
results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(("PASS  " if ok else "FAIL  ") + name + ("" if ok else f"\n      {detail}"))


def header(dim, hidden, layers, heads, kv, vocab, seq, shared=True):
    c = dict(dim=dim, hidden_dim=hidden, n_layers=layers, n_heads=heads, n_kv_heads=kv, vocab_size=vocab, seq_len=seq, shared=shared)
    return c, layout.compute(c)


c15, o15 = header(288, 768, 6, 6, 6, 32000, 256)
c260, o260 = header(64, 172, 5, 8, 4, 512, 512)

# ---- minimum
mn = kernel_plan.plan(c15, o15, BOARD, 1)
check("below the minimum: refused, and the minimum percentage is named", not mn["ok"] and "at least" in mn["problem"] and f"{mn['min_percent']:g}%" in mn["problem"], str(mn))
at_min = kernel_plan.plan(c15, o15, BOARD, mn["min_percent"])
check("at the minimum percentage the smallest kernel is planned: one lane in each unit", at_min["ok"] and (at_min["lanes_mv"], at_min["lanes_att"]) == (1, 1), str(at_min))
check("the minimum is 95 DSPs ((62 + 13.2 + 10.5) x 1.10 margin) = 5.3% of 1824 (rounded up)", at_min["min_dsp"] == 95 and mn["min_percent"] == 5.3, str(mn))
nodsp = kernel_plan.plan(c15, o15, {"name": "x"}, 50)
check("a board file without 'dsp' is refused with a hint", not nodsp["ok"] and "dsp" in nodsp["problem"])

# ---- rules for any percentage
prev, ok_mono, ok_rules, ok_budget = 0, True, True, True
for pct in (5.3, 6, 8, 10, 15, 20, 30, 40, 50, 60, 80, 100):
    for cfg, o in ((c15, o15), (c260, o260)):
        p = kernel_plan.plan(cfg, o, BOARD, pct)
        if not p["ok"]:
            ok_rules = False
            continue
        for lanes in (p["lanes_mv"], p["lanes_att"]):
            ok_rules &= lanes in (1, 2, 4, 8, 16)
        ok_rules &= cfg["dim"] % p["lanes_mv"] == 0 and cfg["hidden_dim"] % p["lanes_mv"] == 0
        ok_rules &= layout.head_size(cfg) % p["lanes_att"] == 0
        ok_budget &= p["dsp_used"] <= BOARD["dsp"] * pct / 100 + 1e-9
    p = kernel_plan.plan(c15, o15, BOARD, pct)
    ok_mono &= p["tokens_per_s"] >= prev - 1e-9
    prev = p["tokens_per_s"]
check("lane counts are powers of two up to 16; matrix lanes divide dim and hidden, attention lanes divide the head size", ok_rules)
check("the DSPs used never exceed the chosen percentage", ok_budget)
check("a higher percentage never lowers the estimated speed", ok_mono)
p100 = kernel_plan.plan(c15, o15, BOARD, 100)
check("at 100% it stops when more lanes would not help, and says so", p100["ok"] and p100["at_maximum"] and p100["dsp_percent_used"] < 100
      and "not make it faster" in "\n".join(kernel_plan.report_lines(p100)), str(p100))
p260 = kernel_plan.plan(c260, o260, BOARD, 100)
check("stories260K (hidden 172, head size 8) is limited to 4 matrix lanes and 8 attention lanes", p260["lanes_mv"] <= 4 and p260["lanes_att"] <= 8, str(p260))
pw = kernel_plan.plan(c15, o15, BOARD, 30)
check("the K/V cache port is as wide as the attention lanes (32 bits per lane) and the report says so", f"K/V cache port {32 * pw['lanes_att']} bits wide" in chr(10).join(kernel_plan.report_lines(pw)))
check("more percentage gives more matrix lanes at first", kernel_plan.plan(c15, o15, BOARD, 10)["lanes_mv"] < kernel_plan.plan(c15, o15, BOARD, 30)["lanes_mv"])
pm = kernel_plan.plan(c15, o15, BOARD, 30)
check("the matrix unit gets at least as many lanes as the attention unit (its share of the work is far bigger)", pm["lanes_mv"] >= pm["lanes_att"])

# ---- memory bound
slow_mem = dict(BOARD, mem_gbps=0.05)
pmem = kernel_plan.plan(c15, o15, slow_mem, 60)
check("a slow memory caps the speed and the report says memory bandwidth limits it",
      pmem["bound"] == "memory bandwidth" and abs(pmem["tokens_per_s"] - pmem["memory_limit_tokens_per_s"]) < 1e-9
      and "limited by memory bandwidth" in "\n".join(kernel_plan.report_lines(pmem)), str(pmem))
check("without mem_gbps the report says how to add it", "mem_gbps" in "\n".join(kernel_plan.report_lines(kernel_plan.plan(c15, o15, dict(BOARD, mem_gbps=None), 50))))

# ---- weight ports (the memory channels)
one_port = dict(BOARD, max_ports=1)
ok1 = all(kernel_plan.plan(c15, o15, one_port, pct)["ports"] == 1 for pct in (6, 10, 30, 60, 100))
check("a board with one port never gets more than one weight port", ok1)
p2 = kernel_plan.plan(c15, o15, dict(BOARD, max_ports=4, mem_gbps=None), 100)
check("with no memory limit and 4 ports allowed, 100% uses more than one weight port", p2["ports"] > 1, str(p2))
pm = kernel_plan.plan(c15, o15, dict(BOARD, mem_gbps=1.0, max_ports=4), 100)
check("a memory that one port already saturates gets one port, and the report says the memory is the limit",
      pm["ports"] == 1 and pm["bound"] == "memory bandwidth" and "memory is the limit" in chr(10).join(kernel_plan.report_lines(pm)), str(pm))
pa = kernel_plan.plan(c15, o15, dict(BOARD, mem_gbps=5.33, clock_mhz=50), 100)
check("the U50 with 5.33 GB/s at 50 MHz (2 ports) reaches 2 ports of 16 lanes at 100%", pa["ports"] == 2 and pa["lanes_mv"] == 16, str(pa))
pr = kernel_plan.plan(c15, o15, BOARD, 100)
check("the U50 file as measured (3.42 GB/s, 70 MHz): one port, and the memory is the limit", pr["ports"] == 1 and pr["bound"] == "memory bandwidth", str(pr))
check("the written settings name the ports", f"-DLLM_PORTS={pa['ports']}" in kernel_plan.cflags(pa) and "2 weight ports" in chr(10).join(kernel_plan.report_lines(pa)))
prev, ok_mono2, ok_budget2 = 0, True, True
for pct in (5.3, 6, 10, 20, 30, 45, 60, 80, 100):
    q = kernel_plan.plan(c15, o15, dict(BOARD, max_ports=4), pct)
    ok_mono2 &= q["tokens_per_s"] >= prev - 1e-9
    ok_budget2 &= q["dsp_used"] <= BOARD["dsp"] * pct / 100 + 1e-9 and q["ports"] * q["lanes_mv"] >= 1
    prev = q["tokens_per_s"]
check("with ports allowed, a higher percentage never lowers the estimated speed and the DSPs stay within the percentage", ok_mono2 and ok_budget2)
check("ports are only added after a port is at its widest (16 lanes, or the model's limit)",
      all(q["ports"] == 1 or q["lanes_mv"] == q["max_lanes_mv"] for q in (kernel_plan.plan(c15, o15, dict(BOARD, max_ports=4), pc) for pc in (20, 40, 60, 80, 100))))

# ---- a 100B-class model: no crash, limits listed, 16 lanes possible
cb, ob = header(8192, 28672, 80, 64, 8, 128256, 8192, shared=False)
big_board = dict(BOARD, dsp=9024, bram36=2016, lut=1304000, ff=2607000, name="big")
pb = kernel_plan.plan(cb, ob, big_board, 60)
check("a 100B-class model plans on a big board with the limits it needs",
      pb["ok"] and pb["limits"]["LLM_MAX_DIM"] == 8192 and pb["limits"]["LLM_MAX_HIDDEN"] == 28672 and pb["lanes_mv"] == 16, str(pb))
hbm = dict(BOARD, dsp=9024, bram36=2016, lut=1304000, ff=2607000, name="hbm", mem_gbps=400.0, max_ports=4)
c7b, o7b = header(4096, 11008, 32, 32, 32, 32000, 4096, shared=False)
phb = kernel_plan.plan(c7b, o7b, hbm, 100)
check("a 7B-class model on an HBM-like board (400 GB/s, 4 ports allowed) uses 4 ports of 16 lanes", phb["ok"] and phb["ports"] == 4 and phb["lanes_mv"] == 16, str(phb))
p100h = kernel_plan.plan(cb, ob, hbm, 100)
check("a 100B-class model on that board is limited by the block RAM (2 MB of attention scores; UltraRAM is not used yet)",
      (not p100h["ok"]) or "block RAM" in p100h["bound"], str(p100h)[:300])

# ---- the other resources (block RAM, LUTs, flip-flops)
sweep_file = ROOT / "hls_sweep_results.json"
if sweep_file.exists():
    base_lim = {"LLM_MAX_DIM": 288, "LLM_MAX_HIDDEN": 768, "LLM_MAX_SEQ": 512, "LLM_MAX_HEADS": 16, "LLM_MAX_HEAD_SIZE": 64}
    worst = {"DSP": -1.0, "LUT": -1.0, "FF": -1.0}      # the largest (estimate - real) / real, over the runs; the smallest is checked as well
    under = 0
    bram_under = 0
    for run in json.loads(sweep_file.read_text()):
        est = kernel_plan.resources(run["ports_n"], run["lanes"], run["lanes_att"], {**base_lim, **run["extra"]})
        got = run["used"]
        for key, real in (("DSP", got["DSP"]), ("LUT", got["LUT"]), ("FF", got["FF"])):
            worst[key] = max(worst[key], (est[key] - real) / real)
            under += est[key] < 0.98 * real
        bram_under += est["BRAM18"] < 0.75 * got["BRAM_18K"]          # the block RAM estimate is meant to be an upper bound, within reason
    check("the DSP, LUT and FF estimates are at most 2% below and at most 30% above the 10 HLS runs they were fitted on",
          all(-0.02 <= v <= 0.30 for v in worst.values()), str(worst))
    check("the block RAM estimate is never more than 25% below an HLS run", bram_under == 0, f"{bram_under} runs")
    check("no DSP, LUT or FF estimate is more than 2% below its HLS run", under == 0, f"{under} estimates below")
    # a run that was NOT in the fit: 3 ports x 4 lanes, 2 attention lanes, dim 768, hidden 2048, context 1024 (HLS: DSP 257, BRAM18 92, LUT 54188, FF 56342)
    held = kernel_plan.resources(3, 4, 2, {"LLM_MAX_DIM": 768, "LLM_MAX_HIDDEN": 2048, "LLM_MAX_SEQ": 1024, "LLM_MAX_HEADS": 12, "LLM_MAX_HEAD_SIZE": 64})
    check("a configuration that was not in the fit: DSP, LUT, FF within 10% (not below), the block RAM above the real 92 (an upper bound, here about 2x)",
          all(0.99 * real <= held[k] <= 1.10 * real for k, real in (("DSP", 257), ("LUT", 54188), ("FF", 56342))) and held["BRAM18"] >= 92, str(held))
small_bram = dict(BOARD, bram36=45)                 # 90 BRAM18
pbr = kernel_plan.plan(c15, o15, small_bram, 100)
check("a board with little block RAM is limited by it: fewer lanes, the report names the block RAM, and the estimate stays under 90%",
      pbr["ok"] and pbr["resources"]["BRAM18"] <= 0.9 * 90 and "block RAM" in pbr["bound"] and pbr["lanes_mv"] * pbr["ports"] < 32, str(pbr))
small_lut = dict(BOARD, lut=60000)
pl = kernel_plan.plan(c15, o15, small_lut, 100)
check("a board with few LUTs is limited by them", pl["ok"] and pl["resources"]["LUT"] <= 0.9 * 60000 and "LUT" in pl["bound"], str(pl))
tiny = dict(BOARD, bram36=10)
pt = kernel_plan.plan(c15, o15, tiny, 100)
check("a board too small for even the smallest kernel is refused, naming the resource", not pt["ok"] and "BRAM18" in pt["problem"], str(pt))
chuge, ohuge = header(16384, 53248, 126, 128, 8, 128256, 131072, shared=False)
phuge = kernel_plan.plan(chuge, ohuge, BOARD, 100)
check("a model with a 16384 dimension and a 131072-token context does not plan on the U50: its buffers are too large for the block RAM",
      not phuge["ok"] and "too large" in phuge["problem"], str(phuge))
p100b = kernel_plan.plan(cb, ob, BOARD, 100)
check("the 100B-class model's buffers do not fit the U50's block RAM: the attention scores of 64 heads x 8192 positions are 2 MB",
      not p100b["ok"] and "too large" in p100b["problem"], str(p100b))
no_lut = {k: v for k, v in BOARD.items() if k not in ("lut", "ff")}
check("without lut and ff in the board file those two are simply not checked", kernel_plan.plan(c15, o15, no_lut, 100)["ok"])
txt = chr(10).join(kernel_plan.report_lines(kernel_plan.plan(c15, o15, BOARD, 30)))
check("the report shows block RAM, LUT and FF with the board's totals", "BRAM18" in txt and "of 960" in txt and "LUT" in txt and "of 216,960" in txt, txt)

# ---- command line
flat = ROOT / "models" / "stories15M.flat"
run = lambda *a: subprocess.run([sys.executable, str(ROOT / "kernel_plan.py"), *map(str, a)], capture_output=True, text=True)
if flat.exists():
    r = run(flat, "--board", BOARD_FILE, "--dsp-percent", 3, "-o", TMP / "x")
    check("command line: below the minimum exits 1 and writes nothing", r.returncode == 1 and "at least 5.3%" in r.stdout and not (TMP / "x").exists(), r.stdout + r.stderr)
    r = run(flat, "--board", BOARD_FILE, "--dsp-percent", 30, "-o", TMP / "out")
    cfgtext = (TMP / "out" / "kernel_defines.cfg").read_text() if (TMP / "out" / "kernel_defines.cfg").exists() else ""
    check("command line: 30% exits 0 and writes kernel_defines.cfg with the lane settings", r.returncode == 0 and "syn.cflags=" in cfgtext
          and "-DLLM_LANES=" in cfgtext and "-DLLM_LANES_ATT=" in cfgtext and "-DLLM_MAX_DIM=288" in cfgtext, r.stdout + r.stderr + cfgtext)
    check("command line: kernel_plan.json is valid JSON", json.loads((TMP / "out" / "kernel_plan.json").read_text())["ok"] is True)
    r = run(flat, "--board", BOARD_FILE, "--dsp-percent", 0)
    check("command line: 0% is rejected", r.returncode != 0)

# ---- the written settings build and run correctly on the PC
if not shutil.which("g++"):
    for d in glob.glob(r"C:\AMDDesignTools\*\Vitis\tps\mingw\*\win64.o\nt\bin"):
        if (Path(d) / "g++.exe").exists():
            os.environ["PATH"] = d + os.pathsep + os.environ["PATH"]
            break
if shutil.which("g++") and shutil.which("gcc") and (ROOT / "models" / "stories260K.flat").exists():
    src = TMP / "src"
    src.mkdir()
    for f in (ROOT / "sw").glob("*"):
        if f.suffix in (".c", ".cpp", ".h"):
            shutil.copy(f, src / f.name)
    subprocess.run(["gcc", "-O2", "-c", str(src / "llm.c"), "-o", str(src / "llm.o")], check=True, capture_output=True)
    f260 = ROOT / "models" / "stories260K.flat"
    for pct, board_used in ((6, BOARD), (25, BOARD), (100, BOARD), (100, dict(BOARD, max_ports=4, mem_gbps=None))):
        p = kernel_plan.plan(c260, o260, board_used, pct)
        flags = kernel_plan.cflags(p).split()
        exe = TMP / f"k{pct}_{p['ports']}_{p['lanes_mv']}_{p['lanes_att']}.exe"
        subprocess.run(["g++", "-O2", "-std=c++14", "-Wno-unknown-pragmas", *flags, "-o", str(exe), str(src / "llm_kernel.cpp"), str(src / "tb_kernel.cpp"),
                        str(src / "llm.o"), "-lm"], check=True, capture_output=True)
        lg = TMP / f"k{pct}_{p['ports']}_{p['lanes_mv']}_{p['lanes_att']}.logits"
        toks = list(map(int, subprocess.run([str(exe), str(f260), "20", str(lg), "1"], capture_output=True, text=True, check=True).stdout.split()))
        port_note = f"{p['ports']} ports x {p['lanes_mv']} lanes" 
        c, w = reference.load_flat(str(f260))
        rt, rl = reference.generate(c, w, [1], 20)
        diff = float(np.abs(np.fromfile(lg, dtype="<f4").reshape(len(rl), c["vocab_size"]) - np.array(rl)).max())
        check(f"{pct}% plan ({port_note}, attention {p['lanes_att']}): the written settings build and give the reference tokens (logit diff {diff:.1e})", toks == rt and diff < 1e-3)
else:
    print("SKIP  no C++ compiler (or stories260K.flat) for the build check")

# ---- the U50 calibration (boards/u50.json): the planner must reproduce the builds measured on the card on 2026-11-15 (stories15M, short run)
u50 = load_board(ROOT / "boards" / "u50.json")
f15 = ROOT / "models" / "stories15M.flat"
if f15.exists():
    import fit_check
    with open(f15, "rb") as fh:
        c15u, _o15u = layout.unpack_header(fh.read(layout.HEADER_BYTES))
    need15 = fit_check.needs(c15u)
    kernel_plan.use_board_model(u50)
    try:
        # (ports, lanes, attention lanes) -> (measured positions/s over 20 positions, routed LUT). The 4x16 build had all its ports on the same
        # HBM banks (3.6 cycles per word) and is NOT a calibration point for the speed; it only checks the resources.
        for (pt, ln, at), (meas_pps, meas_lut) in {(1, 1, 1): (3.90, 26873), (4, 16, 16): (None, 147144), (8, 16, 16): (408.0, 248275)}.items():
            clk = kernel_plan.clock_for(pt * ln, 125.0)
            if meas_pps:
                pps = clk * 1e6 / kernel_plan.cycles_per_token(c15u, (pt, ln), at, 10)
                check(f"U50 calibration {pt}x{ln}: speed {pps:.1f} within 8% of the measured {meas_pps} positions/s", abs(pps - meas_pps) / meas_pps < 0.08)
            lut = kernel_plan.resources(pt, ln, at, need15)["LUT"]
            check(f"U50 calibration {pt}x{ln}: LUT estimate {lut} is 0 to 10% above the routed {meas_lut}", 0 <= (lut - meas_lut) / meas_lut < 0.10)
        # held-out build (not in the fit): 4x16 with one HBM range per port, measured 2026-11-15 at 128.8 MHz: 301 positions/s; the planner was 17% low
        pps = 128.8e6 / kernel_plan.cycles_per_token(c15u, (4, 16), 16, 10)
        check(f"U50 held-out 4x16 (own HBM range per port): speed {pps:.0f} within 25% of the measured 301 positions/s", abs(pps - 301.0) / 301.0 < 0.25)
        check("U50 clock falls as the kernel grows (136.3, 132.5, 123.0 MHz at 1, 64, 128 lanes)",
              [round(kernel_plan.clock_for(m, 125.0), 1) for m in (1, 64, 128)] == [136.3, 132.5, 123.0] and kernel_plan.clock_for(256, 125.0) < 123.0)
    finally:
        kernel_plan.use_board_model({})
    check("without a calibration the old numbers are used unchanged (one word per cycle, one clock)",
          kernel_plan.weight_cycles(4) == 1.0 and kernel_plan.clock_for(100, 77.0) == 77.0)
else:
    print("SKIP  no stories15M.flat for the U50 calibration check")

print(f"\n{sum(results)} of {len(results)} checks passed")
shutil.rmtree(TMP, ignore_errors=True)
sys.exit(0 if all(results) else 1)
