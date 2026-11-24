"""Tests for hls_check.py (no Vitis needed): the parser is run on the saved reports of a real HLS run.   usage: python tests/test_hls_check.py
The fixture is the run of 2 weight ports x 8 lanes + 4 attention lanes (tests/fixtures/hls_2port)."""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "ref")]
import hls_check  # noqa: E402

FIX = ROOT / "tests" / "fixtures" / "hls_2port"
results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(("PASS  " if ok else "FAIL  ") + name + ("" if ok else f"\n      {detail}"))


res = hls_check.collect(FIX)
check("resources are read from the report: DSP 295, BRAM 104, LUT 60920, FF 65590, URAM 0",
      res["used"] == {"BRAM_18K": 104, "DSP": 295, "FF": 65590, "LUT": 60920, "URAM": 0}, str(res.get("used")))
check("the available resources are read too (1824 DSPs, 960 BRAM18, 64 URAM)", res["available"]["DSP"] == 1824 and res["available"]["BRAM_18K"] == 960 and res["available"]["URAM"] == 64, str(res.get("available")))
check("the clock estimate and target are read (13.491 ns against 10 ns)", res["clock_estimated_ns"] == 13.491 and res["clock_target_ns"] == 10.0, str(res))
check("the memory port widths are read (weights 256, state 128, tokens 32, second weight port 256)",
      res["ports"] == {"gmem0": 256, "gmem1": 128, "gmem2": 32, "gmem3": 256}, str(res["ports"]))
check("no loop that was asked for one iteration per cycle missed it", res["loops_missed"] == [], str(res["loops_missed"]))
check("the serial loops are listed (rmsnorm sum II 3, softmax II 3, a tree loop II 4)",
      {"VITIS_LOOP_108_1", "VITIS_LOOP_304_2", "VITIS_LOOP_119_2"} <= set(res["loops_serial"]), str(res["loops_serial"]))
check("no errors in the log", res["errors"] == [])
# the fixture is from an earlier version of the kernel: give the lookup a stand-in source file with the same line numbers
src = Path(tempfile.mkdtemp()) / "kernel_then.cpp"
stand_in = [""] * 400
stand_in[107] = "    for (int j = 0; j < n; j++) ss += in[j] * in[j];"
stand_in[303] = "    for (int i = 0; i < n; i++) { v[i] = expf(v[i] - mx); sum += v[i]; }"
src.write_text(chr(10).join(stand_in), encoding="utf-8")
text = chr(10).join(hls_check.report_lines(res, {"LLM_PORTS": 2, "LLM_LANES": 8, "LLM_LANES_ATT": 4}, source=src))
check("the report shows the resources with percentages and the planner's DSP prediction", "DSP 295 of 1824 (16%)" in text and "predicted" in text, text)
check("the report names the kernel source line of a serial loop", "ss += in[j] * in[j]" in text, text)

# a log with a missed loop and an error
log = ("INFO: [HLS 200-1470] Pipelining result : Target II = 1, Final II = 2, Depth = 12, loop 'VITIS_LOOP_148_1'\n"
       "WARNING: [HLS 200-448] Lower bound of II is 2 due to multiple operations accessing 'gmem0' m_axi read: x\n"
       "ERROR: [HLS 200-1715] Encountered problem during source synthesis\n")
bad = hls_check.parse_reports("", log, "")
check("a missed loop and an error are reported", bad["loops_missed"] == ["VITIS_LOOP_148_1"] and bad["ii_warnings"] == 1 and len(bad["errors"]) == 1, str(bad))
check("a UTF-16 log (PowerShell redirect) is decoded", "Final II" in hls_check.decode(log.encode("utf-16")))
check("the settings written by kernel_plan.py are read", hls_check.cflags_dict("-DLLM_PORTS=2 -DLLM_LANES=8") == {"LLM_PORTS": 2, "LLM_LANES": 8})

# the command line, without running HLS
p = subprocess.run([sys.executable, str(ROOT / "hls_check.py"), "--parse-only", str(FIX)], capture_output=True, text=True)
check("command line --parse-only prints the report and exits 0", p.returncode == 0 and "DSP 295" in p.stdout, p.stdout + p.stderr)
with tempfile.TemporaryDirectory() as d:
    (Path(d) / "hls.log").write_text(log, encoding="utf-8")
    p = subprocess.run([sys.executable, str(ROOT / "hls_check.py"), "--parse-only", d], capture_output=True, text=True)
    check("command line --parse-only exits 1 when the log has an error", p.returncode == 1 and "errors" in p.stdout, p.stdout + p.stderr)

print(f"\n{sum(results)} of {len(results)} checks passed")
sys.exit(0 if all(results) else 1)
