#!/usr/bin/env python3
"""
hls_check.py - synthesize the kernel with one set of settings in Vitis HLS and print what came out.

  python hls_check.py --ports 2 --lanes 8 --lanes-att 4              (settings given directly)
  python hls_check.py --defines kernel_build/kernel_defines.cfg      (the settings kernel_plan.py wrote)
  python hls_check.py --parse-only some_folder                       (read the reports of an earlier run, run nothing)

It prints: resources used and available (block RAM, DSP, flip-flops, LUTs, UltraRAM), the clock the tool estimates against the target, the
width of every memory port, every loop that does not start a new iteration each clock cycle (with the line of the kernel it comes from),
and any error. The planner's DSP prediction for the same settings is printed next to the real count.
A run takes about 2-3 minutes. The numbers are HLS estimates, not the placed-and-routed result of Vivado.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE / "ref")]

DEFAULT_PART = "xcu50-fsvh2104-2-e"      # Alveo U50
KERNEL = HERE / "sw" / "llm_kernel.cpp"


def decode(raw):
    """a log written by v++ may be UTF-16 (PowerShell redirect) or UTF-8"""
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return raw.decode("utf-16", errors="replace")
    return raw.decode("utf-8", errors="replace")


def find_vpp():
    """Locate the v++ executable on PATH or in the usual Vitis install folders."""
    exe = shutil.which("v++")
    if exe:
        return exe
    import glob
    for pattern in (r"C:\AMDDesignTools\*\Vitis\bin\v++.bat", r"C:\Xilinx\Vitis\*\bin\v++.bat", "/opt/Xilinx/Vitis/*/bin/v++", "/tools/Xilinx/Vitis/*/bin/v++"):
        hits = sorted(glob.glob(pattern))
        if hits:
            return hits[-1]
    return None


def read_defines(path):
    """-> the 'syn.cflags' value of a kernel_defines.cfg written by kernel_plan.py"""
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.startswith("syn.cflags="):
            return line.split("=", 1)[1].strip()
    raise SystemExit(f"hls_check: no syn.cflags line in {path}")


def cflags_dict(cflags):
    """Parse -DNAME=value compiler flags into a dict of integers."""
    return {m.group(1): int(m.group(2)) for m in re.finditer(r"-D(\w+)=(\d+)", cflags)}


# ---------------------------------------------------------------- reading the reports
def parse_reports(rpt, log, verilog=""):
    """-> dict from the text of llm_top_csynth.rpt, the v++ log and llm_top.v (any may be empty)"""
    res = {"errors": [e.strip() for e in re.findall(r"^(?:ERROR|CRITICAL WARNING)[^\n]*", log, re.M)][:6]}
    # resource table: the 'Total', 'Available' rows under 'Utilization Estimates'
    names = ["BRAM_18K", "DSP", "FF", "LUT", "URAM"]
    sect = rpt.split("Utilization Estimates", 1)[-1] if "Utilization Estimates" in rpt else ""
    for key, tag in (("used", "Total"), ("available", "Available")):
        m = re.search(r"^\|" + tag + r"\s*\|([^\n]*)", sect, re.M)
        if m:
            cells = [c.strip() for c in m.group(1).split("|") if c.strip() != ""]
            res[key] = {n: int(v) for n, v in zip(names, cells) if v.isdigit()}
    m = re.search(r"\|ap_clk\s*\|\s*([\d.]+) ns\|\s*([\d.]+) ns\|", rpt)
    if m:
        res["clock_target_ns"], res["clock_estimated_ns"] = float(m.group(1)), float(m.group(2))
    # memory ports
    res["ports"] = {f"gmem{n}": int(w) for n, w in re.findall(r"C_M_AXI_GMEM(\d+)_DATA_WIDTH\s*=\s*(\d+)", verilog)}
    # loops: Target II / Final II per pipelined loop
    loops = {}
    for t, f, depth, name in re.findall(r"Pipelining result : Target II = (\S+), Final II = (\d+), Depth = (\d+), loop '([^']+)'", log):
        loops[name] = {"target_ii": t, "final_ii": int(f), "depth": int(depth)}
    res["loops_missed"] = sorted(n for n, v in loops.items() if v["target_ii"] == "1" and v["final_ii"] > 1)      # asked for 1, got more
    res["loops_serial"] = sorted(n for n, v in loops.items() if v["target_ii"] != "1" and v["final_ii"] > 1)       # no target, more than 1
    res["loops"] = loops
    res["ii_warnings"] = len(re.findall(r"Lower bound of II is \d+", log))
    return res


def loop_line(name, source=None):
    """'VITIS_LOOP_108_1' -> the kernel source line 108, trimmed (from `source`, default the kernel in sw/)"""
    source = Path(source) if source else KERNEL
    m = re.match(r"VITIS_LOOP_(\d+)_", name)
    if not m or not source.exists():
        return ""
    lines = source.read_text(encoding="utf-8", errors="replace").splitlines()
    n = int(m.group(1))
    return lines[n - 1].strip()[:70] if 0 < n <= len(lines) else ""


def predicted_dsp(flags):
    """the planner's DSP estimate for the same settings, or None"""
    try:
        import kernel_plan
        ports, lanes, att = flags.get("LLM_PORTS", 1), flags.get("LLM_LANES", 1), flags.get("LLM_LANES_ATT", flags.get("LLM_LANES", 1))
        return kernel_plan.dsp_cost(ports * lanes, att)
    except Exception:
        return None


def report_lines(res, flags=None, source=None):
    """Format the parsed HLS reports (resources, estimated clock, memory ports, loops that do not pipeline) as printable lines."""
    out = ["HLS CHECK  (Vitis HLS estimates, not the placed-and-routed result)"]
    if res.get("errors"):
        out.append("  errors:")
        out += ["    " + e[:200] for e in res["errors"]]
    if "used" in res:
        u, a = res["used"], res.get("available", {})
        parts = []
        for n, label in (("DSP", "DSP"), ("BRAM_18K", "BRAM18"), ("URAM", "URAM"), ("LUT", "LUT"), ("FF", "FF")):
            if n in u:
                pct = f" ({100 * u[n] / a[n]:.0f}%)" if a.get(n) else ""
                parts.append(f"{label} {u[n]}" + (f" of {a[n]}{pct}" if a.get(n) else ""))
        out.append("  resources      " + ", ".join(parts))
        pred = predicted_dsp(flags) if flags else None
        if pred is not None and "DSP" in u:
            out.append(f"  DSP predicted  {pred} by the planner, {u['DSP']} by HLS ({100 * (pred - u['DSP']) / max(u['DSP'], 1):+.0f}%)")
    if "clock_estimated_ns" in res:
        ok = res["clock_estimated_ns"] <= res["clock_target_ns"]
        out.append(f"  clock          estimated {res['clock_estimated_ns']:g} ns against a target of {res['clock_target_ns']:g} ns "
                   f"(about {1000 / res['clock_estimated_ns']:.0f} MHz)" + ("" if ok else "  <- slower than the target; fine for a 50 MHz kernel clock if under 20 ns"))
    if res.get("ports"):
        out.append("  memory ports   " + ", ".join(f"{k} {v} bits" for k, v in sorted(res["ports"].items())))
    if res.get("loops_missed"):
        out.append("  loops that missed one iteration per clock cycle (asked for 1):")
        for n in res["loops_missed"]:
            v = res["loops"][n]
            out.append(f"    {n:<34} II {v['final_ii']}   {loop_line(n, source)}")
    else:
        out.append("  loops          every loop that was asked for one iteration per cycle gets it")
    if res.get("loops_serial"):
        out.append("  serial loops (no target; more than 1 cycle per iteration):")
        for n in res["loops_serial"]:
            v = res["loops"][n]
            out.append(f"    {n:<34} II {v['final_ii']}   {loop_line(n, source)}")
    return out


# ---------------------------------------------------------------- running HLS
def make_cfg(work, cflags, part, clock_ns):
    """Write the small Vitis HLS config for one run: part, clock target, kernel source, top function and the -D settings."""
    cfg = work / "hls.cfg"
    cfg.write_text("\n".join([f"part={part}", "", "[hls]", f"clock={clock_ns}ns", f"syn.file={KERNEL.as_posix()}", "syn.top=llm_top",
                              f"syn.cflags={cflags}", ""]), encoding="ascii")
    return cfg


def kill_tree(proc):
    """Stop the HLS process and everything it started (it spawns child processes)."""
    if os.name == "nt":
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True)
    else:
        import signal
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def run_hls(work, cfg, vpp, timeout):
    """-> (returncode or None on timeout, seconds)"""
    t0 = time.time()
    with open(work / "hls.log", "wb") as log:
        proc = subprocess.Popen([vpp, "-c", "--mode", "hls", "--config", str(cfg), "--work_dir", str(work / "out")], stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=(os.name != "nt"))
        try:
            rc = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            kill_tree(proc)
            rc = None
    return rc, time.time() - t0


def find_file(root, name):
    """First file called name anywhere under root, or None."""
    hits = sorted(Path(root).rglob(name))
    return hits[0] if hits else None


def collect(work):
    """read the three files of a finished run (or of a folder given to --parse-only)"""
    rpt, log, v = find_file(work, "llm_top_csynth.rpt"), find_file(work, "hls.log"), find_file(work, "llm_top.v")
    txt = lambda p: decode(p.read_bytes()) if p else ""
    return parse_reports(txt(rpt), txt(log), txt(v))


def main():
    """Command line: build the settings, run Vitis HLS C synthesis (or parse an earlier run with --parse-only) and print the summary."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--defines", help="kernel_defines.cfg written by kernel_plan.py")
    ap.add_argument("--ports", type=int, help="LLM_PORTS")
    ap.add_argument("--lanes", type=int, help="LLM_LANES (per port)")
    ap.add_argument("--lanes-att", type=int, help="LLM_LANES_ATT")
    ap.add_argument("--acc-depth", type=int, help="LLM_ACC_DEPTH (also used for the attention unit)")
    ap.add_argument("--vec-sums", type=int, help="LLM_VEC_SUMS (running totals in the norm and softmax sums)")
    ap.add_argument("--cflags", help="extra -D settings, e.g. '-DLLM_MAX_DIM=512'")
    ap.add_argument("--name", help="label for the work folder (default: derived from the settings)")
    ap.add_argument("--work", help="folder for the run (default: a short folder in the temp directory; Windows needs short paths)")
    ap.add_argument("--part", default=DEFAULT_PART)
    ap.add_argument("--clock-ns", type=float, default=10.0, help="HLS target clock period (default 10)")
    ap.add_argument("--timeout", type=int, default=1800, help="seconds before the run is stopped (default 1800)")
    ap.add_argument("--parse-only", metavar="FOLDER", help="do not run; read the reports in this folder")
    a = ap.parse_args()

    if a.parse_only:
        res = collect(a.parse_only)
        print("\n".join(report_lines(res)))
        sys.exit(1 if res["errors"] else 0)

    flags_text = read_defines(a.defines) if a.defines else ""
    extra = []
    for opt, macro in ((a.ports, "LLM_PORTS"), (a.lanes, "LLM_LANES"), (a.lanes_att, "LLM_LANES_ATT")):
        if opt:
            extra.append(f"-D{macro}={opt}")
    if a.acc_depth:
        extra += [f"-DLLM_ACC_DEPTH={a.acc_depth}", f"-DLLM_ACC_DEPTH_ATT={a.acc_depth}"]
    if a.vec_sums:
        extra.append(f"-DLLM_VEC_SUMS={a.vec_sums}")
    if a.cflags:
        extra.append(a.cflags)
    cflags = " ".join([flags_text] + extra).strip()
    flags = cflags_dict(cflags)
    vpp = find_vpp()
    if not vpp:
        raise SystemExit("hls_check: Vitis HLS (v++) not found; put it on the PATH or use --parse-only")
    label = a.name or "p{}l{}a{}".format(flags.get("LLM_PORTS", 1), flags.get("LLM_LANES", 1), flags.get("LLM_LANES_ATT", flags.get("LLM_LANES", 1)))
    base = Path(a.work) if a.work else Path(os.environ.get("TEMP") or tempfile.gettempdir()) / "hx"
    work = base / label
    if work.exists():
        shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    cfg = make_cfg(work, cflags, a.part, a.clock_ns)
    print(f"running Vitis HLS ({label}); settings: {cflags or '(defaults)'}\nwork folder: {work}")
    rc, secs = run_hls(work, cfg, vpp, a.timeout)
    if rc is None:
        print(f"stopped after {a.timeout} s (timeout)")
        sys.exit(1)
    res = collect(work)
    res["settings"], res["seconds"] = cflags, round(secs)
    print(f"finished in {secs:.0f} s (exit code {rc})")
    print("\n".join(report_lines(res, flags)))
    (work / "hls_check.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    sys.exit(0 if rc == 0 and not res["errors"] and "used" in res else 1)


if __name__ == "__main__":
    main()
