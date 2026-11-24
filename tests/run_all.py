"""Run every test and print one line per test and a total.   usage: python tests/run_all.py [--fast] [--only name ...] [-v]

  --fast        skip the slow kernel builds (test_lanes, about 3-5 minutes)
  --only NAME   run only the tests whose file name contains NAME (repeatable)
  -v            print the lines that did not pass, for the tests that failed

Each test file prints 'N of M checks passed' (or SKIP when a tool it needs is missing). The tests run side by side, four at a time (one is run alone at the end).
Exit code 0 when every test passed (a skipped test is not a failure), 1 otherwise."""
import argparse
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SLOW = {"test_lanes"}
ALONE = set()


def run_one(path, timeout=1500):
    t0 = time.time()
    try:
        p = subprocess.run([sys.executable, str(path)], capture_output=True, text=True, cwd=str(ROOT), timeout=timeout)
        out, code = p.stdout + p.stderr, p.returncode
    except subprocess.TimeoutExpired:
        return path.stem, "TIMEOUT", f"stopped after {timeout} s", time.time() - t0, ""
    m = re.findall(r"(\d+) of (\d+) checks passed", out)
    if m:
        ok, total = map(int, m[-1])
        status = "PASS" if ok == total and code == 0 else "FAIL"
        return path.stem, status, f"{ok} of {total} checks", time.time() - t0, out
    if "SKIP" in out and code == 0:
        return path.stem, "SKIP", next((l for l in out.splitlines() if l.startswith("SKIP")), "skipped"), time.time() - t0, out
    # a script that prints no summary: judge by the exit code (compare_golden.py and test_big_offsets.py)
    last = out.strip().splitlines()[-1][:70] if out.strip() else "(no output)"
    return path.stem, ("PASS" if code == 0 else "FAIL"), last, time.time() - t0, out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--only", action="append", default=[])
    ap.add_argument("-v", action="store_true")
    a = ap.parse_args()
    files = sorted((ROOT / "tests").glob("test_*.py")) + [ROOT / "tests" / "compare_golden.py"]
    if a.fast:
        files = [f for f in files if f.stem not in SLOW]
    if a.only:
        files = [f for f in files if any(k in f.stem for k in a.only)]
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(run_one, [f for f in files if f.stem not in ALONE]))
    results += [run_one(f) for f in files if f.stem in ALONE]
    results.sort(key=lambda r: [f.stem for f in files].index(r[0]))
    width = max(len(r[0]) for r in results)
    for name, status, detail, secs, _ in results:
        print(f"{status:<8}{name:<{width}}  {detail}  ({secs:.0f} s)")
        if a.v and status in ("FAIL", "TIMEOUT"):
            for line in results[[r[0] for r in results].index(name)][4].splitlines():
                if line.startswith("FAIL") or line.startswith("Traceback") or "Error" in line:
                    print("          " + line[:160])
    bad = [r for r in results if r[1] in ("FAIL", "TIMEOUT")]
    print(f"\n{len(results) - len(bad)} of {len(results)} tests passed" + (f", failed: {', '.join(r[0] for r in bad)}" if bad else "") + f"  ({time.time() - t0:.0f} s)")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
