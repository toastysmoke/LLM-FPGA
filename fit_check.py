#!/usr/bin/env python3
"""
fit_check.py - will this model run on the kernel that is built, and does it fit the board's memory?

  python fit_check.py model.flat [--board boards/u50.json] [--kernel-limits dim=288,hidden=768,seq=512,heads=16,head=64]

Two checks:
  1. Kernel limits. The kernel keeps its working buffers in on-chip arrays whose sizes are fixed when it is built (LLM_MAX_* in
     sw/llm_kernel.h). A model that exceeds one of them is refused by the kernel (error -4). The limits are read from that header;
     if your bitstream was built with other -D values, give them with --kernel-limits. If a limit is exceeded the check says
     which values a rebuild needs.
  2. Memory. Weights, the state area (logits + K/V cache) and the token buffer must fit in the board's external memory, laid out
     the way the host loader lays them out. The size comes from the board file (ext_gb). There is no separate limit on the
     weight file size: the kernel uses 64-bit positions (a bitstream built before that change is limited to 4 GiB).
Also used by hf_to_flat.py, which prints this verdict before it converts.
"""
import argparse
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "ref"))
import layout  # noqa: E402

DEFAULT_HEADER = HERE / "sw" / "llm_kernel.h"
# name shown, define in the header, short key for --kernel-limits
LIMITS = [("dim", "LLM_MAX_DIM", "dim"), ("hidden", "LLM_MAX_HIDDEN", "hidden"), ("context", "LLM_MAX_SEQ", "seq"),
          ("heads", "LLM_MAX_HEADS", "heads"), ("head size", "LLM_MAX_HEAD_SIZE", "head")]
S_BASE_ALIGN = 1 << 24          # run_kernel.tcl puts the state area on the next 16 MB boundary above the weights


LANES = [("LLM_LANES", "lanes"), ("LLM_LANES_ATT", "lanes_att")]     # set when the kernel is built; the model must divide by them


def config_defines(path):
    """-> {name: value} of the -D settings in the 'syn.cflags' line of an HLS config or a kernel_defines.cfg (kernel_plan.py)"""
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("syn.cflags"):
            return {m.group(1): int(m.group(2)) for m in re.finditer(r"-D(\w+)=(\d+)", line)}
    raise SystemExit(f"fit_check: no syn.cflags line in {path}")


def kernel_limits(header=DEFAULT_HEADER, override=None, config=None):
    """-> {define name: value}: the buffer limits and the lane counts of the kernel build. Read from the kernel header; `config` (the
    HLS config or kernel_defines.cfg the bitstream was built with) replaces them with the build's -D settings; `override`
    ('dim=288,seq=1024,lanes=4') replaces single values."""
    text = Path(header).read_text(encoding="utf-8")
    lim = {}
    for _, define, _ in LIMITS:
        m = re.search(r"#define\s+" + define + r"\s+(\d+)", text)
        if not m:
            raise SystemExit(f"fit_check: {define} not found in {header}")
        lim[define] = int(m.group(1))
    m = re.search(r"#define\s+LLM_LANES\s+(\d+)", text)
    lim["LLM_LANES"] = int(m.group(1)) if m else 1
    m = re.search(r"#define\s+LLM_LANES_ATT\s+(\d+)", text)
    lim["LLM_LANES_ATT"] = int(m.group(1)) if m else None          # default: the same as LLM_LANES
    if config:
        d = config_defines(config)
        for define in [x[1] for x in LIMITS] + [x[0] for x in LANES]:
            if define in d:
                lim[define] = d[define]
    if override:
        keys = {short: define for _, define, short in LIMITS}
        keys.update({short: define for define, short in LANES})
        for part in override.split(","):
            k, _, v = part.partition("=")
            if k.strip() not in keys or not v.strip().isdigit():
                raise SystemExit(f"fit_check: bad --kernel-limits entry '{part}' (use {','.join(s + '=N' for _, _, s in LIMITS)},lanes=N,lanes_att=N)")
            lim[keys[k.strip()]] = int(v)
    if lim["LLM_LANES_ATT"] is None:
        lim["LLM_LANES_ATT"] = lim["LLM_LANES"]
    return lim


def needs(cfg):
    """what the model asks of each limit"""
    return {"LLM_MAX_DIM": cfg["dim"], "LLM_MAX_HIDDEN": cfg["hidden_dim"], "LLM_MAX_SEQ": cfg["seq_len"],
            "LLM_MAX_HEADS": cfg["n_heads"], "LLM_MAX_HEAD_SIZE": cfg["dim"] // cfg["n_heads"]}


def memory_layout(cfg, weights_bytes):
    """addresses as the host loader computes them: weights at 0, state at the next 16 MB boundary, then the token buffer"""
    kvd = cfg["dim"] * cfg["n_kv_heads"] // cfg["n_heads"]
    nstate = (cfg["vocab_size"] + 15) // 16 * 16 + 2 * cfg["n_layers"] * cfg["seq_len"] * kvd          # floats
    s_base = (weights_bytes + S_BASE_ALIGN - 1) // S_BASE_ALIGN * S_BASE_ALIGN
    tok_addr = s_base + (4 * nstate + 63) // 64 * 64
    end = tok_addr + 4 * (cfg["seq_len"] + 1)
    return {"weights_bytes": weights_bytes, "state_base": s_base, "state_floats": nstate, "state_bytes": 4 * nstate,
            "tokens_addr": tok_addr, "end": end}


def onchip_buffer_bytes(lim):
    """rough size of the kernel's static arrays at these limits (sw/llm_kernel.cpp: x xb xb2 q, kvec vvec, hb hb2, the attention scores
    of every head, normw, the matrix input copy, kvrow, ropec ropes), fp32"""
    d, h, s, hs, nh = lim["LLM_MAX_DIM"], lim["LLM_MAX_HIDDEN"], lim["LLM_MAX_SEQ"], lim["LLM_MAX_HEAD_SIZE"], lim["LLM_MAX_HEADS"]
    return 4 * (7 * d + 2 * h + nh * s + max(d, h) + d + hs)


def check(cfg, o, limits, ext_bytes=None):
    """Compare the model (header values) with the kernel's build limits and, if the board's memory size is given, with the board's memory. Returns a dict of per-limit rows, lane-multiple checks and the memory layout."""
    need = needs(cfg)
    rows = [(name, define, need[define], limits[define], need[define] <= limits[define]) for name, define, _ in LIMITS]
    # the lanes: rows of the weights and K/V slices are read in whole wide words, so the model's sizes must be multiples (else error -8)
    lanes, lanes_att = limits.get("LLM_LANES", 1), limits.get("LLM_LANES_ATT", limits.get("LLM_LANES", 1))
    hs = cfg["dim"] // cfg["n_heads"]
    lane_rows = [("dim", cfg["dim"], lanes, cfg["dim"] % lanes == 0), ("hidden", cfg["hidden_dim"], lanes, cfg["hidden_dim"] % lanes == 0),
                 ("head size", hs, lanes_att, hs % lanes_att == 0)]
    mem = memory_layout(cfg, o["total"])
    fits_mem = None if ext_bytes is None else mem["end"] <= ext_bytes
    # rough ceiling on the model size: the usable memory minus 5% kept for the K/V cache of a larger model, minus this model's own
    # state/token area and alignment gap, divided by 4 bytes per weight (fp32)
    max_params = None
    if ext_bytes is not None:
        max_params = max(0, int((ext_bytes * 0.95 - (mem["end"] - mem["weights_bytes"])) // 4))
    return {"rows": rows, "memory": mem, "ext_bytes": ext_bytes, "fits_memory": fits_mem,
            "max_params": max_params, "kernel_ok": all(r[4] for r in rows) and all(x[3] for x in lane_rows),
            "lanes_ok": all(x[3] for x in lane_rows), "lane_rows": lane_rows, "lanes": (lanes, lanes_att),
            "rebuild_values": {r[1]: r[2] for r in rows if not r[4]},
            "buffers_now": onchip_buffer_bytes(limits),
            "buffers_needed": onchip_buffer_bytes({**limits, **{r[1]: r[2] for r in rows if not r[4]}})}


def human_params(n):
    """Format a parameter count as 'about N million/billion' for the report."""
    if n >= 10 ** 9:
        return f"about {n / 1e9:.1f} billion"
    if n >= 10 ** 8:
        return f"about {round(n / 1e7) * 10} million"
    if n >= 10 ** 6:
        return f"about {round(n / 1e6)} million"
    return "less than 1 million" if n > 0 else "none"


def report_lines(res):
    """Turn the result of check() into the printable report (one line per limit, then the memory table and the verdict)."""
    mb = lambda b: f"{b / 2**20:,.1f} MB"
    out = ["FIT CHECK  (kernel limits from the kernel header; memory laid out as run_kernel.tcl does)"]
    for name, define, have, cap, ok in res["rows"]:
        out.append(f"  {name:<10}{have:>8} {'<=' if ok else '> '} {cap:<8}{'ok' if ok else 'EXCEEDS  (needs ' + define + ' >= ' + str(have) + ')'}")
    if res.get("lane_rows") and res["lanes"] != (1, 1):
        bad = [f"{n} {v} is not a multiple of {l}" for n, v, l, ok in res["lane_rows"] if not ok]
        out.append(f"  lanes     {res['lanes'][0]} per weight port, {res['lanes'][1]} attention: " + ("ok" if not bad else "NO: " + "; ".join(bad) + " (the kernel returns -8)"))
    m = res["memory"]
    out.append(f"  memory    weights {mb(m['weights_bytes'])} + state {mb(m['state_bytes'])} (starts at {m['state_base']:#x}) + tokens "
               f"= {mb(m['end'])} used")
    if res["ext_bytes"] is not None:
        out.append(f"            board external memory {mb(res['ext_bytes'])}: " + ("fits" if res["fits_memory"] else "DOES NOT FIT"))
        out.append(f"            largest model this board can hold: {human_params(res['max_params'])} parameters (32-bit weights, rough)")
    else:
        out.append("            (give --board to compare with the board's external memory)")
    if not res["kernel_ok"] and res["rebuild_values"]:
        vals = " ".join(f"{k}={v}" for k, v in res["rebuild_values"].items())
        out.append(f"  verdict   needs a kernel rebuild with {vals}  (working buffers {res['buffers_now'] / 1024:.0f} KB -> "
                   f"about {res['buffers_needed'] / 1024:.0f} KB on-chip)")
    elif not res["kernel_ok"]:
        out.append("  verdict   needs a kernel rebuild with fewer lanes (the model's sizes must be multiples of them)")
    else:
        out.append("  verdict   runs on the current kernel build" + ("" if res["fits_memory"] is not False else ", but does not fit the board's memory"))
    if not res["kernel_ok"] and res["fits_memory"] is False:
        out.append("            and it does not fit the board's memory either")
    return out


def main():
    """Command line: read the .flat header, run check(), print the report; exit code 1 if the model does not fit."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("flat", help="the .flat weight file (only its 256-byte header is read)")
    p.add_argument("--board", help="board JSON file (ext_gb is used)")
    p.add_argument("--kernel-limits", help="limits the kernel was built with, if not the header defaults, e.g. dim=512,seq=1024")
    p.add_argument("--kernel-header", default=str(DEFAULT_HEADER), help="kernel header to read the limits from")
    p.add_argument("--kernel-config", help="the HLS config (or kernel_defines.cfg) the bitstream was built with: its limits and lanes are used")
    a = p.parse_args()
    with open(a.flat, "rb") as f:
        cfg, o = layout.unpack_header(f.read(layout.HEADER_BYTES))
    ext = None
    if a.board:
        sys.path.insert(0, str(HERE))
        from model_to_fpga_spec import load_board
        ext = load_board(a.board)["ext_bytes"]
    res = check(cfg, o, kernel_limits(a.kernel_header, a.kernel_limits, a.kernel_config), ext)
    print("\n".join(report_lines(res)))
    sys.exit(0 if res["kernel_ok"] and res["fits_memory"] is not False else 1)


if __name__ == "__main__":
    main()
