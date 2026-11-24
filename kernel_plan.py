#!/usr/bin/env python3
"""
kernel_plan.py - choose the kernel's size limits and compute units for one model on one board.

  python kernel_plan.py model.flat --board boards/u50.json --dsp-percent 50 [--context 256] [-o kernel_build]

You choose one number: the share of the board's DSP slices the kernel may use. The tool does the rest:
  * the buffer limits (LLM_MAX_*) the model needs, from fit_check.py;
  * the split of the compute units between the matrix multiplications and the attention scores, in the ratio of the work each does
    per token for this model, so neither unit sits idle waiting for the other;
  * the weight ports and the lane count of each unit: lanes per port are 1, 2, 4, 8 or 16 (one port is at most 512 bits wide); more
    lanes than that need more weight ports, as many as the board file allows ('max_ports'), up to 8. The matrix lanes must
    divide the model's dimension and hidden size, the attention lanes its head size; each unit's memory port is as wide as its lanes.
It prints the settings with an estimate of the speed and writes them as HLS build settings (kernel_build/kernel_defines.cfg).

There is a minimum and a maximum. The smallest working kernel has one lane in each unit; a percentage below that is refused and the
minimum percentage is printed. A percentage above what the model can use is accepted, and the tool says how much it really uses.

The numbers behind the estimate (DSPs per lane, cycles per weight) come from the HLS reports and PC runs of this kernel. They are
estimates until measured on a board.
"""
import argparse
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE / "ref")]
import fit_check  # noqa: E402
import layout  # noqa: E402

# ---- estimates from the HLS reports of this kernel (xcku5p part, fp32). Change them when measured on a board.
# Resources of the synthesized kernel, fitted on 10 Vitis HLS runs (an HLS sweep: 1 to 4 weight ports, 1 to 16 lanes, 1 to 16 attention lanes,
# three sizes of buffers; kernel with streaming rows, VEC_SUMS 4). M = matrix lanes in all (ports x lanes per port), A = attention lanes.
#   DSP  = 62 + 13.2 M + 6.6 A                     (within 12% of the runs, high at the low end)
#   LUT  = 23293 + 1728 M + 2661 A                 (within 10%)
#   FF   = 22525 + 2092 M + 2835 A                 (within 8%)
#   and all three are multiplied by MODEL_MARGIN, because a run that was not in the fit (3 ports x 4 lanes, 2 attention lanes, mid-size
#   buffers) came out 9% (DSP, LUT) and 5% (FF) above the formulas.
#   BRAM18 = memory ports + on-chip arrays:
#       ports: every AXI port needs about 1.8 BRAM18 per 32-bit lane of its width (exact in the reports) and the token port 2
#       arrays: 30 + 4 x min(M, 16) as an upper bound (HLS moves small partitioned arrays to LUT memory, so the true value is not a line:
#       30, 32, 56, 92, 64, 52, 86 in the runs) plus about (2 + 2.9 x (ports - 1)) per 576 floats of buffers above the default sizes
# These are HLS estimates; the placed-and-routed result of Vivado differs a little, so a headroom is kept (RESOURCE_HEADROOM).
DSP_BASE, DSP_PER_MV_LANE, DSP_PER_ATT_LANE = 62, 13.2, 10.5   # attention lane raised from 6.6: the streamed attention (own multipliers
                                                            # and sum tree in the scores pass) measured 472 DSPs at 1 x 16 + 16 lanes
LUT_BASE, LUT_PER_MV_LANE, LUT_PER_ATT_LANE = 23293, 1728, 3100   # attention lane raised from 2661: streamed attention measured 109,491 LUTs at 1 x 16 + 16
FF_BASE, FF_PER_MV_LANE, FF_PER_ATT_LANE = 22525, 2092, 2835
BRAM_ARRAYS_BASE, BRAM_ARRAYS_PER_LANE, BRAM_ARRAY_LANE_CAP = 30, 4, 16
BRAM_ARRAYS_PER_ATT_LANE = 4.5           # arrays split into attention-lane banks: 1 port x 4 lanes with 8 attention lanes measured 80 BRAM18
                                         # of arrays where the matrix-lane term gives 46
SCORE_FLOATS_PER_BRAM18 = 370            # the attention scores of every head: 16 heads x 512 positions measured 22 BRAM18 (HLS, 2026-10-30)
BRAM_PER_EXTRA_KFLOAT = (2.0, 2.9)       # per 576 floats of buffers above the default sizes: (one port, extra per added port)
DEFAULT_BUFFER_FLOATS = 6 * 288 + 2 * 768 + 512
RESOURCE_HEADROOM = 0.90                  # the other resources may use this share of the board (routing, the rest of the design)
MODEL_MARGIN = 1.10                       # DSP, LUT and FF estimates are raised by this: a configuration that was not in the fit came out 9% higher
ACC_DEPTH = 4            # running totals per lane: a floating-point add takes about 4 cycles
LOAD_II = 1              # cycles per wide read (HLS report: II = 1 since the wide words are bit vectors)
FADD_CYCLES = 4          # latency of one floating-point add
CALL_OVERHEAD = 40       # cycles per matrix multiplication to start the streams and drain the last row
# The one-element-at-a-time parts of the kernel, from the loop table of the HLS run (hls_check.py): iteration interval II and pipeline depth.
# Everything runs at II = 1 except, with a single running total, the sum of squares in the RMSNorm and the exponentials of the softmax
# (chains of dependent adds); with VEC_SUMS = 4 those run at II = 1 too.
VEC_SUMS = 4             # running totals in the RMSNorm and softmax sums (LLM_VEC_SUMS): a floating-point add takes about 4 cycles
AXI_FIRST = 30           # cycles until the first word of a read arrives
SCORE_PASS_DEPTH = 35    # the streamed attention: one pass over all heads per position; pipeline depth of the scores pass (HLS: 35)
AV_PASS_DEPTH = 10       # and of the weighted-sum pass (HLS: 10). Before streaming, each head paid about 150 cycles per position
                         # (measured on a DDR4 test board: 244 positions in 10.7 s)
KV_READ_LATENCY = 100    # cycles each separate key or value slice read waits for DDR4: MEASURED on a DDR4 test board (2026-10-30, 1 port x 4 lanes,
                         # 8 attention lanes, stories15M): 244 positions took 24.25 s against 19.9 s predicted without it; short runs agree within 4%
DIV_SQRT = 60            # the divide, square root and divide of the RMSNorm scale
MAX_LANES = 16           # one memory port is at most 512 bits wide = 16 floats
MAX_PORTS = 8            # weight ports the kernel can have (LLM_PORTS; 8 since 2026-11-15, was 4)
DEFAULT_CLOCK_MHZ = 50.0

# ---- per-board calibration (added 2026-11-15). The constants above were fitted on a DDR4 test board; a board file may carry its own numbers, taken
# from routed builds measured on that board, under these keys (all optional; without them the numbers above are used unchanged):
#   "resource_model": {"DSP": [base, per matrix lane, per attention lane], "LUT": [...], "FF": [...], "margin": 1.05, "bram_factor": 0.4}
#   "weight_cycles":  [a, b]   clock cycles per weight word of a matrix port = a + b / lanes_per_port   (default [1, 0]: one word per cycle)
#   "clock_by_lanes": [[matrix lanes in all, kernel MHz], ...]   the kernel clock after place and route falls as the kernel grows;
#                     linear between the points, and beyond the last point along the last segment
_DEFAULT_MODEL = {"DSP": (DSP_BASE, DSP_PER_MV_LANE, DSP_PER_ATT_LANE), "LUT": (LUT_BASE, LUT_PER_MV_LANE, LUT_PER_ATT_LANE),
                  "FF": (FF_BASE, FF_PER_MV_LANE, FF_PER_ATT_LANE), "margin": MODEL_MARGIN, "bram_factor": 1.0,
                  "weight_cycles": (float(LOAD_II), 0.0), "clock_by_lanes": None}
_model = dict(_DEFAULT_MODEL)


def use_board_model(board):
    """set the module's models from the board file (or back to the defaults); returns nothing"""
    global _model
    m = dict(_DEFAULT_MODEL)
    rm = board.get("resource_model") or {}
    for k in ("DSP", "LUT", "FF"):
        if k in rm:
            m[k] = tuple(rm[k])
    if "margin" in rm:
        m["margin"] = float(rm["margin"])
    if "bram_factor" in rm:
        m["bram_factor"] = float(rm["bram_factor"])
    if board.get("weight_cycles"):
        m["weight_cycles"] = tuple(float(x) for x in board["weight_cycles"])
    if board.get("clock_by_lanes"):
        m["clock_by_lanes"] = sorted((float(a), float(b)) for a, b in board["clock_by_lanes"])
    _model = m


def clock_for(matrix_lanes, board_clock):
    """kernel clock in MHz for a matrix unit of this many lanes in all"""
    pts = _model["clock_by_lanes"]
    if not pts:
        return board_clock
    if matrix_lanes <= pts[0][0] or len(pts) == 1:
        return pts[0][1]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if matrix_lanes <= x1:
            return y0 + (y1 - y0) * (matrix_lanes - x0) / (x1 - x0)
    (x0, y0), (x1, y1) = pts[-2], pts[-1]
    return max(1.0, y1 + (y1 - y0) * (matrix_lanes - x1) / (x1 - x0))


def weight_cycles(lanes):
    """Estimated clock cycles per weight word for a port with this many lanes (fitted from measured runs)."""
    a, b = _model["weight_cycles"]
    return a + b / lanes


def pow2_floor(n, cap):
    """Largest power of two that is at most min(n, cap)."""
    p = 1
    while p * 2 <= min(n, cap):
        p *= 2
    return p


def lane_cap(cfg):
    """most lanes the matrix unit can have: <= 16 and dividing both the dimension and the hidden size (rows are read in whole words)"""
    return max(p for p in (1, 2, 4, 8, 16) if cfg["dim"] % p == 0 and cfg["hidden_dim"] % p == 0)


def lane_cap_att(cfg):
    """most lanes the attention unit can have: <= 16 and dividing the head size (a key or value slice is read in whole words)"""
    return max(p for p in (1, 2, 4, 8, 16) if layout.head_size(cfg) % p == 0)


def work_per_token(cfg, context):
    """-> (matrix weights read per token, attention multiplies per token at the given context length: scores and weighted values)"""
    d, h, L, kvd = cfg["dim"], cfg["hidden_dim"], cfg["n_layers"], layout.kv_dim(cfg)
    matrix = L * (2 * d * d + 2 * kvd * d + 3 * h * d) + cfg["vocab_size"] * d
    attention = 2 * L * cfg["n_heads"] * context * layout.head_size(cfg)       # the scores (query x keys) and the weighted sum of the values
    return matrix, attention


def serial_cycles(cfg, context):
    """cycles per token of the parts that work on one element at a time: vector loads, RMSNorm, RoPE, softmax, the gated activation and the
    residual additions (loop initiation intervals and depths from the HLS loop table)"""
    d, h, L, kvd, hs, heads = cfg["dim"], cfg["hidden_dim"], cfg["n_layers"], layout.kv_dim(cfg), layout.head_size(cfg), cfg["n_heads"]
    load = lambda n: n + 11 + AXI_FIRST                     # II 1, depth 11, first read
    ii = 1 if VEC_SUMS >= FADD_CYCLES else 3                 # the two sums run at II 3 with a single running total
    rmsnorm = lambda n: (ii * n + 9) + DIV_SQRT + (n + 9)    # sum of squares, then the scale at II 1
    rope = lambda n: n // 2 + 45
    softmax = lambda n: (n + 3) + (ii * n + 19) + (n + 12)   # maximum, exponentials and sum, divide
    add = lambda n: n + 7
    per_layer = (2 * load(d) + 2 * rmsnorm(d) + rope(d) + rope(kvd) + heads * (softmax(context) + hs) + 2 * add(d) + (h + 29))
    return L * per_layer + load(d) + 2 * load(hs // 2) + load(d) + rmsnorm(d) + d


def sum_cycles(lanes):
    """final sum of the running totals: a tree of lanes * ACC_DEPTH values, one add after another per level"""
    return FADD_CYCLES * math.ceil(math.log2(lanes * ACC_DEPTH)) + 2


def pct_up(x):
    """percentage rounded up to one decimal, so a printed percentage is always enough for the DSPs named next to it"""
    return math.ceil(x * 10 - 1e-9) / 10


def dsp_cost(lanes_mv, lanes_att):
    """Estimated DSP slices used by a kernel with these matrix and attention lane counts, including the safety margin."""
    base, per_mv, per_att = _model["DSP"]
    return _model["margin"] * (base + per_mv * lanes_mv + per_att * lanes_att)


def buffer_floats(limits):
    """floats in the kernel's on-chip vectors at these LLM_MAX_* limits that grow with the ports (the matrix unit keeps a copy per port):
    6 x dim, 2 x hidden, one context row (the attention scores of all heads are counted apart, in resources())"""
    return 6 * limits["LLM_MAX_DIM"] + 2 * limits["LLM_MAX_HIDDEN"] + limits["LLM_MAX_SEQ"]


def resources(ports, lanes, att, limits):
    """-> estimated {DSP, BRAM18, LUT, FF} of the kernel with these settings (see the constants above)"""
    m = ports * lanes
    axi = ports * math.ceil(1.8 * lanes) + math.ceil(1.8 * att) + 2
    arrays = BRAM_ARRAYS_BASE + BRAM_ARRAYS_PER_LANE * min(m, BRAM_ARRAY_LANE_CAP) + BRAM_ARRAYS_PER_ATT_LANE * min(att, BRAM_ARRAY_LANE_CAP)
    extra = max(0.0, buffer_floats(limits) - DEFAULT_BUFFER_FLOATS) / 576.0
    arrays += extra * (BRAM_PER_EXTRA_KFLOAT[0] + BRAM_PER_EXTRA_KFLOAT[1] * (ports - 1))
    arrays += math.ceil(limits["LLM_MAX_HEADS"] * limits["LLM_MAX_SEQ"] / SCORE_FLOATS_PER_BRAM18)   # the attention scores of every head, shared by the ports
    lb, lm, la = _model["LUT"]
    fb, fm, fa = _model["FF"]
    return {"DSP": dsp_cost(m, att), "BRAM18": int(math.ceil(_model["bram_factor"] * (axi + arrays))),
            "LUT": int(_model["margin"] * (lb + lm * m + la * att)),
            "FF": int(_model["margin"] * (fb + fm * m + fa * att))}


def board_resources(board):
    """-> {DSP, BRAM18, LUT, FF} the board has (only those its file gives)"""
    out = {}
    if board.get("dsp"):
        out["DSP"] = board["dsp"]
    if board.get("bram36"):
        out["BRAM18"] = 2 * board["bram36"]
    if board.get("lut"):
        out["LUT"] = board["lut"]
    if board.get("ff"):
        out["FF"] = board["ff"]
    return out


RESOURCE_NAMES = {"DSP": "the DSP percentage", "BRAM18": "the block RAM of the board", "LUT": "the LUTs of the board", "FF": "the flip-flops of the board"}


def fits(ports, lanes, att, limits, have, dsp_budget):
    """-> None if the settings fit, else the name of the first resource they exceed (DSP against the user's budget, the others against the
    board with RESOURCE_HEADROOM)"""
    use = resources(ports, lanes, att, limits)
    if math.ceil(use["DSP"]) > dsp_budget:
        return "DSP"
    for name in ("BRAM18", "LUT", "FF"):
        if name in have and use[name] > RESOURCE_HEADROOM * have[name]:
            return name
    return None


def mv_steps(cfg, max_ports):
    """the matrix unit's possible sizes as (ports, lanes per port), smallest first: 1, 2, 4, ... lanes on one port, then more ports of the
    widest port size"""
    cap = lane_cap(cfg)
    steps = [(1, p) for p in (1, 2, 4, 8, 16) if p <= cap]
    steps += [(n, cap) for n in range(2, min(max_ports, MAX_PORTS) + 1)]
    return steps


def split_lanes(fit, cfg, matrix, attention, context, steps, speed):
    """-> (index into steps, attention lanes, why it stopped). Start with the smallest unit of each kind and keep growing the one that is
    furthest below its share of the work. A step is only taken if it fits (`fit`: the DSP budget and the board's other resources) and makes the estimated speed (the slower of
    compute and memory) at least 0.5% better: past that point more lanes only add summing overhead, or the memory is the limit."""
    cap_att = lane_cap_att(cfg)
    k, att = 0, 1
    want = matrix / max(attention, 1)                    # matrix lanes per attention lane
    while True:
        here = speed(steps[k], att)
        options, blocked_by = [], []
        total = steps[k][0] * steps[k][1]
        for unit, nk, natt in (("mv", k + 1, att), ("att", k, att * 2)):
            if (unit == "mv" and nk >= len(steps)) or (unit == "att" and natt > cap_att):
                continue
            if speed(steps[nk], natt) < here * 1.005:
                continue
            why = fit(steps[nk][0], steps[nk][1], natt)
            if why:
                blocked_by.append(why)
            else:
                options.append((unit, nk, natt))
        if not options:
            return k, att, (blocked_by[0] if blocked_by else "no_gain")
        # a new weight port is the most expensive step: take it only when no cheaper step (more lanes on a port, more attention lanes) helps
        cheap = [o for o in options if not (o[0] == "mv" and steps[o[1]][0] > steps[k][0])]
        options = cheap or options
        behind = "mv" if total / att < want else "att"     # which unit has fewer lanes than its share of the work
        pick = next((o for o in options if o[0] == behind), options[0])
        k, att = pick[1], pick[2]


def cycles_per_token(cfg, mv, lanes_att, context):
    """rough cycle count of one token; mv = (weight ports, lanes per port)"""
    ports, lanes_mv = mv
    d, h, L, kvd, hs = cfg["dim"], cfg["hidden_dim"], cfg["n_layers"], layout.kv_dim(cfg), layout.head_size(cfg)

    def matrix_rows(rows, n, copies=True):
        # each port streams its own block of rows, one wide word per cycle, and the sum of one row overlaps the next row, so a row costs
        # its words; the input vector is copied in (n cycles) and the result copied out (rows cycles) except for the output head
        words = math.ceil(n / lanes_mv)
        return math.ceil(rows / ports) * words * weight_cycles(lanes_mv) + sum_cycles(lanes_mv) + CALL_OVERHEAD + ((n + rows) if copies else 0)

    per_layer = (matrix_rows(d, d) + 2 * matrix_rows(kvd, d) + matrix_rows(d, d)        # wq, wk, wv, wo
                 + 2 * matrix_rows(h, d) + matrix_rows(d, h))                            # w1, w3, w2
    # attention: per position, the keys and then the values of all heads are read in one burst each through the wide state port
    # (LOAD_II cycles per word, plus the DDR4 wait); then per head the score (lanes, plus the final sum) and the weighted sum (lanes)
    words = math.ceil(hs / lanes_att)
    row_words = math.ceil(kvd / lanes_att)
    passes = (cfg["n_heads"] * words + SCORE_PASS_DEPTH) + (cfg["n_heads"] * words + AV_PASS_DEPTH)
    attention = context * (2 * (row_words * LOAD_II + KV_READ_LATENCY) + passes)
    return L * (per_layer + attention) + matrix_rows(cfg["vocab_size"], d, copies=False) + serial_cycles(cfg, context)


def plan(cfg, o, board, dsp_percent, context=None, limits=None):
    """-> dict with the chosen settings and estimates; 'ok' is False (with 'problem') when the board file or percentage cannot be used"""
    use_board_model(board)
    try:
        return _plan(cfg, o, board, dsp_percent, context, limits)
    finally:
        use_board_model({})


def _plan(cfg, o, board, dsp_percent, context=None, limits=None):
    """Core of the planner: pick ports, lanes and buffer limits for one model within the DSP budget and check the other resources and the memory bandwidth. Returns a dict, with 'problem' set when nothing fits."""
    context = context or max(1, cfg["seq_len"] // 2)
    dsp = board.get("dsp")
    out = {"ok": False, "context": context}
    if not dsp:
        out["problem"] = "the board file has no 'dsp' (number of DSP slices); add it, e.g. \"dsp\": 1824"
        return out
    matrix, attention = work_per_token(cfg, context)
    min_dsp = int(math.ceil(dsp_cost(1, 1)))
    out.update(dsp_total=dsp, min_dsp=min_dsp, min_percent=pct_up(100.0 * min_dsp / dsp))
    budget = dsp * dsp_percent / 100.0
    if budget < min_dsp:
        out["problem"] = (f"{dsp_percent:g}% of the board's {dsp} DSPs is {budget:.0f}; the smallest working kernel (one lane in each unit) "
                          f"needs about {min_dsp}, which is {out['min_percent']:g}%. Choose at least {out['min_percent']:g}%.")
        return out
    clock = board.get("clock_mhz") or DEFAULT_CLOCK_MHZ
    bytes_token = o["total"] + 4 * 2 * cfg["n_layers"] * context * layout.kv_dim(cfg)    # whole weight file + K/V cache reads
    mem_limit = board["mem_gbps"] * 1e9 / bytes_token if board.get("mem_gbps") else None
    max_ports = int(board.get("max_ports") or 1)
    steps = mv_steps(cfg, max_ports)
    need = fit_check.needs(cfg)
    have = board_resources(board)
    smallest = resources(1, 1, 1, need)
    for name in ("BRAM18", "LUT", "FF"):
        if name in have and smallest[name] > RESOURCE_HEADROOM * have[name]:
            out["problem"] = (f"even the smallest kernel for this model needs about {smallest[name]} {name} and the board has {have[name]} "
                              f"({int(100 * RESOURCE_HEADROOM)}% usable); the buffer sizes the model needs ({need}) are too large for this board")
            return out
    fit = lambda ports, lanes, att: fits(ports, lanes, att, need, have, budget)

    def compute_speed(mv, att):
        return clock_for(mv[0] * mv[1], clock) * 1e6 / cycles_per_token(cfg, mv, att, context)

    def speed(mv, att):
        v = compute_speed(mv, att)
        return min(v, mem_limit) if mem_limit is not None else v

    k, att, stop = split_lanes(fit, cfg, matrix, attention, context, steps, speed)
    ports, lanes = steps[k]
    use = resources(ports, lanes, att, need)
    used = int(math.ceil(use["DSP"]))
    lim = limits or fit_check.kernel_limits()
    comp = compute_speed(steps[k], att)
    out.update(ok=True, resources=use, resources_available=have, ports=ports, lanes_mv=lanes, lanes_att=att, acc_depth=ACC_DEPTH, dsp_used=used, dsp_percent_used=pct_up(100.0 * used / dsp),
               max_lanes_mv=lane_cap(cfg), max_lanes_att=lane_cap_att(cfg), max_ports=min(max_ports, MAX_PORTS), clock_mhz=clock_for(ports * lanes, clock),
               cycles_per_token=cycles_per_token(cfg, steps[k], att, context), tokens_per_s=speed(steps[k], att),
               compute_tokens_per_s=comp, serial_share=serial_cycles(cfg, context) / cycles_per_token(cfg, steps[k], att, context), memory_limit_tokens_per_s=mem_limit, matrix_work=matrix, attention_work=attention,
               limits=need, defaults_exceeded=[x for x in need if need[x] > lim[x]], at_maximum=(stop == "no_gain"),
               port_gbps=lanes * 4 * clock_for(ports * lanes, clock) * 1e6 / 1e9, total_gbps=ports * lanes * 4 * clock_for(ports * lanes, clock) * 1e6 / 1e9)
    # what limits the speed
    out["stop"] = stop
    if mem_limit is not None and comp >= mem_limit * 0.995:
        out["bound"] = "memory bandwidth"
    elif stop in RESOURCE_NAMES:
        out["bound"] = RESOURCE_NAMES[stop]
    else:
        out["bound"] = "the kernel design (more lanes would not make it faster)"
    return out


def cflags(p):
    """The -D settings of a plan as one compiler-flag string, ready for the HLS config."""
    parts = [f"-D{k}={v}" for k, v in p["limits"].items()]
    parts += [f"-DLLM_PORTS={p['ports']}", f"-DLLM_LANES={p['lanes_mv']}", f"-DLLM_ACC_DEPTH={p['acc_depth']}",
              f"-DLLM_LANES_ATT={p['lanes_att']}", f"-DLLM_ACC_DEPTH_ATT={p['acc_depth']}", f"-DLLM_VEC_SUMS={VEC_SUMS}"]
    return " ".join(parts)


def report_lines(p, board_name="the board"):
    """Format a plan as printable lines (resources used against available, ports and lanes, expected speed)."""
    if not p["ok"]:
        return ["KERNEL PLAN", "  " + p["problem"]]
    out = ["KERNEL PLAN  (estimates from the HLS reports; not yet measured on a board)"]
    out.append(f"  DSP slices     {p['dsp_used']} of {p['dsp_total']} ({p['dsp_percent_used']:g}%)  (smallest working kernel: {p['min_dsp']}, {p['min_percent']:g}%)")
    use, have = p["resources"], p["resources_available"]
    others = ", ".join(f"{n} {use[n]:,}" + (f" of {have[n]:,} ({100 * use[n] / have[n]:.0f}%)" if n in have else "") for n in ("BRAM18", "LUT", "FF"))
    out.append(f"  other resources {others}   (estimates; kept under {int(100 * RESOURCE_HEADROOM)}% of the board)")
    total = p["ports"] * p["lanes_mv"]
    out.append(f"  matrix unit    {total} lanes = {p['ports']} weight port{'s' if p['ports'] > 1 else ''} x {p['lanes_mv']} lanes (most per port for this model: "
               f"{p['max_lanes_mv']}; ports the board allows: {p['max_ports']}), each port {32 * p['lanes_mv']} bits wide")
    out.append(f"                 reads {p['total_gbps']:.1f} GB/s of weights at {p['clock_mhz']:g} MHz")
    out.append(f"  attention unit {p['lanes_att']} lanes (most: {p['max_lanes_att']}), K/V cache port {32 * p['lanes_att']} bits wide   [work ratio matrix : attention = {p['matrix_work'] / max(p['attention_work'], 1):.0f} : 1 at context {p['context']}]")
    out.append(f"  buffer limits  " + " ".join(f"{k}={v}" for k, v in p["limits"].items()))
    out.append(f"  speed          about {p['tokens_per_s']:.2f} tokens/s at {p['clock_mhz']:g} MHz  (limited by {p['bound']})")
    out.append(f"                 {100 * p['serial_share']:.0f}% of the cycles are the one-element-at-a-time parts (norms, RoPE, softmax, activation), which more DSPs do not speed up")
    if p["memory_limit_tokens_per_s"] is not None:
        out.append(f"                 memory bandwidth alone would allow about {p['memory_limit_tokens_per_s']:.1f} tokens/s; the compute alone {p['compute_tokens_per_s']:.1f}")
    else:
        out.append("                 (add 'mem_gbps' to the board file to also compare with the memory bandwidth)")
    if p["bound"] == "memory bandwidth":
        out.append("  note           the memory is the limit: more lanes or ports would not make it faster")
    elif p["at_maximum"]:
        out.append("  note           more lanes would not make the estimate faster, so a higher percentage changes nothing")
    elif p["bound"] == "the DSP percentage":
        out.append("  note           a higher DSP percentage gives more lanes and more speed")
    return out


def main():
    """Command line: read the model header and board file, run the planner and write the build settings and a report to the output folder."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("flat", help="the .flat weight file (only its header is read)")
    ap.add_argument("--board", required=True, help="board JSON file (needs dsp; clock_mhz and mem_gbps are optional)")
    ap.add_argument("--dsp-percent", type=float, default=50.0, help="share of the board's DSP slices the kernel may use (default 50)")
    ap.add_argument("--context", type=int, help="typical number of tokens in context, for the attention share (default: half the model's context)")
    ap.add_argument("-o", "--out", default="kernel_build", help="folder for the build settings (default kernel_build)")
    a = ap.parse_args()
    if not 0 < a.dsp_percent <= 100:
        raise SystemExit("kernel_plan: --dsp-percent must be above 0 and at most 100")
    with open(a.flat, "rb") as f:
        cfg, o = layout.unpack_header(f.read(layout.HEADER_BYTES))
    from model_to_fpga_spec import load_board
    board = load_board(a.board)
    p = plan(cfg, o, board, a.dsp_percent, a.context)
    print("\n".join(report_lines(p, board["name"])))
    if not p["ok"]:
        sys.exit(1)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "kernel_defines.cfg").write_text(
        "# Add this line to the [hls] section of the HLS config (replace an existing syn.cflags line).\n"
        f"# Generated by kernel_plan.py for {Path(a.flat).name} on {board['name']} at {a.dsp_percent:g}% of the DSPs.\n"
        f"syn.cflags={cflags(p)}\n", encoding="utf-8")
    (out / "kernel_plan.json").write_text(json.dumps({k: v for k, v in p.items()}, indent=2), encoding="utf-8")
    print(f"\nwritten: {out / 'kernel_defines.cfg'} and {out / 'kernel_plan.json'}")


if __name__ == "__main__":
    main()
