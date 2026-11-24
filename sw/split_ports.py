#!/usr/bin/env python3
"""split_ports.py - cut a flat weight file into one file per weight port (the per-port layout).

  python sw/split_ports.py models/Llama-3.2-1B.flat --ports 8 --chunk 8192 --out run/llama

writes run/llama.p0 ... run/llama.p7. Port p of the kernel is given file p, each in its own part of the HBM, so no two ports share a channel
(the shared single copy costs about 1.9x in speed, see CLAUDE.md). The kernel's matrix unit gives port p a block of rows of every matrix
(rows p*per_port ... of each matrix, per_port = ceil(rows / ports)); file p holds exactly those rows and nothing else, so the files together
are as big as the original (plus one copy of the embedding table in file 0, which the kernel reads one row of per token).

  file 0 : the original 256-byte header (words 27..36 added, see below), the embedding table, the norm vectors (2 per layer), the final
           norm, the rope cos / sin tables, then its own block (every matrix's rows, layer after layer, then the output head)
  file p : its own block only

The block of a port: for each layer, wq wk wv wo w1 w2 w3 (each: that port's rows, packed, rounded up to 16 floats), then the output head
in chunks of --chunk vocabulary rows (the kernel does the head chunk by chunk, MAX_COLS = max(LLM_MAX_DIM, LLM_MAX_HIDDEN) rows at a time:
use the same number as the kernel build, it checks it). The head is the embedding table when the model shares it with the input.

Header words added (float slots of the 256-byte header; all offsets in floats, from the start of the file):
  27 'SPLT'   28 chunk rows   29 ports   30 start of port 0's own block   31 embedding table   32 norm vectors   33 final norm
  34 rope cos   35 rope sin   36 floats per layer in a port's block
The kernel computes the same positions and refuses (-5 / -9) a file that does not agree.
"""
import argparse
import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ref"))
import layout  # noqa: E402

A = 16  # floats in 64 bytes


def al(n):
    """round a float count up to a 64-byte boundary (every section starts aligned, as the kernel's wide reads need)"""
    return (n + A - 1) // A * A


def cdiv(a, b):
    """ceiling division: rows per port when `a` rows are shared by `b` ports"""
    return (a + b - 1) // b


def plan(c, o, ports, chunk):
    """-> dict of offsets (floats) of everything in the split files"""
    d, h, kvd, v, seq, nl = c["dim"], c["hidden_dim"], layout.kv_dim(c), c["vocab_size"], c["seq_len"], c["n_layers"]
    hs = layout.head_size(c)
    p = {}
    # port 0's shared part
    p["tok"] = layout.HEADER_BYTES // 4
    p["norms"] = al(p["tok"] + v * d)
    p["fin"] = al(p["norms"] + nl * 2 * d)
    p["cos"] = al(p["fin"] + d)
    p["sin"] = al(p["cos"] + seq * (hs // 2))
    p["pb0"] = al(p["sin"] + seq * (hs // 2))
    # a port's block: for each layer its rows of wq wk wv wo w1 w2 w3, back to back; then the output head in chunks of `chunk` rows
    mats = [("wq", d, d), ("wk", kvd, d), ("wv", kvd, d), ("wo", d, d), ("w1", h, d), ("w2", d, h), ("w3", h, d)]  # (name, rows, columns)
    pos, prel = 0, {}
    for name, rows, cols in mats:
        prel[name] = pos
        pos = al(pos + cdiv(rows, ports) * cols)
    p["prel"], p["mats"], p["lstride"] = prel, mats, pos
    p["chunkstride"] = al(cdiv(chunk, ports) * d)
    p["nchunks"] = cdiv(v, chunk)
    p["block"] = nl * pos + p["nchunks"] * p["chunkstride"]
    return p


def split(src, ports, chunk, out):
    """write OUT.p0 .. OUT.p<ports-1> from the flat file `src`; returns [(file name, bytes)]"""
    with open(src, "rb") as f:
        hdr = f.read(layout.HEADER_BYTES)
    c, o = layout.unpack_header(hdr)
    d, kvd, v, nl = c["dim"], layout.kv_dim(c), c["vocab_size"], c["n_layers"]
    if not 1 <= ports <= 27:
        sys.exit("ports must be 1 to 27")
    P = plan(c, o, ports, chunk)
    w = np.memmap(src, dtype="<f4", mode="r")
    stride = o["layer_stride"] // 4

    def region(byte_off, n):
        # n floats of the source file starting at a byte offset
        return w[byte_off // 4: byte_off // 4 + n]

    # the output head is the embedding table when the model shares it, otherwise the separate classifier matrix
    head_base = (o["tok_emb"] if c["shared"] else o["wcls"]) // 4
    outs = []
    for pt in range(ports):
        total = P["block"] + (P["pb0"] if pt == 0 else 0)
        buf = np.zeros(total, dtype="<f4")
        base = P["pb0"] if pt == 0 else 0
        # this port's rows of every matrix of every layer: rows [pt*pp, (pt+1)*pp) of each (the same split the kernel's rows_of() uses)
        for l in range(nl):
            lb = o["layers_start"] // 4 + l * stride
            for name, rows, cols in P["mats"]:
                rel = o["layer_rel"][name] // 4
                pp = cdiv(rows, ports)
                r0, r1 = pt * pp, min(rows, (pt + 1) * pp)
                if r1 > r0:
                    dst = base + l * P["lstride"] + P["prel"][name]
                    buf[dst: dst + (r1 - r0) * cols] = w[lb + rel + r0 * cols: lb + rel + r1 * cols]
        hb = base + nl * P["lstride"]
        # the output head, chunk by chunk, split by rows the same way
        for k in range(P["nchunks"]):
            v0 = k * chunk
            n = min(chunk, v - v0)
            pp = cdiv(n, ports)
            r0, r1 = pt * pp, min(n, (pt + 1) * pp)
            if r1 > r0:
                dst = hb + k * P["chunkstride"]
                buf[dst: dst + (r1 - r0) * d] = w[head_base + (v0 + r0) * d: head_base + (v0 + r1) * d]
        if pt == 0:
            # port 0 also carries the shared part: the header (with the split layout recorded in words 27..36 so the kernel can check it),
            # the embedding table, the norm vectors, the final norm and the RoPE tables
            h32 = np.frombuffer(bytearray(hdr), dtype="<u4").copy()
            vals = {27: 0x53504C54, 28: chunk, 29: ports, 30: P["pb0"], 31: P["tok"], 32: P["norms"], 33: P["fin"], 34: P["cos"],
                    35: P["sin"], 36: P["lstride"]}
            for i, x in vals.items():
                assert x < 2 ** 32
                h32[i] = x
            buf[:64] = h32.view("<f4")
            buf[P["tok"]: P["tok"] + v * d] = region(o["tok_emb"], v * d)
            for l in range(nl):
                lb = o["layers_start"] + l * o["layer_stride"]
                buf[P["norms"] + l * 2 * d: P["norms"] + l * 2 * d + d] = region(lb + o["layer_rel"]["rms_att"], d)
                buf[P["norms"] + l * 2 * d + d: P["norms"] + (l + 1) * 2 * d] = region(lb + o["layer_rel"]["rms_ffn"], d)
            buf[P["fin"]: P["fin"] + d] = region(o["final_norm"], d)
            n_rope = c["seq_len"] * (layout.head_size(c) // 2)
            buf[P["cos"]: P["cos"] + n_rope] = region(o["rope_cos"], n_rope)
            buf[P["sin"]: P["sin"] + n_rope] = region(o["rope_sin"], n_rope)
        fn = f"{out}.p{pt}"
        with open(fn, "wb") as fh:
            buf.tofile(fh)
        outs.append((fn, buf.nbytes))
        print(f"{fn}: {buf.nbytes / 2**20:.1f} MiB")
        del buf
    return outs


def main():
    """Command line wrapper around split()."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("flat")
    ap.add_argument("--ports", type=int, default=8)
    ap.add_argument("--chunk", type=int, required=True, help="vocabulary rows per head chunk = the kernel's MAX_COLS = max(LLM_MAX_DIM, LLM_MAX_HIDDEN)")
    ap.add_argument("--out", required=True, help="output prefix; files are OUT.p0 ... OUT.p<ports-1>")
    a = ap.parse_args()
    split(a.flat, a.ports, a.chunk, a.out)


if __name__ == "__main__":
    main()
