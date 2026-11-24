#!/usr/bin/env python3
"""chat.py - a continuous chat with an instruct model running on the U50.

  python sw/chat.py --ssh USER@VM --template llama3 --tokenizer models/Llama-3.2-1B-Instruct/tokenizer.json \n      --xclbin '~/build/llm_top.xclbin' --host '~/llm-host/host_serve27' --split '~/run/split/llamainst27' --ports 27

Starts sw/host_serve.cpp on the U50 VM over ssh (it loads the weights into the HBM ONCE and then answers one request per line), keeps the
conversation here, applies the model's chat template, tokenizes on this PC (the VM has no tokenizer) and sends the card token numbers only.
Every turn sends the whole conversation, but the card keeps the K/V cache between turns and only computes the new tokens; the kernel
ends the reply at the model's end-of-turn token (--max-new is just the upper limit). When the conversation no longer fits in the context, the oldest turns are dropped.
Commands: /reset (forget the conversation), /system TEXT (set the system prompt and reset), /quit.
It never flashes or resets the card.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent.parent

# One entry per chat format. system/user/assistant wrap a message in the model's special tokens, `prompt` is the text that opens the
# assistant's turn (the model continues from there), and `stop` are the token ids that end a turn (the kernel stops on them too).
TEMPLATES = {
    # TinyLlama-1.1B-Chat-v1.0 (zephyr style); the tokenizer adds <s> (id 1) first; </s> (id 2) ends a turn
    "tinyllama": dict(
        system=lambda s: f"<|system|>\n{s}</s>\n",
        user=lambda s: f"<|user|>\n{s}</s>\n",
        assistant=lambda s: f"<|assistant|>\n{s}</s>\n",
        prompt="<|assistant|>\n",
        stop={2},
        default_system="You are a friendly assistant. Answer briefly and clearly.",
    ),
    # Llama 3 instruct; the tokenizer adds <|begin_of_text|> (128000); <|eot_id|> (128009) / <|end_of_text|> (128001) end a turn
    "llama3": dict(
        system=lambda s: f"<|start_header_id|>system<|end_header_id|>\n\n{s}<|eot_id|>",
        user=lambda s: f"<|start_header_id|>user<|end_header_id|>\n\n{s}<|eot_id|>",
        assistant=lambda s: f"<|start_header_id|>assistant<|end_header_id|>\n\n{s}<|eot_id|>",
        prompt="<|start_header_id|>assistant<|end_header_id|>\n\n",
        stop={128009, 128001},
        default_system="You are a helpful assistant. Answer briefly and clearly.",
    ),
}


def main():
    """Start the card-side server over ssh, then run the chat loop: read a message, apply the chat template, tokenize, send the ids, decode the reply."""
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--template", choices=sorted(TEMPLATES), default="tinyllama")
    ap.add_argument("--tokenizer", default=str(ROOT / "models" / "TinyLlama-1.1B-Chat" / "tokenizer.json"))
    ap.add_argument("--ssh", required=True, help="ssh host of the machine that has the card (user@host or an ssh config alias)")
    ap.add_argument("--key", default=None, help="ssh private key file (optional; default: your ssh config)")
    ap.add_argument("--xclbin", required=True, help="xclbin path on the VM")
    ap.add_argument("--split", required=True, help="prefix of the per-port weight files on the VM (sw/split_ports.py)")
    ap.add_argument("--ports", type=int, default=27)
    ap.add_argument("--host", required=True, help="host_serve binary on the VM, built from sw/host_serve.cpp with -DLLM_PORTS=<ports>")
    ap.add_argument("--max-new", type=int, default=96, help="tokens the kernel generates per reply (fixed; the reply is cut at the end-of-turn token)")
    ap.add_argument("--system", default=None)
    a = ap.parse_args()

    T = TEMPLATES[a.template]
    tok = Tokenizer.from_file(a.tokenizer)
    system = a.system or T["default_system"]

    # Start sw/host_serve.cpp on the card machine over ssh. It loads the weights into HBM once, then speaks a line protocol on stdin/stdout:
    #   in : GEN <max_new> <prompt token ids...>      out: TOKENS <generated ids...>  then  TIME <s> <positions computed> <pos/s> <start>
    #   after loading it prints READY <vocab> <context> <load seconds>.
    remote = ("source /opt/xilinx/xrt/setup.sh >/dev/null 2>&1; cd ~/run; "
              f"{a.host} --xclbin {a.xclbin} --split {a.split} --ports {a.ports} --stop {','.join(map(str, sorted(T['stop'])))}")
    srv = subprocess.Popen(["ssh"] + (["-i", a.key, "-o", "IdentitiesOnly=yes"] if a.key else []) + ["-o", "ServerAliveInterval=30", a.ssh, remote],
                           stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
    print("loading the model onto the card ...", flush=True)
    line = srv.stdout.readline()
    if not line.startswith("READY"):
        print("the server did not start:", line.strip(), srv.stderr.read()[:2000])
        return 1
    _, vocab, ctx, load_s = line.split()
    ctx = int(ctx)
    print(f"ready ({load_s} s to load). Type a message; /reset, /system TEXT, /quit.\n", flush=True)

    turns = []  # (role, text)

    def build(ts):
        # the whole conversation as one string: system prompt, every turn, then the opening of the assistant's reply
        return T["system"](system) + "".join(T[r](t) for r, t in ts) + T["prompt"]

    try:
        while True:
            try:
                msg = input("you> ").strip()
            except EOFError:
                break
            if not msg:
                continue
            if msg == "/quit":
                break
            if msg == "/reset":
                turns = []
                print("(conversation cleared)")
                continue
            if msg.startswith("/system"):
                system = msg[7:].strip() or T["default_system"]
                turns = []
                print("(system prompt set, conversation cleared)")
                continue
            turns.append(("user", msg))
            # Tokenize here (the card machine has no tokenizer): only token ids cross the wire.
            # drop the oldest turns until the conversation plus the reply fits in the context
            while True:
                ids = tok.encode(build(turns)).ids
                if len(ids) + a.max_new <= ctx or len(turns) <= 1:
                    break
                turns = turns[2:] if len(turns) > 2 else turns[1:]
            if len(ids) + a.max_new > ctx:
                print("(message too long for the context)")
                turns.pop()
                continue
            t0 = time.time()
            # Every turn sends the whole conversation; the card keeps the K/V cache of the earlier tokens and only computes the new ones
            # (the TIME line's last field is the position it resumed from).
            srv.stdin.write(f"GEN {a.max_new} " + " ".join(map(str, ids)) + "\n")
            srv.stdin.flush()
            resp = srv.stdout.readline()
            if not resp.startswith("TOKENS"):
                print("card error:", resp.strip())
                turns.pop()
                continue
            out = [int(x) for x in resp.split()[1:]]
            timeline = srv.stdout.readline().split()
            # the reply ends at the first end-of-turn token; anything after it is not shown or kept in the history
            cut = next((i for i, t in enumerate(out) if t in T["stop"]), len(out))
            reply = tok.decode(out[:cut], skip_special_tokens=True).strip()
            turns.append(("assistant", reply))
            used = len(ids) + cut
            print(f"bot> {reply}")
            resumed = f", resumed at position {timeline[4]} ({timeline[2]} computed)" if len(timeline) > 4 else ""
            print(f"     [{len(ids)} prompt + {cut} reply tokens, {time.time() - t0:.1f} s, card {timeline[3] if len(timeline) > 3 else '?'} pos/s{resumed}]\n")
    finally:
        try:
            srv.stdin.write("QUIT\n")
            srv.stdin.flush()
        except Exception:
            pass
        srv.terminate()
    return 0


if __name__ == "__main__":
    sys.exit(main())
