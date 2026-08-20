#!/usr/bin/env python3
"""SWE-rebench OpenHands trajectories -- a TEXT dataset, hashed every UNIT bytes.

https://huggingface.co/datasets/nebius/SWE-rebench-openhands-trajectories

AGENTIC traffic, which is the regime where Mooncake's toolagent trace looked best
(73% at the 50% bar against 40% for chat). 67k trajectories, ~64 turns each,
prompts up to 131K tokens and averaging ~27K at the deepest turn. Each turn's
prompt is the whole message list so far -- system prompt, tool definitions, every
prior action and observation -- so the shared prefix grows enormous.

NO TIMESTAMPS. It is a benchmark trajectory set, not a serving trace, so arrival
order is synthesised the same way as ShareGPT: interleave trajectories
round-robin by turn (everyone's turn 1, then turn 2, ...), which at least keeps a
turn strictly after its predecessor. Say so in the paper.

    python swerebench.py          # MAX_TRAJ trajectories
    python swerebench.py 2000     # this many
"""
import hashlib
import sys

import pandas as pd

import plots

PATH = "traces/swe-rebench.parquet"
UNIT_BYTES = 64
MAX_TRAJ = 4_000                       # 64 turns x 27 KB each adds up fast
MAX_TURNS = 100


def prefix_ids(prompt, unit=UNIT_BYTES):
    """One id per `unit` bytes; id k covers bytes 0..(k+1)*unit.
    One streaming pass: extend the hash, finalise a copy at each boundary."""
    h, out, pos = hashlib.blake2b(digest_size=8), [], 0
    while pos < len(prompt):
        end = min(pos + unit, len(prompt))
        h.update(prompt[pos:end]); pos = end
        out.append(h.copy().digest())
    return out


def _messages(row):
    """The flat message list. Column name has moved before, so find it rather
    than hardcode: it is the one holding a list of role/content dicts."""
    for v in row:
        if isinstance(v, str) or not hasattr(v, "__len__") or len(v) == 0:
            continue
        if isinstance(v[0], dict) and "role" in v[0]:
            return v
    return []


def load(path=PATH, max_traj=MAX_TRAJ, unit=UNIT_BYTES):
    """-> list of prefix-id lists, in synthesised arrival order."""
    df = pd.read_parquet(path).head(max_traj)
    per_turn = []                       # per_turn[t] = every trajectory's turn t
    for row in df.itertuples(index=False):
        msgs = _messages(row)
        if len(msgs) == 0:      # numpy array -- "not msgs" is ambiguous
            continue
        text, t = "", 0
        for msg in msgs:
            text += (msg.get("content") or "") + "\n"
            if msg.get("role") != "user":
                continue
            if t >= MAX_TURNS:
                break
            while len(per_turn) <= t:
                per_turn.append([])
            per_turn[t].append(prefix_ids(text.encode(), unit))
            t += 1
    if not per_turn:
        sys.exit(f"found no message lists in {path}; columns are {list(df)}")
    print(f"   {len(df):,} trajectories, deepest turn {len(per_turn)}")
    return [ids for turn in per_turn for ids in turn]   # round-robin by turn


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else MAX_TRAJ
    ids = load(max_traj=n)
    # Prompts here are huge (tens of KB), so bucket 1 at base_block=1 reaches only
    # 1,024 bytes and almost everything buckets -- which is the point: this is the
    # one dataset where the ladder is exercised across many rungs.
    plots.run(ids, name="swe-rebench", unit=UNIT_BYTES, unit_name="bytes",
              scalings=(2, 3, 4), n_hashes=16, base_block=1,
              thresholds=(0.50, 0.75), coverages=(1.00, 0.75),
              capacities=(100, 1_000, 10_000, 100_000, 1_000_000))
