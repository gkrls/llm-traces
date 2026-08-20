#!/usr/bin/env python3
"""WildChat-1M -- a TEXT dataset: prefix ids are built by hashing every UNIT bytes.

https://huggingface.co/datasets/allenai/WildChat-1M

Real ChatGPT traffic, 838k conversations, full text, and -- unlike ShareGPT --
REAL timestamps, so arrival order is not synthesised. Assistant messages carry a
per-turn timestamp; we use the assistant timestamp of turn t as the arrival time
of the user request that produced it, then sort every request globally by time.

Turn t of a conversation has everything so far as its prompt; that is where the
sharing comes from.

The dataset also ships hashed_ip / state / country / request headers. We never
read those columns.

    python wildchat.py            # every shard in traces/
    python wildchat.py 3          # first 3 shards only
"""
import glob
import hashlib
import sys

import pandas as pd

import plots

DIR = "traces"
PATTERN = "wildchat-*.parquet"
UNIT_BYTES = 64
MAX_CONVS = 120_000                     # per shard; whole dataset needs ~50 GB RAM


def prefix_ids(prompt, unit=UNIT_BYTES):
    """One id per `unit` bytes; id k covers bytes 0..(k+1)*unit.
    One streaming pass: extend the hash, finalise a copy at each boundary."""
    h, out, pos = hashlib.blake2b(digest_size=8), [], 0
    while pos < len(prompt):
        end = min(pos + unit, len(prompt))
        h.update(prompt[pos:end]); pos = end
        out.append(h.copy().digest())
    return out


def requests_in(path, max_convs=MAX_CONVS, unit=UNIT_BYTES):
    """-> list of (arrival_time, prefix_ids) for one parquet shard."""
    df = pd.read_parquet(path, columns=["timestamp", "conversation"])
    out = []
    for end_time, convo in df.head(max_convs).itertuples(index=False):
        text = ""
        for msg in convo:
            text += (msg.get("content") or "") + "\n"
            if msg.get("role") != "user":
                continue
            # arrival = when the assistant answered this turn, if we have it
            out.append((msg.get("timestamp") or end_time,
                        prefix_ids(text.encode(), unit)))
    return out


def load(paths, unit=UNIT_BYTES):
    """-> list of prefix-id lists, in real arrival order."""
    reqs = []
    for p in paths:
        reqs += requests_in(p, unit=unit)
        print(f"   read {p}  ({len(reqs):,} requests so far)")
    reqs.sort(key=lambda r: r[0])
    return [ids for _, ids in reqs]


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    paths = sorted(glob.glob(f"{DIR}/{PATTERN}"))[:n or None]
    if not paths:
        sys.exit(f"no {DIR}/{PATTERN} -- run ./download.sh first")
    ids = load(paths)
    # 64-byte blocks, base_block=1 -> bucket 1 reaches 1,024 bytes. Raw bytes so
    # there is no tokenizer floor; this is the finest any of our datasets gets.
    plots.run(ids, name="wildchat", unit=UNIT_BYTES, unit_name="bytes",
              scalings=(2, 3, 4), n_hashes=16, base_block=1,
              thresholds=(0.50, 0.75), coverages=(1.00, 0.75),
              capacities=(1_000, 10_000, 100_000, 1_000_000))
