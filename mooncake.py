#!/usr/bin/env python3
"""Mooncake -- a BLOCK dataset: prefix ids ship with the trace, one per 512 tokens.

512 tokens is the finest block expressible here, so bucket scalings round to whole
multiples of it and fractional scalings are meaningless on this data.

    python mooncake.py
"""
import json
import plots

DIR = "traces"
FILES = {"mooncake_conversation": "traces/conversation_trace.jsonl",
         "mooncake_toolagent":    "traces/toolagent_trace.jsonl",
         "mooncake_synthetic":    "traces/synthetic_trace.jsonl"}


def load(path):
    """-> list of prefix-id lists, in arrival order."""
    return [json.loads(l)["hash_ids"] for l in open(path) if l.strip()]


if __name__ == "__main__":
    for name, path in FILES.items():
        plots.run(load(path), name=name, unit=512, unit_name="tokens",
                  scalings=(2, 3), n_hashes=16,
                  base_block=1,        # 1 unit = 512 tokens; the trace's floor,
                                       # so bucket 1 already reaches 8,192 tokens
                  thresholds=(0.50, 0.75, 1.00),
                  capacities=(100, 1_000, 10_000, 100_000, 1_000_000))
