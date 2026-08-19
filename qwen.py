#!/usr/bin/env python3
"""Qwen / Bailian usage traces -- a BLOCK dataset, 16 tokens per block.

https://github.com/alibaba-edu/qwen-bailian-usagetraces-anon

Unlike Mooncake, `hash_ids` here are PER-BLOCK hashes: the id identifies that
block's 16 tokens on their own, not the prefix leading up to it. Checked on
traceA: 52k ids appear at more than one position, and 245k co-occurrences of the
same id have different prefixes before them.

So we chain them ourselves -- prefix_id[k] mixes prefix_id[k-1] with block k --
which turns them into what the analysis needs: an id per prefix, equal exactly
when two requests share that prefix.

16 tokens per block is fine enough that fractional bucket scalings are real here.

    python qwen.py
"""
import json
import plots

MASK = (1 << 64) - 1
FILES = {"qwen-chat":     "traces/qwen_traceA_blksz_16.jsonl",   # to-C interactive chat
         "qwen-api":      "traces/qwen_traceB_blksz_16.jsonl",   # to-B API automation
         "qwen-thinking": "traces/qwen_thinking_blksz_16.jsonl",  # reasoning-heavy
         "qwen-coder":    "traces/qwen_coder_blksz_16.jsonl"}    # code generation
DIR = "traces"


def chain(block_ids):
    """Per-block ids -> per-prefix ids. splitmix64-style mixing, 64 bits."""
    out, h = [], 0
    for b in block_ids:
        h = (h * 0x9E3779B97F4A7C15 + b + 1) & MASK
        h ^= h >> 31
        h = (h * 0xBF58476D1CE4E5B9) & MASK
        out.append(h)
    return out


def load(path):
    """-> list of prefix-id lists, in arrival order (the file is time-sorted)."""
    return [chain(json.loads(l)["hash_ids"]) for l in open(path) if l.strip()]

UNIT=16 # tokens
BASE=1
NUM_HASHES=32

if __name__ == "__main__":
    for name, file in FILES.items():
        try:
            ids = load(file)
        except FileNotFoundError:
            print(f"skip {name}: {DIR}/{file} not downloaded")
            continue
        # base_block=1 -> 16-token blocks, bucket 1 reaches only 256 tokens, so
        # nearly every request buckets and the scheme is genuinely exercised.
        # base_block=32 -> 512-token blocks, i.e. Mooncake's granularity.
        plots.run(ids, name=f"{name}-{NUM_HASHES}x{UNIT * BASE}", unit=UNIT, unit_name="tokens",
                  # x1.5 needs base_block >= 4 (block sizes are whole units,
                  # so a 50% step from 1 unit is not representable). Use base 8
                  # -- 128-token blocks -- if you want a fractional scaling.
                  scalings=(2, 3, 4), n_hashes=NUM_HASHES, base_block=BASE,
                  thresholds=(0.4, 0.5, 0.6),
                  capacities=(100, 1_000, 10_000, 100_000, 1_000_000))
