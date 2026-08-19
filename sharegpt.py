#!/usr/bin/env python3
"""ShareGPT -- a TEXT dataset: prefix ids are built by hashing every UNIT bytes.

The unit is the finest block we can express, so keep it small: it is what makes
fractional bucket scalings (x1.5) mean anything.

Turn t of a conversation has everything so far as its prompt; that is where the
sharing comes from. ShareGPT has no timestamps, so conversations are interleaved
round-robin by turn (everyone's turn 1, then turn 2, ...), which keeps a turn
strictly after its predecessor.

    python sharegpt.py traces/ShareGPT_V3_unfiltered_cleaned_split.json
"""
import hashlib
import json
import sys
import plots

UNIT_BYTES = 256
MAX_CONVS = 20_000


def prefix_ids(prompt, unit=UNIT_BYTES):
    """One id per `unit` bytes; id k covers bytes 0..(k+1)*unit.
    One streaming pass: extend the hash, finalise a copy at each boundary."""
    h, out, pos = hashlib.blake2b(digest_size=8), [], 0
    while pos < len(prompt):
        end = min(pos + unit, len(prompt))
        h.update(prompt[pos:end]); pos = end
        out.append(h.copy().digest())
    return out


def load(path, max_convs=MAX_CONVS, unit=UNIT_BYTES):
    """-> list of prefix-id lists, in arrival order."""
    per_turn = []
    with open(path) as f:
      for c in json.load(f)[:max_convs]:
          text, t = "", 0
          for msg in c.get("conversations", []):
              text += msg.get("value", "") + "\n"
              if msg.get("from") in ("human", "user"):
                  while len(per_turn) <= t: per_turn.append([])
                  per_turn[t].append(text.encode("utf-8", "ignore"))
                  t += 1
    return [prefix_ids(p, unit) for turn in per_turn for p in turn]

FILE="traces/ShareGPT_V3_unfiltered_cleaned_split.json"

if __name__ == "__main__":
    plots.run(load(FILE), name=f"sharegpt-{UNIT_BYTES}", unit=UNIT_BYTES, unit_name="bytes",
              scalings=(2, 3, 4, 5), n_hashes=16,
              thresholds=(0.4, 0.5, 0.6),
              capacities=(10_000, 100_000, 1_000_000))
