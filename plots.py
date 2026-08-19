#!/usr/bin/env python3
"""
The analysis and the figure. Dataset-independent -- never edit this to add a dataset.

One entry point, called once per trace, producing one figure:

    plots.run(ids, name="conversation", unit="512 tokens", scalings=(2, 4))

`ids[i]` is request i's list of prefix ids: ids[i][k] identifies the prefix made
of the first (k+1) units of the prompt, so two requests share their first (k+1)
units exactly when those ids are equal, and len(ids[i]) is the prompt length in
units. Requests must be in arrival order.

Mooncake ships ids directly. Text datasets produce them by hashing every N bytes
-- see the loader in the dataset script.

Block sizes are whole numbers of units, so a fine unit is what makes fractional
bucket scalings (x1.5) meaningful. On a coarse unit they round away.

Everything is causal: a request only ever matches prefixes that EARLIER requests
put in the cache.
"""

from collections import Counter, OrderedDict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

# defaults -- override per dataset in run()
N_HASHES = 16                                         # pipeline stages on the switch
THRESHOLDS = (0.50, 0.75, 1.00)                       # match must cover this much
CAPACITIES = (100, 1000, 10_000, 100_000, 1_000_000)  # chains the switch holds
SCALINGS = (2,)                                       # each bucket is this much coarser
                                                      # (whole numbers only)
BASE_BLOCK = 1                                        # smallest block size, IN UNITS

# --------------------------------------------------------------------------
# BASE_BLOCK is the most important knob, and it is easy to misread, so:
#
#   Bucket 1 uses base_block; every bucket after it is `scaling` times coarser.
#   Bucket 1 therefore reaches  n_hashes * base_block  of prompt. Anything
#   shorter is hashed at full precision -- every block gets a hash. Anything
#   longer falls into a coarser bucket.
#
#   And a match is always rounded DOWN to a multiple of the block size, which is
#   never smaller than base_block. So base_block is also the floor on how finely
#   any match can be reported.
#
#   Large base  -> most prompts fit bucket 1, bucketing rarely engages,
#                  but small matches are invisible.
#   Small base  -> matches resolved finely, but almost every prompt buckets.
#
#   Concretely, with 16 hashes:
#
#     qwen (unit 16 tokens)
#       base  1  ->   16-token blocks, bucket 1 reaches    256 tokens
#       base 32  ->  512-token blocks, bucket 1 reaches  8,192 tokens
#                    (this is Mooncake's granularity -- use it to compare)
#
#     Mooncake (unit 512 tokens)
#       base  1  ->  512-token blocks, bucket 1 reaches  8,192 tokens
#                    it cannot go finer; 512 is the trace's floor
#
#   This is why Mooncake showed no difference between scalings: its base is 512
#   tokens and its median prompt is ~7,000, so most requests never leave bucket 1
#   and the bucketing never does anything.
# --------------------------------------------------------------------------

PALETTE = ["#2563EB", "#EA580C", "#0D9488", "#9333EA", "#B45309"]
GREY = "#71717a"


# --------------------------------------------------------------------------
# the measurement
# --------------------------------------------------------------------------

def block_size(prompt_units, scaling, n_hashes, base):
    """Smallest bucket whose n_hashes hashes span the whole prompt, in units.
    Buckets are base, base*scaling, base*scaling^2, ... with `scaling` a whole
    number, so every block size is a whole number of units and coverage is never
    over-promised. See BASE_BLOCK above."""
    b = base
    while n_hashes * b < prompt_units:
        b *= int(scaling)
    return b


def chains_for(ids, scaling, n_hashes, base):
    """Each request's chain of hashes, and its block size. Computed once.
    The block size is part of the key, so different buckets never match."""
    out = []
    for one in ids:
        b = block_size(len(one), scaling, n_hashes, base)
        out.append(([(b, one[b * j - 1]) for j in range(1, n_hashes + 1)
                     if b * j <= len(one)], b))
    return out


def switch_matches(chains, ids, capacity=None):
    """How much of each prompt the switch matches, as a fraction of the prompt.

    capacity = chains the switch holds. None = never evict (best case).
    A number = worst case: when full, drop the least-recently-used chain
    ENTIRELY, all its hashes at once."""
    resident, present, cache, out = OrderedDict(), Counter(), set(), []
    for i, (chain, b) in enumerate(chains):
        n = len(ids[i])
        match = 0
        for j, key in enumerate(chain, 1):
            here = key in cache if capacity is None else present[key] > 0
            if here: match = b * j
            else: break                          # hits are a prefix of the chain
        out.append(min(match, n) / n)
        if capacity is None:
            cache.update(chain)
        else:
            resident[i] = chain
            for k in chain: present[k] += 1
            while len(resident) > capacity:
                _, old = resident.popitem(last=False)
                for k in old:
                    present[k] -= 1
                    if present[k] == 0: del present[k]
    return out


def exact_matches(ids):
    """How much of each prompt an EARLIER request already computed, found exactly.
    No switch, no hashing scheme, no cache limit -- a property of the workload."""
    cache, out = set(), []
    for one in ids:
        match = 0
        for j, x in enumerate(one, 1):
            if x in cache: match = j
            else: break
        out.append(match / len(one))
        cache.update(one)
    return out


def share_over(fractions, threshold):
    return sum(f >= threshold for f in fractions) / len(fractions)


def _style(ax, xlabel, ylabel, title):
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontsize=10)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    ax.grid(True, lw=.5, color="#e4e4e7"); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)


# --------------------------------------------------------------------------
# panel 1 -- is there any sharing in this workload at all?
# --------------------------------------------------------------------------

def _draw_sharing(ax, exact, thresholds, bins=20):
    """Two views of the same numbers, on one axis because both are shares of
    requests: bars = how many requests have AROUND x already computed, line =
    how many have AT LEAST x, the cumulative tail of those bars.

    Read the line for the decision, the bars for the shape. The median needs no
    marker -- it is where the line crosses 50%."""
    xs = [i / 100 for i in range(101)]
    ax.hist(exact, bins=bins, range=(0, 1), color=PALETTE[0], alpha=.30,
            edgecolor="white", linewidth=.6, weights=[1 / len(exact)] * len(exact),
            label=f"around this much  (per {100//bins}% bin)")
    ax.plot(xs, [share_over(exact, x) for x in xs], lw=2.6, color=PALETTE[0],
            label="at least this much  (cumulative)")

    placed = []                                  # keep labels from stacking
    for t in thresholds:
        f = share_over(exact, t)
        if f < .02:
            continue
        ax.plot([t, t], [0, f], lw=1.0, ls=(0, (2, 2)), color=GREY)
        ax.plot([0, t], [f, f], lw=1.0, ls=(0, (2, 2)), color=GREY)
        dy = -11
        while any(abs(f + dy / 400 - q) < .035 for q in placed):
            dy -= 13
        placed.append(f + dy / 400)
        ax.annotate(f"{f*100:.0f}% have $\\geq${int(t*100)}%"
                    f"   ->  at most {1/(1-f):.1f}x throughput",
                    (0, f), xytext=(4, dy), textcoords="offset points",
                    fontsize=7.5, color="#3f3f46")

    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1))
    _style(ax, "fraction of the prompt seen in some earlier request",
           "share of requests",
           "Amount of sharing in the trace")
    ax.legend(frameon=False, fontsize=8, loc="upper right")


# --------------------------------------------------------------------------
# panel 2 -- how many requests can the switch route by itself?
# --------------------------------------------------------------------------

def _draw_switch(ax, ids, exact, scaling, n_hashes, base, thresholds, capacities):
    """x = chains the switch holds, y = share of requests it routes alone.
    Dotted = the ceiling for that threshold, i.e. the same question asked of a
    perfect matcher."""
    chains = chains_for(ids, scaling, n_hashes, base)
    print(f"   bucket scaling x{scaling:g} geometric")
    for t, colour in zip(thresholds, PALETTE):
        worst = [share_over(switch_matches(chains, ids, C), t) for C in capacities]
        ceil = share_over(exact, t)
        ax.plot(capacities, worst, lw=2.2, color=colour, marker="o", ms=4,
                label=f"switch, $\\geq${int(t*100)}%")
        ax.axhline(ceil, color=colour, lw=1.2, ls=(0, (2, 2)), alpha=.65)
        ax.annotate(f"ceiling {ceil*100:.0f}%", (capacities[0], ceil), xytext=(0, 4),
                    textcoords="offset points", fontsize=7, color=colour)
        gain = 1 / (1 - worst[-1]) if worst[-1] < 1 else float("inf")
        print(f"      >={int(t*100)}%  ceiling {ceil*100:5.1f}%   switch " +
              " ".join(f"C={c:,}:{w*100:.0f}%" for c, w in zip(capacities, worst))
              + f"   -> {gain:.1f}x throughput")
    ax.set_xscale("log"); ax.set_ylim(0, 1)
    _style(ax, "chains the switch holds", "requests the switch routes by itself",
           f"switch, bucket scaling x{scaling:g} geometric")


# --------------------------------------------------------------------------
# the one entry point -- one figure per trace
# --------------------------------------------------------------------------

def ladder(base, scaling, n_hashes, max_units):
    """The bucket ladder, in units: base, base*scaling, base*scaling^2, ...

    `scaling` must be a whole number. Block sizes are whole units -- there is no
    id for half a block -- so a fractional scaling would silently round to
    something else (at base 1, x1.5 becomes x2), and we would be measuring a
    ladder nobody asked for."""
    if scaling != int(scaling) or scaling < 2:
        raise ValueError(f"scaling must be a whole number >= 2, got {scaling}")
    rungs, b = [base], base
    while n_hashes * b < max_units:
        b *= int(scaling)
        rungs.append(b)
    return rungs


def run(ids, name, unit=1, unit_name="units", scalings=SCALINGS, n_hashes=N_HASHES,
        base_block=BASE_BLOCK, thresholds=THRESHOLDS, capacities=CAPACITIES,
        out=None):
    """Every plot for one trace, in one figure. `ids` in arrival order.
    `unit` is how many tokens/bytes one id covers; `unit_name` labels them."""
    out = out or f"fig_{name}.png"
    exact = exact_matches(ids)

    # sharing panel across the top, then the switch panels two per row
    nrow = 1 + (len(scalings) + 1) // 2
    height = 5.0 + 4.3 * (nrow - 1)
    fig = plt.figure(figsize=(10.8, height), dpi=200)
    gs = fig.add_gridspec(nrow, 2, height_ratios=[1.15] + [1] * (nrow - 1))

    max_units = max(len(i) for i in ids)
    print(f"\n{name}  ({len(ids):,} requests, unit {unit} {unit_name}, "
          f"longest prompt {max_units:,} units = {max_units*unit:,} {unit_name})")
    print("   workload  " + "  ".join(
        f">={int(t*100)}%:{share_over(exact,t)*100:.0f}%" for t in thresholds))
    for sc in scalings:
        rungs = ladder(base_block, sc, n_hashes, max_units)
        # A prompt lands in bucket k only because it did not fit bucket k-1, so
        # its length is between n_hashes*b(k-1) and n_hashes*b(k) while its block
        # is b(k). One block is therefore 1/n_hashes .. scaling/n_hashes of the
        # prompt. A match rounds down to a whole block, so the top of that range
        # is the worst-case rounding loss. (Bucket 1 is exempt -- it has no
        # predecessor, so a very short prompt can be one block on its own.)
        print(f"   bucket ladder, scaling x{sc:g} geometric -- one block is "
              f"{100/n_hashes:.1f}%..{100*sc/n_hashes:.1f}% of the prompt")
        for k, b in enumerate(rungs, 1):
            print(f"      bucket {k:>2}  block {b:>6,} units = {b*unit:>9,} {unit_name}"
                  f"   covers prompts to {n_hashes*b*unit:>10,} {unit_name}")

    _draw_sharing(fig.add_subplot(gs[0, :]), exact, thresholds)
    for i, sc in enumerate(scalings):
        row = 1 + i // 2
        ax = fig.add_subplot(gs[row, :] if len(scalings) == 1 else gs[row, i % 2])
        _draw_switch(ax, ids, exact, sc, n_hashes, base_block,
                     thresholds, capacities)
    fig.axes[1].legend(frameon=False, fontsize=8.5, loc="lower right")

    # bucket 1 is the same in every panel -- it is set by base_block alone --
    # so it belongs here, once, and not in each subplot title.
    fig.suptitle(f"{name} — {len(ids):,} requests, {n_hashes} hashes per request, "
                 f"trace_unit {unit} {unit_name}\n"
                 f"bucket 1: hash_block {base_block*unit:,} {unit_name} ({base_block}x trace_unit) , "
                 f"covers prompts to {n_hashes*base_block*unit:,} {unit_name}",
                 x=.005, ha="left", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 1 - .6 / height])
    fig.savefig(out, bbox_inches="tight")
    print("   wrote", out)
