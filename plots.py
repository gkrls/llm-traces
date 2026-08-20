#!/usr/bin/env python3
"""
The analysis and the figure. Dataset-independent -- never edit this to add a dataset.
One entry point, called once per trace, producing one figure:
    plots.run(ids, name="conversation", unit=512, unit_name="tokens", scalings=(2, 4))
`ids[i]` is request i's list of prefix ids: ids[i][k] identifies the prefix made
of the first (k+1) units of the prompt, so two requests share their first (k+1)
units exactly when those ids are equal, and len(ids[i]) is the prompt length in
units. Requests must be in arrival order.
Mooncake ships ids directly. Text datasets produce them by hashing every N bytes
-- see the loader in the dataset script.
Block sizes are whole numbers of units, so the unit is the floor on how finely
any match can ever be reported. Bucket scalings must be whole numbers.
Everything is causal: a request only ever matches prefixes that EARLIER requests
put in the cache.
"""
from collections import Counter, OrderedDict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, PercentFormatter
# defaults -- override per dataset in run()
N_HASHES = 16                                         # pipeline stages on the switch
THRESHOLDS = (0.50, 0.75)                             # match must cover this much
CAPACITIES = (100, 1000, 10_000, 100_000, 1_000_000)  # chains the switch holds
SCALINGS = (2,)                                       # each bucket is this much coarser
                                                      # (whole numbers only)
BASE_BLOCK = 1                                        # smallest block size, IN UNITS
COVERAGES = (1.00,)                                   # how much of the prompt the
                                                      # hashes must span -- one
                                                      # figure per value
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
# COVERAGE -- how much of the prompt the n_hashes hashes are required to span.
#
#   A request goes up a bucket until  n_hashes * block >= coverage * prompt.
#   So a bucket accepts prompts up to  n_hashes * block / coverage.
#
#   coverage = 1.00 means the hashes must reach the end of the prompt. That is
#   what this code did before coverage existed, and it is the PESSIMISTIC end of
#   the design: it forces the largest block, hence the coarsest rounding.
#
#   Since  block ~= coverage * prompt / n_hashes,  lowering coverage shrinks the
#   block proportionally. One block is
#
#       coverage / n_hashes  ..  coverage * scaling / n_hashes    of the prompt
#
#   and since a match rounds DOWN to a whole block, the top of that range is the
#   worst-case rounding loss. With 16 hashes and x2 buckets:
#
#       coverage 1.00  ->  one block is  6.2% .. 12.5%  of the prompt
#       coverage 0.75  ->               4.7% ..  9.4%
#       coverage 0.50  ->               3.1% ..  6.2%
#
#   The cost is that you can never demonstrate a match larger than you hashed:
#
#       THRESHOLDS MUST ALL BE <= COVERAGE.  run() raises otherwise.
#
#   So coverage is chosen from the threshold you care about, not independently.
#   Set it just above your largest threshold and the blocks get as fine as the
#   design allows; set it to 1.00 and you are measuring your own worst case.
#
#   Rule of thumb: one figure per (coverage, threshold) pair you care about.
#   Mixing a 0.50 and a 0.75 threshold in one figure forces coverage >= 0.75,
#   which needlessly coarsens the 0.50 line.
#
#   Do NOT set coverage equal to your threshold. The deepest rung sits at
#   coverage * prompt, so at coverage 0.50 with a 0.50 bar every one of the
#   n_hashes hashes has to hit. Leave margin.
#
#   And lower coverage is not automatically better. Finer blocks mean less
#   rounding loss, but coverage also moves the bucket boundaries, so it
#   re-partitions the requests: two requests that shared a bucket at coverage
#   1.00 can land in different buckets at 0.50 and then cannot match at all.
#   Rounding loss against fragmentation loss. Which wins is a property of the
#   trace, which is why `coverages` is a list -- sweep it and look.
#
#   `run(coverages=(1.00, 0.75, 0.60))` produces one figure per value, each
#   labelled with its own coverage, written to fig_<name>_c100.png etc.
#   Thresholds above a given coverage are dropped from that figure with a
#   printed note, so a broad sweep does not blow up.
# --------------------------------------------------------------------------
PALETTE = ["#2563EB", "#EA580C", "#0D9488", "#9333EA", "#B45309"]
GREY = "#71717a"
LINESTYLES = ["-", (0, (5, 2)), (0, (1.6, 1.4)), (0, (7, 2, 1.5, 2))]  # per threshold
MARKERS    = ["o", "s", "^", "D"]              # per COVERAGE


def _shades(hexcolour, n):
    """n tints of one hue, darkest first. Threshold gets the HUE because its
    lines are far apart; coverage gets the tint because its lines cluster, and
    coverage is ordered so a light-to-dark ramp says the right thing."""
    r, g, b = (int(hexcolour[i:i + 2], 16) for i in (1, 3, 5))
    return [tuple((c + (255 - c) * (.5 * k / max(n - 1, 1))) / 255
                  for c in (r, g, b)) for k in range(n)]
# --------------------------------------------------------------------------
# the measurement
# --------------------------------------------------------------------------
def block_size(prompt_units, scaling, n_hashes, base, coverage=1.0):
    """Smallest bucket whose n_hashes hashes span `coverage` of the prompt, in
    units. Buckets are base, base*scaling, base*scaling^2, ... with `scaling` a
    whole number, so every block size is a whole number of units and coverage is
    never over-promised. See BASE_BLOCK and COVERAGE above."""
    b = base
    while n_hashes * b < coverage * prompt_units:
        b *= int(scaling)
    return b
def chains_for(ids, scaling, n_hashes, base, coverage=1.0):
    """Each request's chain of hashes, and its block size. Computed once.
    The block size is part of the key, so different buckets never match."""
    out = []
    for one in ids:
        b = block_size(len(one), scaling, n_hashes, base, coverage)
        out.append(([(b, one[b * j - 1]) for j in range(1, n_hashes + 1)
                     if b * j <= len(one)], b))
    return out
def switch_matches(chains, ids, capacity=None):
    """How much of each prompt the switch matches, as a fraction of the prompt.
    capacity = chains the switch holds. None = never evict (best case).
    A number = worst case: when full, drop the OLDEST chain ENTIRELY, all its
    hashes at once. Insertion order, never refreshed on a hit -- FIFO, not LRU,
    which is the pessimistic reading on purpose."""
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
        gain = f"at most {1/(1-f):.1f}x throughput" if f < 1 else "no router left"
        ax.annotate(f"{f*100:.0f}% have $\\geq${int(t*100)}%   ->  {gain}",
                    (0, f), xytext=(4, dy), textcoords="offset points",
                    fontsize=7.5, color="#3f3f46")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1))
    _style(ax, "fraction of the prompt seen in some earlier request",
           "share of requests",
           "Amount of sharing in the trace")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
# --------------------------------------------------------------------------
# panel 2 -- how long are the prompts, and where do the buckets cut?
# --------------------------------------------------------------------------
def _draw_sizes(ax, ids, unit, unit_name, n_hashes, base_block, bins=40):
    """Prompt length distribution. Log x, because lengths span decades.

    Bars on the LEFT axis, CCDF on the RIGHT, so the bars keep their own scale
    instead of being flattened against a 0-100% line.

    CCDF, not CDF: at any x it is the share of requests AT LEAST x long. Same
    sense as the "at least this much" line in the sharing panel below -- a rising
    CDF next to a falling tail made the same gesture mean opposite things two
    panels apart. It also answers the question this data gets asked: read it at
    bucket 1's reach for the share of prompts that have to bucket.

    Axis labels are coloured to match their mark; that is cheaper than a legend."""
    sizes = sorted(len(one) * unit for one in ids)
    n = len(sizes)
    lo, hi = max(sizes[0], 1), sizes[-1]
    edges = [lo * (hi / lo) ** (k / bins) for k in range(bins + 1)]
    ax.hist(sizes, bins=edges, color=PALETTE[2], alpha=.55, edgecolor="white",
            linewidth=.6, weights=[1 / n] * n)
    ax.set_xscale("log")
    _style(ax, f"prompt length ({unit_name}, log)", "share of requests",
           "Prompt sizes")
    ax.yaxis.label.set_color(PALETTE[2])

    cum = ax.twinx()
    cum.plot(sizes, [1 - i / n for i in range(n)], lw=2.2, color=GREY)
    cum.set_ylim(0, 1)          # x is shared; setting the scale here
                                # would reset the tick setup below
    cum.set_ylabel("at least this long (CCDF)", color=GREY)
    cum.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    cum.tick_params(axis="y", colors=GREY)
    cum.spines["top"].set_visible(False)
    cum.spines["right"].set_color(GREY)

    # percentile marks, so "most requests are this size or less" is a number on
    # the plot instead of something you squint off a log axis
    for q in (.50, .90):
        x = sizes[int(q * (n - 1))]
        cum.plot([x, x], [0, 1 - q], lw=1.0, ls=(0, (2, 2)), color=GREY)
        cum.annotate(f"p{int(q*100)} = {x:,}", (x, 1 - q), xytext=(4, 4),
                     textcoords="offset points", fontsize=7.5, color="#3f3f46")

    # 10^2 / 10^3 ticks are unreadable for this -- label 1-2-5 per decade with
    # plain numbers so a length can actually be read off the axis
    ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5), numticks=20))
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=range(10), numticks=99))
    ax.xaxis.set_minor_formatter(FuncFormatter(lambda *_: ""))
    ax.xaxis.set_major_formatter(FuncFormatter(
        lambda v, _: f"{v/1e6:g}M" if v >= 1e6 else
                     f"{v/1e3:g}k" if v >= 1e3 else f"{v:g}"))

    floor_at, reach_at = base_block * unit, n_hashes * base_block * unit
    print(f"      prompt {unit_name}: median {sizes[n//2]:,}, "
          f"p90 {sizes[int(.9*(n-1))]:,}, longest {hi:,}")
    print(f"      {sum(s < floor_at for s in sizes)/n*100:.1f}% shorter than one "
          f"block ({floor_at:,}) -> no hashes at all;  "
          f"{sum(s > reach_at for s in sizes)/n*100:.1f}% longer than bucket 1 "
          f"({reach_at:,}) -> coarser bucket")
# --------------------------------------------------------------------------
# panel 3 -- how many requests can the switch route by itself?
# --------------------------------------------------------------------------
def _draw_switch(ax, ids, exact, scaling, n_hashes, base, thresholds, capacities,
                 coverages):
    """x = chains the switch holds, y = share of requests it routes alone.
    COLOUR = coverage, LINE STYLE = threshold. Grey dotted = the ceiling for that
    threshold, i.e. the same question asked of a perfect matcher -- it does not
    depend on coverage, so there is one per threshold, not one per line."""
    print(f"   bucket scaling x{scaling:g} geometric")
    # the match fractions do not depend on the threshold, so simulate once per
    # (coverage, capacity) and test every threshold against the same numbers
    matched = {cov: [switch_matches(chains_for(ids, scaling, n_hashes, base, cov),
                                   ids, C) for C in capacities]
               for cov in coverages}
    for t, hue in zip(thresholds, PALETTE):
        tints = _shades(hue, len(coverages))
        for cov, tint, mark in zip(coverages, tints, MARKERS):
            if t > cov:                      # cannot match more than you hashed
                continue
            worst = [share_over(m, t) for m in matched[cov]]
            ax.plot(capacities, worst, lw=2.0, color=tint, marker=mark, ms=5,
                    label=f"$\\geq${int(t*100)}%,  cov {cov*100:.0f}%")
            gain = 1 / (1 - worst[-1]) if worst[-1] < 1 else float("inf")
            print(f"      >={int(t*100)}%  cov {cov*100:>3.0f}%   switch " +
                  " ".join(f"C={c:,}:{w*100:.0f}%"
                           for c, w in zip(capacities, worst))
                  + f"   -> {gain:.1f}x throughput")
    for t in thresholds:                     # ceilings: grey, coverage-free
        ceil = share_over(exact, t)
        ax.axhline(ceil, color=GREY, lw=1.1, ls=(0, (2, 2)), alpha=.8)
        ax.annotate(f"ceiling $\\geq${int(t*100)}%: {ceil*100:.0f}%",
                    (capacities[0], ceil), xytext=(0, 3),
                    textcoords="offset points", fontsize=7, color=GREY)
    ax.set_xscale("log"); ax.set_ylim(0, 1)
    _style(ax, "chains the switch holds", "requests the switch routes by itself",
           f"switch, bucket scaling x{scaling:g} geometric")


# --------------------------------------------------------------------------
# the one entry point -- one figure per trace
# --------------------------------------------------------------------------
def ladder(base, scaling, n_hashes, max_units, coverage=1.0):
    """The bucket ladder, in units: base, base*scaling, base*scaling^2, ...
    `scaling` must be a whole number. Block sizes are whole units -- there is no
    id for half a block -- so a fractional scaling would silently round to
    something else (at base 1, x1.5 becomes x2), and we would be measuring a
    ladder nobody asked for."""
    if scaling != int(scaling) or scaling < 2:
        raise ValueError(f"scaling must be a whole number >= 2, got {scaling}")
    rungs, b = [base], base
    while n_hashes * b < coverage * max_units:
        b *= int(scaling)
        rungs.append(b)
    return rungs
def run(ids, name, unit=1, unit_name="units", scalings=SCALINGS, n_hashes=N_HASHES,
        base_block=BASE_BLOCK, thresholds=THRESHOLDS, capacities=CAPACITIES,
        coverages=COVERAGES, out=None):
    """Every plot for one trace, in ONE figure. `ids` in arrival order. `unit` is
    how many tokens/bytes one id covers; `unit_name` labels them. `coverages` is
    how much of the prompt the hashes must span -- a list, drawn as one coloured
    line per value inside each panel. A single number is accepted too.
    See COVERAGE above."""
    if isinstance(coverages, (int, float)):
        coverages = (coverages,)
    for cov in coverages:
        if not 0 < cov <= 1:
            raise ValueError(f"coverage must be in (0, 1], got {cov}")
    if min(thresholds) > max(coverages):
        raise ValueError(
            f"every threshold is above every coverage (smallest threshold "
            f"{min(thresholds):.2f}, largest coverage {max(coverages):.2f}): "
            f"nothing can be drawn. You cannot match more than you hashed.")
    out = out or f"fig_{name}.png"
    exact = exact_matches(ids)
    # sharing panel across the top, then the switch panels two per row
    nrow = 2 + (len(scalings) + 1) // 2          # sharing, sizes, then switch
    height = 5.0 + 2.6 + 4.3 * (nrow - 2)
    fig = plt.figure(figsize=(10.8, height), dpi=200)
    gs = fig.add_gridspec(nrow, 2,
                          height_ratios=[.55, 1.15] + [1] * (nrow - 2))
    max_units = max(len(i) for i in ids)
    print(f"\n{name}  ({len(ids):,} requests, unit {unit} {unit_name}, "
          f"longest prompt {max_units:,} units = {max_units*unit:,} {unit_name})")
    print("   workload  " + "  ".join(
        f">={int(t*100)}%:{share_over(exact,t)*100:.0f}%" for t in thresholds))
    for cov in coverages:
        gone = tuple(t for t in thresholds if t > cov)
        if gone:
            print(f"   coverage {cov*100:.0f}%: no line for threshold "
                  + ", ".join(f"{t*100:.0f}%" for t in gone)
                  + " -- cannot match more of the prompt than you hashed")
        for sc in scalings:
            rungs = ladder(base_block, sc, n_hashes, max_units, cov)
            # A prompt lands in bucket k only because it did not fit bucket k-1,
            # so its length is between n_hashes*b(k-1)/coverage and
            # n_hashes*b(k)/coverage while its block is b(k). One block is
            # therefore coverage/n_hashes .. coverage*scaling/n_hashes of the
            # prompt. A match rounds down to a whole block, so the top of that
            # range is the worst-case rounding loss. (Bucket 1 is exempt -- it
            # has no predecessor, so a short prompt can be one block on its own.)
            print(f"   coverage {cov*100:.0f}%, scaling x{sc:g} geometric -- "
                  f"one block is {100*cov/n_hashes:.1f}%.."
                  f"{100*cov*sc/n_hashes:.1f}% of the prompt")
            for k, b in enumerate(rungs, 1):
                print(f"      bucket {k:>2}  block {b:>6,} units = "
                      f"{b*unit:>9,} {unit_name}   covers prompts to "
                      f"{int(n_hashes*b*unit/cov):>10,} {unit_name}")
    _draw_sizes(fig.add_subplot(gs[0, :]), ids, unit, unit_name, n_hashes,
                base_block)
    _draw_sharing(fig.add_subplot(gs[1, :]), exact, thresholds)
    first_switch = None
    for i, sc in enumerate(scalings):
        row = 2 + i // 2
        ax = fig.add_subplot(gs[row, :] if len(scalings) == 1 else gs[row, i % 2])
        first_switch = first_switch or ax
        _draw_switch(ax, ids, exact, sc, n_hashes, base_block,
                     thresholds, capacities, coverages)
    first_switch.legend(frameon=False, fontsize=8, loc="lower right",
                        handlelength=1.8, labelspacing=.35)
    # bucket 1 is the same in every panel -- it is set by base_block and coverage
    # alone -- so it belongs here, once, and not in each subplot title.
    reach = ", ".join(f"{c*100:.0f}%->{int(n_hashes*base_block*unit/c):,}"
                      for c in coverages)
    fig.suptitle(f"{name} — {len(ids):,} requests, {n_hashes} hashes per request, "
                 f"trace_unit {unit} {unit_name}\n"
                 f"bucket 1: hash_block {base_block*unit:,} {unit_name} "
                 f"({base_block}x trace_unit), covers prompts to "
                 f"[{reach}] {unit_name}",
                 x=.005, ha="left", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 1 - .6 / height])
    fig.savefig(out, bbox_inches="tight")
    print("   wrote", out)
