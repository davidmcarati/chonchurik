"""Does a fly predict direction at all? The measurement that decides the rest.

Selection on money has one term for skill and one for cost, and on this market
the cost term is much the louder: a round trip pays 1.3% (0.6% fee and 0.05%
spread a side), so at chance-level direction every trade loses, and a search
graded on money ranks genomes by how little they trade. The first real
evolution did exactly that -- fitness correlated -0.97 with the number of
sells, and the champion was buy-and-hold with the change.

This asks the question underneath that one, with no money in it at all: does
the proposal at an observation carry information about where price goes next?

    python -m tools.ic --granularity ONE_HOUR --observations 100 --starts 5

Reads `runs/<run>/population.json` when given one, so an evolved genome can be
compared against the wild type it came from.

What is reported
----------------
For each genome and each horizon `h`, with `s` the proposal (BUY +1, SELL -1,
HOLD 0) and `r` the forward return over the next `h` bars:

    edge = mean(s * (r - mean r))     in basis points, per proposal
    ic   = edge / (std(r) * rms(s))   the same number as a correlation

The returns are demeaned inside each start. That is the whole point: a fly
that proposes BUY at every observation scores exactly zero however hard the
market rose, because a constant signal cannot correlate with a deviation. Only
tracking the ups and downs scores. Drift is not skill and is not counted as
skill here.

The null
--------
`ic` on a hundred observations with an eleven-bar horizon has perhaps nine
independent samples in it, so a number alone means nothing. The null is the
same proposal sequence *circularly shifted* against the same prices, drawn
many times. A shift keeps the signal exactly as autocorrelated as it really is
-- flies propose in runs, and a plain shuffle would break those runs and give
a null far too narrow to fail against. It destroys only the alignment, which
is the thing being tested.

`p` is two-sided over those draws. With six genomes and five horizons there
are thirty tests, so read `p` against that: one at 0.03 is what a null looks
like.
"""

import argparse
import json
import random
from pathlib import Path

from stonkfly.config import Settings
from stonkfly.genome import WILD_TYPE
from tools.evolve.evaluate import CHART_WINDOW, WARMUP, build
from tools.evolve.herd import Herd, replay_herd
from tools.evolve.information import (SIGN, aligned, coefficient, edge,
                                      forward, normaliser, readout_ic,
                                      shifted, test)
from tools.evolve.series import candle_series, klines, split, starts

# The first scored observation sits this far past the fly's start.
BASE = CHART_WINDOW + WARMUP

# The statistic itself lives with the search now, in
# `tools/evolve/information.py`, because selection uses it and a number that
# decides what breeds cannot live in a tool that imports the thing it grades.
# The names above are re-exported so this stays the place to read about them.
HORIZONS = (1, 3, 6, 11, 24)
DRAWS = 2000


def genomes_from(path, ranks):
    """Named genomes out of an evolution's saved state."""
    state = json.loads(path.read_text(encoding="utf-8"))
    survivors = state["survivors"]
    out = []
    for rank in ranks:
        if rank >= len(survivors):
            raise ValueError(f"{path} has {len(survivors)} survivors, "
                             f"no rank {rank}")
        row = survivors[rank]
        out.append((f"rank {rank} (fit {row['fitness']:+.4f})", row["genome"]))
    return out


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--candles", type=Path, default=Path("data/candles.json"))
    p.add_argument("--product", default="BTC-USDC")
    p.add_argument("--bars", action="store_true",
                   help="read whole klines from --candles and open the three "
                        "whole-bar odour channels. Without it the fly smells "
                        "the close-only descriptors it always did, which is "
                        "what every recorded diagnostic measured")
    p.add_argument("--granularity", default="ONE_HOUR")
    p.add_argument("--segment", default="train",
                   choices=["train", "validation", "test"])
    p.add_argument("--observations", type=int, default=100)
    p.add_argument("--starts", type=int, default=5)
    p.add_argument("--population", type=Path,
                   default=Path("runs/evolution-5min/population.json"))
    p.add_argument("--ranks", default="0,1,2,14,27")
    p.add_argument("--horizons", default=",".join(str(h) for h in HORIZONS))
    p.add_argument("--draws", type=int, default=DRAWS)
    p.add_argument("--seed", type=int, default=20260915)
    p.add_argument("--out", type=Path)
    a = p.parse_args()

    horizons = [int(h) for h in a.horizons.split(",") if h]
    whole = candle_series(a.candles, a.product, a.granularity)
    sizes = {k: len(v) for k, v in split(whole).items()}
    lo = {"train": 0, "validation": sizes["train"],
          "test": sizes["train"] + sizes["validation"]}
    cut = slice(lo[a.segment], lo[a.segment] + sizes[a.segment])
    prices = whole[cut]
    # Sliced with the same offsets as the closes, which `klines` guarantees
    # are index-aligned with it.
    bars = klines(a.candles, a.product, a.granularity)[cut] if a.bars else None
    offsets = starts(prices, a.starts, CHART_WINDOW, a.observations + WARMUP)

    named = [("wild type", dict(WILD_TYPE))]
    if a.population and a.population.exists() and a.ranks:
        named += genomes_from(a.population,
                              [int(r) for r in a.ranks.split(",") if r])

    settings = Settings()
    controller, pristine = build(settings)
    tasks = [(i, start) for i in range(len(named)) for start in offsets]

    print(f"{a.product} {a.granularity} {a.segment}: {len(prices):,} candles, "
          f"{len(named)} genomes x {len(offsets)} starts x {a.observations} "
          f"observations, whole-bar channels "
          f"{'open' if a.bars else 'resting'}")
    print(f"starts {offsets}\n")

    herd = Herd(controller, pristine, [named[i][1] for i, _ in tasks],
                settings, [s for _, s in tasks])
    try:
        rows = replay_herd(herd, prices, a.observations, record=True,
                           bars=bars)
    finally:
        del herd
        import cupy as cp

        cp.get_default_memory_pool().free_all_blocks()

    report = []
    for g, (name, _) in enumerate(named):
        mine = [(row, start) for (i, start), row in zip(tasks, rows) if i == g]
        buy = sum(r["buy"] for r, _ in mine)
        sell = sum(r["sell"] for r, _ in mine)
        hold = sum(r["hold"] for r, _ in mine)
        total = buy + sell + hold
        profit = sum(r["profit"] for r, _ in mine) / len(mine)
        print(name)
        print(f"  proposals  BUY {buy / total:5.1%}  SELL {sell / total:5.1%}  "
              f"HOLD {hold / total:5.1%}   mean profit {profit:+.3f}")
        open_gate = sum(sum(1 for x in r["gate_spikes"] if x) for r, _ in mine)
        print(f"  gate open  {open_gate / total:5.1%} of observations")
        print(f"  {'horizon':>8}{'edge bps':>11}{'ic':>9}{'z':>8}{'p':>8}"
              f"{'n':>7}{'readout ic':>12}{'z':>8}{'p':>8}")
        for h in horizons:
            runs = [aligned(prices, r["sides"], s + BASE, h)
                    for r, s in mine]
            runs = [r for r in runs if r[1]]
            if not runs:
                continue
            rng = random.Random(a.seed + 1000 * g + h)
            t = test(runs, rng, a.draws)
            # The same test on the number the threshold was applied to. If the
            # proposals carry nothing but this does, the loss is in the
            # decoder and is repairable; if neither does, it is upstream.
            raw = [aligned(prices, r["difference_hz"], s + BASE, h,
                           centre=True) for r, s in mine]
            raw = [r for r in raw if r[1]]
            u = test(raw, random.Random(a.seed + 7 + 1000 * g + h), a.draws)
            print(f"  {h:>8}{t['edge_bps']:>11.2f}{t['ic']:>9.4f}"
                  f"{t['z']:>8.2f}{t['p']:>8.4f}{t['n']:>7}"
                  f"{u['ic']:>12.4f}{u['z']:>8.2f}{u['p']:>8.4f}")
            report.append({"genome": name, "horizon": h, **t,
                           "readout": u})
        print()

    if report:
        for label, pick in [("proposals", lambda r: abs(r["z"])),
                            ("readout", lambda r: abs(r["readout"]["z"]))]:
            best = max(report, key=pick)
            row = best if label == "proposals" else best["readout"]
            print(f"largest deviation from the null, {label}: "
                  f"{best['genome']} at horizon {best['horizon']}, "
                  f"z {row['z']:+.2f}, p {row['p']:.4f}")
        hits = sum(1 for r in report if r["p"] < 0.05)
        raw_hits = sum(1 for r in report if r["readout"]["p"] < 0.05)
        print(f"{len(report)} tests each: {hits} proposals and {raw_hits} "
              f"readouts under p 0.05, {0.05 * len(report):.1f} expected "
              f"if nothing is there")
    if a.out:
        a.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
        print(f"written to {a.out}")


if __name__ == "__main__":
    main()
