"""Is there anything in the fly's input to predict with? No brain required.

`tools/ic.py` measured the output and found nothing: neither the proposals nor
the continuous readout behind them carry information about where price goes
next, on train or on validation. That leaves two very different diagnoses --
the brain destroys a signal that was there, or there was no signal in what it
was fed.

This settles that from the other end. It takes the five descriptors
`stonkfly.neural.olfaction.features` actually computes, before any neuron sees
them, and asks each one directly whether it predicts the forward return. No
connectome, no GPU, a second to run.

    python -m tools.features --granularity ONE_HOUR --segment train

A feature with no edge here cannot acquire one by being turned into an odour.
If every feature is flat, the repair is a better description of the market,
not a better nose.

Method is `tools/ic.py`'s, so the numbers are comparable: Pearson correlation
against the forward return demeaned over the segment, and a null built by
circularly shifting the feature against the same prices, which keeps the
feature exactly as autocorrelated as it really is. `combined` is the best a
linear reader of all five could do, fitted on four chronological fifths and
scored on the fifth it did not see, pooled -- the ceiling for this input under
any readout that adds them up.
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np

from stonkfly.neural.olfaction import FEATURES, RANGE, features
from tools.ic import HORIZONS, test
from tools.evolve.series import candle_series, klines, split
from tools import microstructure

FOLDS = 5


def table(prices):
    """The five descriptors at every bar they are defined at.

    `features` reads at most `RANGE` returns back, so handing it exactly that
    many prices gives the same answer as handing it the whole history and
    keeps this linear in the length of the segment.
    """
    rows = []
    for i in range(RANGE, len(prices)):
        f = features(prices[i - RANGE:i + 1])
        rows.append([f[name] for name in FEATURES])
    return np.asarray(rows, dtype=float), RANGE


def rich_table(bars):
    """The whole-bar descriptors, same alignment as `table`."""
    rows = []
    for i in range(RANGE, len(bars)):
        f = microstructure.features(bars[i - RANGE:i + 1])
        rows.append([f[name] for name in microstructure.FEATURES])
    return np.asarray(rows, dtype=float), RANGE


def forward(prices, offset, count, horizon):
    """Demeaned return over the next `horizon` bars, aligned to the table."""
    out = np.full(count, np.nan)
    for i in range(count):
        j = offset + i
        if j + horizon < len(prices):
            out[i] = prices[j + horizon] / prices[j] - 1.0
    good = np.isfinite(out)
    out[good] -= out[good].mean()
    return out, good


def folded(x, y, folds=FOLDS):
    """Out-of-fold linear prediction, chronological folds, no shuffling.

    Fitted on everything outside the fold and scored inside it, so the number
    it produces is one the same fit would have produced on unseen bars. A fit
    scored on its own data would report an edge for any five columns at all.
    """
    n = len(y)
    out = np.full(n, np.nan)
    edges = [round(i * n / folds) for i in range(folds + 1)]
    for lo, hi in zip(edges, edges[1:]):
        mask = np.ones(n, dtype=bool)
        mask[lo:hi] = False
        a = np.column_stack([x[mask], np.ones(mask.sum())])
        coefficients, *_ = np.linalg.lstsq(a, y[mask], rcond=None)
        b = np.column_stack([x[lo:hi], np.ones(hi - lo)])
        out[lo:hi] = b @ coefficients
    return out


def measure(signal, ret, seed, draws):
    """One (signal, return) pair through `tools.ic`, which owns the statistic."""
    signal = np.asarray(signal, dtype=float)
    runs = [(list(signal - signal.mean()), list(ret))]
    return test(runs, random.Random(seed), draws)


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--candles", type=Path, default=Path("data/candles.json"))
    p.add_argument("--product", default="BTC-USDC")
    p.add_argument("--source", default="closes", choices=["closes", "binance"],
                   help="`binance` reads whole klines and measures the "
                        "whole-bar descriptors in tools/microstructure.py "
                        "instead of the five the fly is fed today")
    p.add_argument("--granularity", default="ONE_HOUR")
    p.add_argument("--segment", default="train",
                   choices=["train", "validation", "test"])
    p.add_argument("--horizons", default=",".join(str(h) for h in HORIZONS))
    p.add_argument("--draws", type=int, default=2000)
    p.add_argument("--seed", type=int, default=20260915)
    p.add_argument("--out", type=Path)
    a = p.parse_args()

    horizons = [int(h) for h in a.horizons.split(",") if h]
    if a.source == "binance":
        bars = klines(a.candles, a.product, a.granularity)
        whole = [b["close"] for b in bars]
        sizes = {k: len(v) for k, v in split(whole).items()}
        lo = {"train": 0, "validation": sizes["train"],
              "test": sizes["train"] + sizes["validation"]}
        cut = slice(lo[a.segment], lo[a.segment] + sizes[a.segment])
        prices, bars = whole[cut], bars[cut]
        x, offset = rich_table(bars)
        names = list(microstructure.FEATURES) + ["combined"]
    else:
        prices = split(
            candle_series(a.candles, a.product, a.granularity))[a.segment]
        x, offset = table(prices)
        names = list(FEATURES) + ["combined"]
    print(f"{a.product} {a.granularity} {a.segment}: {len(prices):,} candles, "
          f"{len(x):,} bars with all {len(names) - 1} descriptors\n")

    report = []
    width = max(len(n) for n in names) + 2
    print(f"{'':{width}}" + "".join(f"{('h=' + str(h)):>18}" for h in horizons))
    print(f"{'':{width}}" + "".join(f"{'ic':>10}{'p':>8}" for _ in horizons))
    for name in names:
        line = f"{name:{width}}"
        for h in horizons:
            y, good = forward(prices, offset, len(x), h)
            xs, ys = x[good], y[good]
            if name == "combined":
                signal = folded(xs, ys)
            else:
                signal = xs[:, names.index(name)]
            t = measure(signal, ys, a.seed + h + hash(name) % 1000, a.draws)
            line += f"{t['ic']:>10.4f}{t['p']:>8.4f}"
            report.append({"feature": name, "horizon": h, **t})
        print(line)

    hits = sum(1 for r in report if r["p"] < 0.05)
    best = max(report, key=lambda r: abs(r["ic"]))
    print(f"\n{len(report)} tests: {hits} under p 0.05, "
          f"{0.05 * len(report):.1f} expected if nothing is there")
    print(f"largest: {best['feature']} at h={best['horizon']}, "
          f"ic {best['ic']:+.4f}, p {best['p']:.4f}")
    if a.out:
        a.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
        print(f"written to {a.out}")


if __name__ == "__main__":
    main()
