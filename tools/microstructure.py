"""Candidate descriptors built from a whole bar, and the three that were kept.

`tools/features.py` measured the five descriptors the fly was fed before this
and found no edge at any horizon on 10,446 hourly bars. They are all functions
of one number, the close, so that they share a fate is not a surprise: there
is only one series in them.

A Binance kline carries five more observations per bar. These are the
first-order readings that can be built out of them. Nothing here was claimed
to work -- the point of writing them down was to measure them, and
`tools/features.py --source binance` does exactly that before any of it
reaches a neuron.

What the measurement said, on BTCUSDT, both timeframes, train and validation:

    flow, bar_position   same sign in all four panels, about -0.03 at one bar
                         and gone by six. Short-horizon reversion. Small, and
                         the only effect any descriptor in this project has
                         shown. Adopted.
    flow_slow            the same reading over the fast window. Adopted, so a
                         single bar cannot carry the channel alone.
    volume_z, trade_size, range_z
                         nothing, at any horizon, on either timeframe. Kept
                         here so the arm they were rejected on can be re-run,
                         and not given to the fly: the glomeruli are a fixed
                         budget and every channel costs the others resolution.

The three that were adopted are imported from the model rather than repeated
here, so the numbers this tool reports are the ones the fly actually smells.
Two copies of a feature definition drift apart exactly where nobody looks.
"""

import math

import numpy as np

from stonkfly.neural.olfaction import BARS, RANGE, bar_features, squash

# Adopted first, in the model's own order, then the three that were not.
REJECTED = ["volume_z", "trade_size", "range_z"]
FEATURES = list(BARS) + REJECTED
# Full deflection at this many standard deviations of the descriptor's own
# recent history, matching olfaction.DEVIATIONS in spirit: a fixed number of
# per cent would tie the channel to one venue and one interval.
DEVIATIONS = 2.0


def _z(values, deviations=DEVIATIONS):
    """Last value as a squashed z-score against the rest of the window."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) < 3:
        return 0.5
    sd = float(v[:-1].std())
    if sd <= 0:
        return 0.5
    return squash(float(v[-1] - v[:-1].mean()) / (sd * deviations))


def features(bars):
    """All six descriptors from a window of klines, newest last.

    The adopted three come from `olfaction.bar_features`, which is what the
    fly is fed; the rejected three are computed here.
    """
    if len(bars) < 3:
        return {name: 0.5 for name in FEATURES}
    window = bars[-RANGE:]

    def size(bar):
        n = bar.get("trades", 0)
        return bar.get("quote_volume", 0.0) / n if n else math.nan

    return {
        **bar_features(bars),
        "volume_z": _z([math.log(b["volume"]) if b["volume"] > 0 else math.nan
                        for b in window]),
        "trade_size": _z([math.log(s) if np.isfinite(s) and s > 0 else math.nan
                          for s in (size(b) for b in window)]),
        "range_z": _z([(b["high"] - b["low"]) / b["close"]
                       if b["close"] > 0 else math.nan for b in window]),
    }
