"""Price series, and the chronological split the experiment is judged on.

Two providers. `fixture` is the repository's own deterministic sine and is
offline: a sine is not a market, and a fly that profits on one has shown that
the evolution machinery works, not that it can trade. `candles` replays real
one-minute Coinbase closes from a local file that a human fetched on purpose;
nothing here reaches the network.

The split is chronological and fixed by fraction. Train is for evolution,
validation for every choice made while looking at results, and test is
evaluated exactly once, at the end, against the pre-declared kill criterion.
"""

import json
import math
from pathlib import Path

TRAIN, VALIDATION = 0.6, 0.2
SPREAD = 0.0005  # per side, matching the repository's own fixture quotes


def fixture_series(length, product="BTC-USDC"):
    """The repository's FixtureMarket sine, extended to any length.

    Deliberately identical in form to stonkfly/market.py so a result here can
    be reproduced by the normal run loop.
    """
    base = {"BTC-USDC": 60000.0, "ETH-USDC": 2500.0, "SOL-USDC": 100.0}[product]
    return [base * (1 + 0.025 * math.sin(i * 0.6)) for i in range(length)]


def candle_series(path, product="BTC-USDC"):
    """Closes from a local candle file, oldest first.

    The file is produced by a separate, deliberate fetch. This function never
    opens a socket, so an evolution run cannot quietly acquire new data or
    change the segment it is judged on halfway through.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = data[product] if isinstance(data, dict) else data
    closes = [float(r["close"] if isinstance(r, dict) else r) for r in rows]
    if len(closes) < 200 or not all(math.isfinite(c) and c > 0 for c in closes):
        raise ValueError("Candle file needs at least 200 finite positive closes")
    return closes


def split(series, train=TRAIN, validation=VALIDATION):
    """Chronological, never shuffled: shuffling would leak the future."""
    n = len(series)
    a = int(n * train)
    b = a + int(n * validation)
    if min(a, b - a, n - b) < 50:
        raise ValueError("Series too short to split into three usable segments")
    return {"train": series[:a], "validation": series[a:b], "test": series[b:]}


def starts(segment, count, window, observations):
    """Independent chronological start points inside one segment.

    This replaces the multi-seed protocol the plan asked for. The kernel is
    deterministic -- the same genome on the same prices gives a byte-identical
    spike train every time -- so running five random seeds would produce five
    identical numbers and a false impression of robustness. Independent start
    points vary the thing that can actually vary: when the fly is switched on.
    """
    span = window + observations
    if len(segment) < span:
        raise ValueError("Segment shorter than one evaluation")
    room = len(segment) - span
    if count == 1 or room == 0:
        return [0]
    return [round(i * room / (count - 1)) for i in range(count)]


def quotes(price, spread=SPREAD):
    return price * (1 - spread), price * (1 + spread)
