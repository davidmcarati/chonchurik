"""Is there money on the table at this sampling interval? Check before evolving.

No connectome, no workers, a second to run. It answers the only question worth
asking before spending a night on a search: how much could anything take from
these prices under these account rules, and how much of that does simply
holding already collect.

    python -m tools.evolve.survey
    python -m tools.evolve.survey --candles data/candles.json --observations 50

A granularity whose ceiling is at or below zero cannot be traded profitably by
any policy. Evolution there selects noise, and the fittest fly will be the one
that never trades.
"""

import argparse
from pathlib import Path

from stonkfly.config import Settings

from .evaluate import CHART_WINDOW, WARMUP, Account, ceiling
from .loop import FULL_OBSERVATIONS, FULL_STARTS, median
from .series import candle_series, quotes, split, starts

GRANULARITIES = ["ONE_MINUTE", "FIFTEEN_MINUTE", "ONE_HOUR", "SIX_HOUR", "ONE_DAY"]


def buy_and_hold(prices, start, observations, settings):
    """Deploy the whole budget as fast as the order size allows, then hold.

    The same account rules every fly is bound by, so it is a fair comparison
    rather than an idealised index.
    """
    account = Account(settings.capital, settings.order_limit, settings.paper_fee)
    window = prices[start + CHART_WINDOW + WARMUP:][:observations]
    for price in window:
        bid, ask = quotes(price)
        account.apply("BUY", bid, ask)
    return account.equity(quotes(window[-1])[0]) - account.start if window else 0.0


def bars_to_clear(prices, cost, samples=400, cap=4000):
    """How long a position must be held before it can cover the round trip.

    The number that decides whether scalping is possible at all. It is a
    property of the market, not of how often the fly looks at it: if BTC needs
    half a day to move 1.2%, sampling every minute instead of every hour does
    not shorten that -- it only gives more looks at the same move.
    """
    import random

    rng = random.Random(20260914)
    hits = []
    room = len(prices) - cap - 1
    if room <= 0:
        cap, room = max(1, len(prices) // 2), len(prices) // 2
    for _ in range(samples):
        i = rng.randrange(room)
        entry = prices[i]
        for step in range(1, cap):
            if abs(prices[i + step] / entry - 1) >= cost:
                hits.append(step)
                break
        else:
            hits.append(cap)
    hits.sort()
    return hits[len(hits) // 2]


def survey(path, product, observations):
    settings = Settings()
    rows = []
    for granularity in GRANULARITIES:
        try:
            segments = split(candle_series(path, product, granularity))
        except (ValueError, KeyError) as e:
            rows.append({"granularity": granularity, "error": str(e)})
            continue
        for name in ["train", "validation", "test"]:
            prices = segments[name]
            try:
                offsets = starts(
                    prices, FULL_STARTS, CHART_WINDOW, observations + WARMUP
                )
            except ValueError:
                rows.append({"granularity": granularity, "segment": name,
                             "error": "segment shorter than one evaluation"})
                continue
            detail = [ceiling(prices, o, observations) for o in offsets]
            rows.append({
                "granularity": granularity,
                "segment": name,
                "candles": len(prices),
                "bars_to_clear_cost": bars_to_clear(
                    prices, 2 * float(settings.paper_fee)
                ),
                "ceiling": median([d["ceiling"] for d in detail]),
                "price_range_percent": median(
                    [d["price_range_percent"] for d in detail]
                ),
                "buy_and_hold": median(
                    [buy_and_hold(prices, o, observations, settings) for o in offsets]
                ),
                "round_trip_cost_percent": detail[0]["round_trip_cost_percent"],
            })
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candles", type=Path, default=Path("data/candles.json"))
    p.add_argument("--product", default="BTC-USDC")
    p.add_argument("--observations", type=int, default=FULL_OBSERVATIONS)
    a = p.parse_args()

    rows = survey(a.candles, a.product, a.observations)
    cost = next((r["round_trip_cost_percent"] for r in rows if "ceiling" in r), 0)
    print(f"{a.product}, {a.observations} observations per evaluation, "
          f"{FULL_STARTS} starts, round trip costs {cost:.1f}% of the order\n")
    print(f"{'granularity':16}{'segment':12}{'candles':>9}{'range %':>9}"
          f"{'ceiling':>10}{'buy+hold':>10}{'hold':>7}  verdict")
    for r in rows:
        if "error" in r:
            print(f"{r['granularity']:16}{r.get('segment', ''):12}"
                  f"  {r['error']}")
            continue
        if r["ceiling"] <= 0:
            note = "DEAD: perfect foresight makes nothing"
        elif r["ceiling"] < abs(r["buy_and_hold"]):
            note = "holding already collects more than trading can"
        else:
            note = "tradeable"
        hold = r["bars_to_clear_cost"]
        if hold > a.observations:
            note = (f"hold of {hold} bars needed, evaluation is only "
                    f"{a.observations}")
        print(f"{r['granularity']:16}{r['segment']:12}{r['candles']:9,}"
              f"{r['price_range_percent']:9.2f}{r['ceiling']:10.3f}"
              f"{r['buy_and_hold']:10.3f}{hold:7}  {note}")
    print("\nA ceiling at or below zero cannot be traded by any policy. `hold` "
          "is the median number of bars before price moves far enough to cover "
          "the round trip -- a scalp cannot be shorter than that, whatever the "
          "sampling interval, and an evaluation shorter than it has no room "
          "for one.")


if __name__ == "__main__":
    main()
