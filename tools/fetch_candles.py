"""Fetch public Coinbase candles into a local file. Run deliberately, by hand.

This is the only thing in tools/ that opens a socket, and it is separate from
the evolution on purpose: an evolution run must not be able to quietly acquire
data or change the segment it is judged on halfway through.

Public endpoints only. The SDK is constructed with api_key=None and
api_secret=None, the same override stonkfly/market.py uses, so no account
credential is read or sent even if one exists in this process. Nothing is
uploaded; the request carries a product id, a time range and a granularity.

    python tools/fetch_candles.py
    python tools/fetch_candles.py --product ETH-USDC --granularity ONE_HOUR

Output goes to data/candles.json, which is gitignored.
"""

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

# Candles per request, and how far back to go for each granularity. The spans
# are chosen so every granularity yields enough for a 60/20/20 chronological
# split with room for several independent evaluation starts in each segment.
PAGE = 300
SECONDS = {
    "ONE_MINUTE": 60,
    "FIVE_MINUTE": 300,
    "FIFTEEN_MINUTE": 900,
    "ONE_HOUR": 3600,
    "SIX_HOUR": 21600,
    "ONE_DAY": 86400,
}
SPANS = {
    "ONE_MINUTE": 10 * 86400,
    "FIFTEEN_MINUTE": 90 * 86400,
    "ONE_HOUR": 730 * 86400,
    "SIX_HOUR": 1460 * 86400,
    "ONE_DAY": 2000 * 86400,
}


def client():
    from coinbase.rest import RESTClient

    # Public observations never need an account key, even when a live broker
    # exists in this process.
    return RESTClient(api_key=None, api_secret=None, timeout=20)


def unwrap(value):
    return value.to_dict() if hasattr(value, "to_dict") else value


def fetch(rest, product, granularity, span, pause=0.25):
    """Page backwards from the last completed candle. Never reads the future."""
    step = SECONDS[granularity]
    end = int(time.time() // step) * step
    stop = end - span
    rows, cursor = {}, end
    requests = 0
    while cursor > stop:
        start = max(stop, cursor - PAGE * step)
        page = unwrap(
            rest.get_public_candles(product, str(start), str(cursor), granularity)
        ).get("candles", [])
        requests += 1
        fresh = 0
        for candle in page:
            begin = int(candle["start"])
            # `end` is the start of the candle still forming; exclude it.
            if begin >= end or begin < stop:
                continue
            if begin not in rows:
                fresh += 1
            rows[begin] = float(candle["close"])
        if not page or not fresh:
            break
        cursor = start
        time.sleep(pause)
    ordered = sorted(rows.items())
    return [{"start": t, "close": c} for t, c in ordered], requests


def inspect(rows, granularity):
    """What was actually received, so a gap is visible rather than assumed."""
    step = SECONDS[granularity]
    times = [r["start"] for r in rows]
    closes = [r["close"] for r in rows]
    gaps = sum(1 for a, b in zip(times, times[1:]) if b - a != step)
    returns = [
        (b / a - 1) for a, b in zip(closes, closes[1:]) if a > 0
    ]
    mean = sum(returns) / len(returns) if returns else 0.0
    variance = (
        sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
        if len(returns) > 1 else 0.0
    )
    return {
        "candles": len(rows),
        "missing_intervals": gaps,
        "first": datetime.fromtimestamp(times[0], timezone.utc).isoformat()
        if times else None,
        "last": datetime.fromtimestamp(times[-1], timezone.utc).isoformat()
        if times else None,
        "low": min(closes) if closes else None,
        "high": max(closes) if closes else None,
        "return_sd_percent": variance**0.5 * 100,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--product", default="BTC-USDC")
    p.add_argument("--granularity", action="append", choices=sorted(SECONDS),
                   help="repeatable; default fetches every granularity")
    p.add_argument("--out", type=Path, default=Path("data/candles.json"))
    a = p.parse_args()
    wanted = a.granularity or ["ONE_MINUTE", "FIFTEEN_MINUTE", "ONE_HOUR",
                               "SIX_HOUR", "ONE_DAY"]

    rest = client()
    series, report, total = {}, {}, 0
    for granularity in wanted:
        rows, requests = fetch(rest, a.product, granularity, SPANS[granularity])
        total += requests
        series[granularity] = rows
        report[granularity] = {**inspect(rows, granularity), "requests": requests}
        r = report[granularity]
        print(f"  {granularity:15} {r['candles']:7,} candles  "
              f"{str(r['first'])[:10]} to {str(r['last'])[:10]}  "
              f"gaps {r['missing_intervals']:4}  per-bar sd "
              f"{r['return_sd_percent']:.3f}%  ({requests} requests)", flush=True)

    payload = {a.product: series}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, separators=(",", ":"))
    a.out.write_text(text, encoding="utf-8")
    manifest = {
        "product": a.product,
        "source": "Coinbase Advanced public candles, unauthenticated",
        "fetched_utc": datetime.now(timezone.utc).isoformat(),
        "requests": total,
        "sha256": hashlib.sha256(text.encode()).hexdigest(),
        "granularities": report,
    }
    a.out.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\nwritten {a.out} ({len(text) / 1e6:.1f} MB) and its manifest; "
          f"{total} requests total", flush=True)


if __name__ == "__main__":
    main()
