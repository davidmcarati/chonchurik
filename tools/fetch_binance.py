"""Fetch public Binance klines into a local file. Run deliberately, by hand.

Same contract as `tools/fetch_candles.py`: it opens a socket, the evolution
never does, and the two are separate so a run cannot quietly acquire data or
move the segment it is judged on halfway through.

Public market data only. `/api/v3/klines` takes a symbol, an interval and a
time range and needs no key, so no credential is read or sent. Nothing is
uploaded.

    python tools/fetch_binance.py
    python tools/fetch_binance.py --symbol ETHUSDT --interval 1h

Why this exists
---------------
Two measured reasons, both in docs/validation.md.

A round trip on Coinbase Advanced costs 130 basis points, and at that price
the information coefficient needed merely to break even on hourly bars is
1.08 -- larger than perfect foresight. Binance spot charges 10 basis points a
side, which is 20 for the round trip, and moves the whole break-even table
into a range real signals occupy.

And the Coinbase fetcher kept one number per bar. `tools/features.py` measured
the five descriptors built from that number and found no edge at any horizon
on 10,446 bars. A kline carries the high, the low, the volume, the number of
trades, and -- the one that is hard to get anywhere else -- how much of that
volume was taker buying. Buy volume over total volume is order flow, which is
a different kind of observation from price, not another function of it.

Output goes to data/binance.json, which is gitignored, in the shape
`tools/evolve/series.candle_series` already reads.
"""

import argparse
import hashlib
import json
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

# The public market-data host first; the main API host is geo-restricted in
# places where this one is not, and both serve identical klines.
HOSTS = ["https://data-api.binance.vision", "https://api.binance.com"]
LIMIT = 1000
# Binance interval strings, under the names the rest of the repository uses so
# a granularity means the same thing whichever venue supplied it.
INTERVALS = {
    "ONE_MINUTE": ("1m", 60),
    "FIVE_MINUTE": ("5m", 300),
    "FIFTEEN_MINUTE": ("15m", 900),
    "ONE_HOUR": ("1h", 3600),
    "SIX_HOUR": ("6h", 21600),
    "ONE_DAY": ("1d", 86400),
}
# Matching tools/fetch_candles.py exactly, so the two venues split into the
# same shaped segments and a result on one can be compared with the other.
SPANS = {
    "ONE_MINUTE": 365 * 86400,
    "FIVE_MINUTE": 365 * 86400,
    "FIFTEEN_MINUTE": 365 * 86400,
    "ONE_HOUR": 730 * 86400,
    "SIX_HOUR": 1460 * 86400,
    "ONE_DAY": 2000 * 86400,
}
# Position of each field in a kline array, per the documented response.
OPEN, HIGH, LOW, CLOSE, VOLUME = 1, 2, 3, 4, 5
QUOTE_VOLUME, TRADES, TAKER_BASE, TAKER_QUOTE = 7, 8, 9, 10


def get(path, params, pause=0.2):
    """One GET against the first host that answers, as JSON."""
    query = "&".join(f"{k}={v}" for k, v in params.items())
    last = None
    for host in HOSTS:
        url = f"{host}{path}?{query}"
        try:
            request = urllib.request.Request(
                url, headers={"User-Agent": "stonkfly-fetch/1"}
            )
            with urllib.request.urlopen(request, timeout=20) as response:
                return json.loads(response.read().decode())
        except (urllib.error.URLError, urllib.error.HTTPError, OSError) as e:
            last = f"{host}: {e}"
            time.sleep(pause)
    raise RuntimeError(f"no Binance host answered ({last})")


def row(kline):
    """One kline as a dict. `close` is the key `candle_series` reads."""
    return {
        "start": int(kline[0]) // 1000,
        "open": float(kline[OPEN]),
        "high": float(kline[HIGH]),
        "low": float(kline[LOW]),
        "close": float(kline[CLOSE]),
        "volume": float(kline[VOLUME]),
        "quote_volume": float(kline[QUOTE_VOLUME]),
        "trades": int(kline[TRADES]),
        # Of the volume in this bar, how much was somebody lifting the offer.
        # The remainder was somebody hitting the bid. This is the order flow.
        "taker_buy_base": float(kline[TAKER_BASE]),
        "taker_buy_quote": float(kline[TAKER_QUOTE]),
    }


def fetch(symbol, granularity, span, pause=0.2):
    """Page forward over the range, stopping before the bar still forming."""
    interval, step = INTERVALS[granularity]
    end = int(time.time() // step) * step
    cursor = end - span
    rows = {}
    requests = 0
    while cursor < end:
        page = get("/api/v3/klines", {
            "symbol": symbol, "interval": interval,
            "startTime": cursor * 1000, "endTime": end * 1000,
            "limit": LIMIT,
        })
        requests += 1
        if not page:
            break
        fresh = 0
        for kline in page:
            r = row(kline)
            # The last bar of the range is still open; a close that can still
            # change is a look at the future by one interval.
            if r["start"] >= end:
                continue
            if r["start"] not in rows:
                fresh += 1
            rows[r["start"]] = r
        advanced = int(page[-1][0]) // 1000 + step
        if not fresh or advanced <= cursor:
            break
        cursor = advanced
        time.sleep(pause)
    return [rows[t] for t in sorted(rows)], requests


def inspect(rows, granularity):
    """What was actually received, so a gap is visible rather than assumed."""
    _, step = INTERVALS[granularity]
    times = [r["start"] for r in rows]
    closes = [r["close"] for r in rows]
    gaps = sum(1 for a, b in zip(times, times[1:]) if b - a != step)
    returns = [b / a - 1 for a, b in zip(closes, closes[1:]) if a > 0]
    mean = sum(returns) / len(returns) if returns else 0.0
    variance = (sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
                if len(returns) > 1 else 0.0)
    flow = [r["taker_buy_base"] / r["volume"] for r in rows if r["volume"] > 0]
    return {
        "candles": len(rows),
        "missing_intervals": gaps,
        "first": datetime.fromtimestamp(times[0], timezone.utc).isoformat()
        if times else None,
        "last": datetime.fromtimestamp(times[-1], timezone.utc).isoformat()
        if times else None,
        "low": min(closes) if closes else None,
        "high": max(closes) if closes else None,
        "return_sd_percent": variance ** 0.5 * 100,
        "mean_taker_buy_share": sum(flow) / len(flow) if flow else None,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--interval", action="append", choices=sorted(INTERVALS),
                   help="repeatable; default fetches every interval")
    p.add_argument("--out", type=Path, default=Path("data/binance.json"))
    a = p.parse_args()
    wanted = a.interval or ["ONE_MINUTE", "FIFTEEN_MINUTE", "ONE_HOUR",
                            "SIX_HOUR", "ONE_DAY"]

    print(f"{a.symbol} from Binance public klines", flush=True)
    series, report, total = {}, {}, 0
    for granularity in wanted:
        rows, requests = fetch(a.symbol, granularity, SPANS[granularity])
        total += requests
        series[granularity] = rows
        report[granularity] = {**inspect(rows, granularity),
                               "requests": requests}
        r = report[granularity]
        print(f"  {granularity:15} {r['candles']:7,} candles  "
              f"{str(r['first'])[:10]} to {str(r['last'])[:10]}  "
              f"gaps {r['missing_intervals']:4}  per-bar sd "
              f"{r['return_sd_percent']:.3f}%  taker buy "
              f"{r['mean_taker_buy_share']:.3f}  ({requests} requests)",
              flush=True)

    payload = {a.symbol: series}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, separators=(",", ":"))
    a.out.write_text(text, encoding="utf-8")
    manifest = {
        "symbol": a.symbol,
        "source": "Binance public klines (/api/v3/klines), unauthenticated",
        "fetched_utc": datetime.now(timezone.utc).isoformat(),
        "requests": total,
        "sha256": hashlib.sha256(text.encode()).hexdigest(),
        "fields": sorted(row([0] * 12)),
        "granularities": report,
    }
    a.out.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\nwritten {a.out} ({len(text) / 1e6:.1f} MB) and its manifest; "
          f"{total} requests total", flush=True)


if __name__ == "__main__":
    main()
