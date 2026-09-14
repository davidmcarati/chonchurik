"""Run the evolution. Offline by default; never opens a socket.

    python -m tools.evolve --source fixture --generations 4 --population 12
    python -m tools.evolve --source candles --candles data/candles.json --resume

`fixture` replays the repository's own deterministic sine. A sine is not a
market: profit on one shows the machinery works, not that anything can trade.
`candles` replays real one-minute closes from a file a human fetched on
purpose, so an evolution run cannot quietly acquire data or change the segment
it is judged on halfway through.
"""

import argparse
import dataclasses
import contextlib
import json
import random
import time
from pathlib import Path

from stonkfly.config import Settings

from . import genome as G
from .evaluate import ceiling
from .loop import (FULL_OBSERVATIONS, FULL_STARTS, evolve,
                   judge)
from .pool import DEFAULT_WORKERS, PoolRunner, pool
from .series import candle_series, fixture_series, klines, split


def build_runner(a, settings, bars=None):
    """The pool or the card, behind the one interface the loop knows about."""
    train = None if bars is None else bars["train"]
    if a.device == "cpu":
        executor = pool(settings, a.workers)
        return PoolRunner(executor, a.workers, a.horizon, train), executor
    from .herd import HerdRunner

    return HerdRunner(settings, a.batch, out=a.out, horizon=a.horizon,
                      bars=train), None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", choices=["fixture", "candles", "binance"],
                   default="fixture",
                   help="`binance` reads whole klines, which opens the three "
                        "whole-bar odour channels -- order flow among them, "
                        "the only descriptor in this repository measured to "
                        "carry anything")
    p.add_argument("--candles", type=Path, default=Path("data/candles.json"))
    p.add_argument("--symbol", default="BTCUSDT",
                   help="binance only: the key inside the kline file")
    p.add_argument("--fee", type=float, default=None,
                   help="per-side fee for the search, overriding Settings. "
                        "Binance spot charges 0.001 against Coinbase's 0.006, "
                        "and the difference decides whether the objective is "
                        "reachable at all: at 0.006 an hourly trade needs an "
                        "information coefficient of 1.08 to break even, which "
                        "is more than perfect foresight")
    p.add_argument("--horizon", type=int, default=6,
                   help="bars ahead the readout is graded against. The search "
                        "is ranked on that information coefficient and not on "
                        "money: money ranked the first run by how little each "
                        "fly traded, correlation -0.97 with the number of "
                        "sells, because at chance-level direction every trade "
                        "loses its fee")
    # One minute is measured dead: perfect foresight makes exactly nothing on
    # every segment, because fifty bars span 0.15% and a round trip costs 1.2%.
    # Run `python -m tools.evolve.survey` before choosing this.
    p.add_argument("--granularity", default="ONE_HOUR")
    p.add_argument("--length", type=int, default=1200,
                   help="fixture only: observations to synthesise")
    p.add_argument("--observations", type=int, default=FULL_OBSERVATIONS,
                   help="scored observations per evaluation; must exceed the "
                        "`hold` column of tools.evolve.survey or no trade has "
                        "room to pay for itself")
    p.add_argument("--generations", type=int, default=10)
    p.add_argument("--population", type=int, default=24)
    p.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                   help="cpu only: 8 leaves the machine usable, raise it for "
                        "a night")
    p.add_argument("--device", choices=["cpu", "gpu"], default="cpu",
                   help="gpu runs a whole wave of flies as one CUDA launch. "
                        "`python -m tools.herd_check` is what says the two "
                        "devices produce the same flies")
    p.add_argument("--batch", type=int, default=None,
                   help="gpu only: flies per wave. Defaults to what fits in "
                        "free device memory, capped at 84 -- the point past "
                        "which the card stops going faster")
    p.add_argument("--seed", type=int, default=20260914)
    p.add_argument("--out", type=Path, default=Path("runs/evolution"))
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()

    bars = None
    if a.source == "binance":
        whole = klines(a.candles, a.symbol, a.granularity)
        series = [b["close"] for b in whole]
    else:
        whole = None
        series = (
            fixture_series(a.length) if a.source == "fixture"
            else candle_series(a.candles, granularity=a.granularity)
        )
    segments = split(series)
    if whole is not None:
        # Sliced with the same offsets as the closes, which `klines`
        # guarantees are index-aligned with them.
        sizes = {k: len(v) for k, v in segments.items()}
        cut, bars = 0, {}
        for name in ["train", "validation", "test"]:
            bars[name] = whole[cut:cut + sizes[name]]
            cut += sizes[name]
    if a.source in ("candles", "binance"):
        dead = [
            name for name, prices in segments.items()
            if ceiling(prices, 0, a.observations)["ceiling"] <= 0
        ]
        if dead:
            raise SystemExit(
                f"nothing to take on {', '.join(dead)} at {a.granularity}: a "
                f"trader with perfect foresight makes zero there, so no policy "
                f"can profit and evolution would select noise. Run "
                f"`python -m tools.evolve.survey` and pick a coarser interval."
            )
    rng = random.Random(a.seed)
    state = None
    if a.resume:
        path = a.out / "population.json"
        if path.exists():
            state = json.loads(path.read_text(encoding="utf-8"))
            print(f"resuming at generation {state['generation']}", flush=True)

    label = a.source if a.source == "fixture" else f"{a.source} {a.granularity}"
    print(f"source {label}: {len(series)} observations, train "
          f"{len(segments['train'])} / validation {len(segments['validation'])} "
          f"/ test {len(segments['test'])}", flush=True)
    settings = Settings()
    if a.fee is not None:
        settings = dataclasses.replace(settings, paper_fee=str(a.fee))
    runner, executor = build_runner(a, settings, bars)
    print(f"{a.population} genomes x {a.generations} generations on "
          f"{runner.describe()}; {a.observations} observations x "
          f"{FULL_STARTS} starts per full evaluation", flush=True)
    # Written before the first generation so a watcher started at any moment
    # knows what it is watching. Nothing reads it back into the run.
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "plan.json").write_text(json.dumps({
        "label": label,
        "generations": a.generations,
        "population": a.population,
        "observations": a.observations,
        "starts": FULL_STARTS,
        "device": a.device,
        "workers": a.workers if a.device == "cpu" else 1,
        "runner": runner.describe(),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }, indent=2) + "\n", encoding="utf-8")
    started = time.time()
    with contextlib.ExitStack() as stack:
        if executor is not None:
            stack.enter_context(executor)
        state = evolve(
            runner, segments, rng, a.generations, a.population, a.out, state,
            a.observations, bars,
        )
        survivors = state["survivors"]
        champion = max(survivors, key=lambda i: i["fitness"])
        print(f"champion {G.identity(champion['genome'])} at "
              f"{champion['fitness']:+.4f} on train; opening validation and "
              f"test once", flush=True)
        report = judge(
            runner, champion, survivors, segments, rng, a.out, a.seed, label,
            a.observations, bars,
        )
    print(f"\n{report['verdict']}\n", flush=True)
    print(f"written {a.out / 'champion.json'} and {a.out / 'population.json'} "
          f"({time.time() - started:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
