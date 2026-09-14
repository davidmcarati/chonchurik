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
import json
import random
import time
from pathlib import Path

from stonkfly.config import Settings

from . import genome as G
from .loop import FULL_OBSERVATIONS, FULL_STARTS, evolve, judge
from .pool import DEFAULT_WORKERS, pool
from .series import candle_series, fixture_series, split


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", choices=["fixture", "candles"], default="fixture")
    p.add_argument("--candles", type=Path, default=Path("data/candles.json"))
    p.add_argument("--length", type=int, default=1200,
                   help="fixture only: observations to synthesise")
    p.add_argument("--generations", type=int, default=10)
    p.add_argument("--population", type=int, default=24)
    p.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                   help="8 leaves the machine usable; raise it for a night")
    p.add_argument("--seed", type=int, default=20260914)
    p.add_argument("--out", type=Path, default=Path("runs/evolution"))
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()

    series = (
        fixture_series(a.length) if a.source == "fixture"
        else candle_series(a.candles)
    )
    segments = split(series)
    rng = random.Random(a.seed)
    state = None
    if a.resume:
        path = a.out / "population.json"
        if path.exists():
            state = json.loads(path.read_text(encoding="utf-8"))
            print(f"resuming at generation {state['generation']}", flush=True)

    print(f"source {a.source}: {len(series)} observations, train "
          f"{len(segments['train'])} / validation {len(segments['validation'])} "
          f"/ test {len(segments['test'])}", flush=True)
    print(f"{a.population} genomes x {a.generations} generations on "
          f"{a.workers} workers at below-normal priority; "
          f"{FULL_OBSERVATIONS} observations x {FULL_STARTS} starts per full "
          f"evaluation", flush=True)
    started = time.time()
    settings = Settings()
    with pool(settings, a.workers) as executor:
        state = evolve(
            executor, segments, rng, a.generations, a.population, a.out, state
        )
        survivors = state["survivors"]
        champion = max(survivors, key=lambda i: i["fitness"])
        print(f"champion {G.identity(champion['genome'])} at "
              f"{champion['fitness']:+.4f} on train; opening validation and "
              f"test once", flush=True)
        report = judge(executor, champion, survivors, segments, rng, a.out, a.seed)
    print(f"\n{report['verdict']}\n", flush=True)
    print(f"written {a.out / 'champion.json'} and {a.out / 'population.json'} "
          f"({time.time() - started:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
