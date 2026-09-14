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
from .evaluate import ceiling
from .loop import (FULL_OBSERVATIONS, FULL_STARTS, evolve,
                   judge)
from .pool import DEFAULT_WORKERS, pool
from .series import candle_series, fixture_series, split


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", choices=["fixture", "candles"], default="fixture")
    p.add_argument("--candles", type=Path, default=Path("data/candles.json"))
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
                   help="8 leaves the machine usable; raise it for a night")
    p.add_argument("--seed", type=int, default=20260914)
    p.add_argument("--out", type=Path, default=Path("runs/evolution"))
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()

    series = (
        fixture_series(a.length) if a.source == "fixture"
        else candle_series(a.candles, granularity=a.granularity)
    )
    segments = split(series)
    if a.source == "candles":
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
    print(f"{a.population} genomes x {a.generations} generations on "
          f"{a.workers} workers at below-normal priority; "
          f"{a.observations} observations x {FULL_STARTS} starts per full "
          f"evaluation", flush=True)
    # Written before the first generation so a watcher started at any moment
    # knows what it is watching. Nothing reads it back into the run.
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "plan.json").write_text(json.dumps({
        "label": label,
        "generations": a.generations,
        "population": a.population,
        "observations": a.observations,
        "starts": FULL_STARTS,
        "workers": a.workers,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }, indent=2) + "
", encoding="utf-8")
    started = time.time()
    settings = Settings()
    with pool(settings, a.workers) as executor:
        state = evolve(
            executor, segments, rng, a.generations, a.population, a.out, state,
            a.observations,
        )
        survivors = state["survivors"]
        champion = max(survivors, key=lambda i: i["fitness"])
        print(f"champion {G.identity(champion['genome'])} at "
              f"{champion['fitness']:+.4f} on train; opening validation and "
              f"test once", flush=True)
        report = judge(
            executor, champion, survivors, segments, rng, a.out, a.seed, label,
            a.observations,
        )
    print(f"\n{report['verdict']}\n", flush=True)
    print(f"written {a.out / 'champion.json'} and {a.out / 'population.json'} "
          f"({time.time() - started:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
