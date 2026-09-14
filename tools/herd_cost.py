"""Where a herd's time goes, at the batch size an evolution would use.

    python -m tools.herd_cost --batch 84 --observations 2

Four buckets, measured rather than estimated, with the device synchronised at
each boundary so the kernel's time is the kernel's and not the next host call's:

  sensory   rgb_bin and prepare_drive, once per fly per 10 ms bin, on the CPU;
  upload    the drive vectors across the bus and into the packed cells;
  kernel    the integration, one launch per bin;
  rule      the counts back, rule.advance once per fly, weights back out.

The point of the split is that the kernel is the only part that moved to the
card. When it stops being the largest bucket, the next thing to work on is on
the host, and this says which host thing.

Offline: the fixture series, no network, no run directory, nothing written.
"""

import argparse
import time

from stonkfly.config import Settings

from .evolve.evaluate import WARMUP, build
from .evolve.genome import WILD_TYPE, random_genome


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch", type=int, default=84)
    p.add_argument("--starts", type=int, default=1)
    p.add_argument("--observations", type=int, default=2)
    p.add_argument("--seed", type=int, default=20260914)
    a = p.parse_args()

    import random

    from .evolve.herd import Herd, replay_herd
    from .evolve.series import fixture_series, starts

    settings = Settings()
    print("building…", flush=True)
    controller, pristine = build(settings)
    rng = random.Random(a.seed)
    genomes = [dict(WILD_TYPE)] + [random_genome(rng) for _ in range(a.batch - 1)]
    prices = fixture_series(2400)
    offsets = starts(prices, a.starts, 100, a.observations + WARMUP)
    tasks = [(g, offsets[i % len(offsets)]) for i, g in enumerate(genomes)]

    herd = Herd(controller, pristine, [g for g, _ in tasks], settings,
                [s for _, s in tasks], measure=True)
    started = time.perf_counter()
    replay_herd(herd, prices, a.observations)
    wall = time.perf_counter() - started

    passes = WARMUP + a.observations
    total = sum(herd.cost.values())
    print(f"\n{a.batch} flies x {passes} observations "
          f"({a.observations} scored after {WARMUP} of warm-up) "
          f"in {wall:.1f}s")
    for name, seconds in sorted(herd.cost.items(), key=lambda kv: -kv[1]):
        print(f"  {name:9} {seconds:7.1f}s  {100 * seconds / total:5.1f}%")
    other = wall - total
    print(f"  {'charts':9} {other:7.1f}s  {100 * other / wall:5.1f}% "
          f"(rendering and the accounts, outside the bins)")
    per = wall / passes / a.batch
    print(f"\n{per * 1000:.0f} ms per fly-observation, "
          f"{1 / per:.1f} fly-observations a second")


if __name__ == "__main__":
    main()
