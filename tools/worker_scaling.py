"""How many workers actually help? Measure, because the answer is not obvious.

    python -m tools.worker_scaling
    python -m tools.worker_scaling --workers 8,16,24 --observations 40

The machine has 32 threads and the pool uses 8 by default, so tripling the
throughput looks free. Two measurements argue about whether it is. The kernel
moves 0.2% of this machine's memory bandwidth, which says cores are the only
limit; but each worker keeps roughly 3.3 MB of per-neuron state live, so eight
workers fit in a 36 MB L3 and twenty-four do not. Scaling is therefore an
empirical question, and this answers it in a few minutes rather than by
argument.

Each worker builds its own brain, so the first minute is construction, not
work. Only the evaluation phase is timed. Read-only: no run directory, no
genome search, no state written anywhere.
"""

import argparse
import time
from pathlib import Path

from stonkfly.config import Settings

from .evolve.evaluate import CHART_WINDOW, WARMUP
from .evolve.genome import WILD_TYPE
from .evolve.pool import pool, run_one
from .evolve.series import fixture_series


def throughput(settings, workers, observations, tasks):
    """Wall time for a fixed amount of work, at one worker count.

    The same wild-type genome at every start, so no genome can be cheaper than
    another and confuse the comparison.
    """
    prices = fixture_series(CHART_WINDOW + WARMUP + observations + tasks + 5)
    built = time.perf_counter()
    with pool(settings, workers) as executor:
        # One trivial task first, so brain construction is not timed as work.
        executor.submit(run_one, (WILD_TYPE, prices, 0, 1)).result()
        warm = time.perf_counter()
        futures = [
            executor.submit(run_one, (WILD_TYPE, prices, i, observations))
            for i in range(tasks)
        ]
        rows = [f.result() for f in futures]
        done = time.perf_counter()
    return {
        "workers": workers,
        "build_seconds": warm - built,
        "seconds": done - warm,
        "observations": sum(r["observations"] for r in rows),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workers", default="8,16,24",
                   help="comma-separated worker counts to compare")
    p.add_argument("--observations", type=int, default=30,
                   help="scored observations per task")
    p.add_argument("--tasks", type=int, default=48,
                   help="tasks per measurement; keep it a multiple of every "
                        "worker count so none finishes with idle workers")
    p.add_argument("--out", type=Path)
    a = p.parse_args()

    counts = [int(x) for x in a.workers.split(",") if x.strip()]
    settings = Settings()
    print(f"{a.tasks} tasks x {a.observations} observations, wild type, "
          f"fixture prices\n", flush=True)
    rows = []
    for workers in counts:
        r = throughput(settings, workers, a.observations, a.tasks)
        rate = r["observations"] / r["seconds"]
        r["per_second"] = rate
        rows.append(r)
        print(f"  {workers:3} workers  {r['seconds']:7.1f}s  "
              f"{rate:6.2f} observations/s   "
              f"({r['build_seconds']:.0f}s building)", flush=True)

    base = rows[0]
    print(f"\n{'workers':>8}{'obs/s':>9}{'speedup':>10}{'per worker':>13}"
          f"{'efficiency':>12}")
    for r in rows:
        gain = r["per_second"] / base["per_second"]
        ideal = r["workers"] / base["workers"]
        print(f"{r['workers']:>8}{r['per_second']:>9.2f}{gain:>10.2f}x"
              f"{r['per_second'] / r['workers']:>13.3f}"
              f"{100 * gain / ideal:>11.0f}%")
    print("\nEfficiency is the share of the ideal speedup actually obtained. "
          "Falling\nefficiency means the workers are contending for something "
          "-- most likely L3,\nsince each holds about 3.3 MB of per-neuron "
          "state live.")

    if a.out:
        import json
        a.out.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
        print(f"\nwritten {a.out}")


if __name__ == "__main__":
    main()
