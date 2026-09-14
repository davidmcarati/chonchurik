"""Where does a neural observation actually spend its time? Measure, don't guess.

    python tools/kernel_cost.py
    python tools/kernel_cost.py --observations 3 --out cost.json

The kernel has two costs and they are independent. Every timestep it walks the
*active list*, evolving each neuron on it whether or not anything happened to
it; and separately it delivers each spike along that neuron's outgoing edges.
The first scales with how many neurons stay awake, the second with how many
spikes fire. Which dominates decides what a faster implementation would have to
be faster at, and the two answers point at completely different machines.

This asks the kernel itself rather than reasoning about the source: `nactive`
is a live array, `step()` already runs in hundred-tick chunks, and the spike
counts come back per neuron, so the number of active-list visits and the number
of synaptic deliveries can both be counted exactly.

Read-only. Builds its own brain, touches no run directory, and drops itself to
below-normal priority so it can be run beside an evolution without stealing
from it.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from stonkfly.config import Settings
from stonkfly.display import market_frame

def sampled(brain):
    """Wrap the brain's own stepper so the active list is read from inside it.

    Measuring by calling `_neural_step` directly would measure a different
    animal: `observe` re-applies odour, taste and any reinforcement pulse on
    every ten-millisecond bin, and the drive regime is exactly what decides how
    many neurons stay awake. So the real path runs untouched and the counter is
    read at each bin boundary it already has.
    """
    samples, original = [], brain.rgb_step

    def wrapper(frame, duration_ms, **kwargs):
        opening = int(brain.nactive[0])
        out = original(frame, duration_ms, **kwargs)
        # The list grows and shrinks inside a bin; both ends are read and
        # averaged rather than crediting the whole bin to its final size.
        samples.append((0.5 * (opening + int(brain.nactive[0])),
                        round(duration_ms / brain.dt)))
        return out

    brain.rgb_step = wrapper
    return samples, (lambda: brain.__dict__.pop("rgb_step", None))


def series(count, base=60000.0):
    """The repository's own fixture sine, long enough to fill a chart."""
    import math

    return [base * (1 + 0.025 * math.sin(i * 0.6)) for i in range(count)]


def observe(controller, history):
    """One observation through the normal path, so nothing is special-cased."""
    bid, ask = history[-1] * 0.9995, history[-1] * 1.0005
    frame = market_frame("BTC-USDC", history, bid, ask)
    return controller.observe(
        frame, "none", history, None, (str(100.0), str(100.0))
    )


def measure(controller, history):
    """One observation, counting active-list visits and synaptic deliveries.

    Both are exact rather than estimated: the active count is read at every bin
    the kernel already stops at, and a delivery is one traversal of one
    outgoing edge, so the count is the fired neurons weighted by out-degree.
    """
    brain = controller.brain
    outdeg = np.diff(brain.ptr).astype(np.int64)
    samples, restore = sampled(brain)
    started = time.perf_counter()
    try:
        observe(controller, history)
    finally:
        restore()
    elapsed = time.perf_counter() - started
    # `observe` writes this observation's own totals into counts rather than
    # accumulating, so this is already the delta. Subtracting a snapshot would
    # subtract the previous observation and go negative.
    fired = brain.counts.astype(np.int64)
    counts = [n for n, _ in samples]
    return {
        "seconds": elapsed,
        "bins": len(samples),
        "active_mean": float(np.mean(counts)) if counts else 0.0,
        "active_max": int(np.max(counts)) if counts else 0,
        # Each bin's active list is walked once per tick inside that bin.
        "active_visits": int(round(sum(n * ticks for n, ticks in samples))),
        "spikes": int(fired.sum()),
        "deliveries": int((fired * outdeg).sum()),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--observations", type=int, default=2,
                   help="measured observations, after the warm-up")
    p.add_argument("--warmup", type=int, default=15,
                   help="Kenyon activity needs about eight to plateau")
    p.add_argument("--out", type=Path)
    a = p.parse_args()

    from tools.evolve.pool import deprioritise
    deprioritise()

    from stonkfly.neural.controller import FlyController

    from tools.evolve.evaluate import CHART_WINDOW

    settings = Settings()
    print("building the graph…", flush=True)
    built = time.perf_counter()
    controller = FlyController(settings)
    brain = controller.brain
    print(f"  {brain.n:,} neurons, {len(brain.post):,} edges, "
          f"{time.perf_counter() - built:.0f}s", flush=True)

    prices = series(CHART_WINDOW + a.warmup + a.observations + 5)

    for i in range(a.warmup):
        observe(controller, prices[:CHART_WINDOW + i + 1])
    print(f"warmed up over {a.warmup} observations", flush=True)

    rows = []
    for i in range(a.observations):
        history = prices[:CHART_WINDOW + a.warmup + i + 1]
        rows.append(measure(controller, history))
        r = rows[-1]
        print(f"  observation {i}: {r['seconds']:.2f}s  "
              f"{r['spikes']:,} spikes  {r['deliveries']:,} deliveries  "
              f"active mean {r['active_mean']:,.0f} max {r['active_max']:,}",
              flush=True)

    visits = sum(r["active_visits"] for r in rows)
    deliveries = sum(r["deliveries"] for r in rows)
    spikes = sum(r["spikes"] for r in rows)
    seconds = sum(r["seconds"] for r in rows)
    work = visits + deliveries

    # The counts are exact. The split of *time* between them is not measured:
    # separating it would need counters inside the kernel, and that file is
    # hashed into provenance. What is reported is the share of operations,
    # which is what an alternative implementation has to move.
    print(f"\n{'':26}{'operations':>18}{'share':>9}{'per observation':>18}")
    for name, value in [("active-list visits", visits),
                        ("synaptic deliveries", deliveries)]:
        print(f"{name:26}{value:>18,}{100 * value / work:>8.1f}%"
              f"{value / len(rows):>18,.0f}")
    print(f"{'total':26}{work:>18,}{100:>8.1f}%{work / len(rows):>18,.0f}")
    print(f"\n{1e9 * seconds / max(1, work):.1f} ns per operation if the two "
          f"cost alike; both are dominated by the same lazy state update,\n"
          f"so that is a fair first approximation and not a measurement.")
    print(f"\n{spikes:,} spikes over {seconds:.2f}s in "
          f"{len(rows)} observations ({seconds / len(rows):.2f}s each)")
    print(f"mean active list {np.mean([r['active_mean'] for r in rows]):,.0f} "
          f"of {brain.n:,} neurons "
          f"({100 * np.mean([r['active_mean'] for r in rows]) / brain.n:.1f}%)")

    ratio = visits / max(1, deliveries)
    print(f"\nactive-list visits are {ratio:.1f}x the synaptic deliveries." if
          ratio >= 1 else
          f"\nsynaptic deliveries are {1 / ratio:.1f}x the active-list visits.")
    print("The larger number is what a faster kernel has to be faster at.")

    if a.out:
        a.out.write_text(json.dumps({
            "neurons": int(brain.n), "edges": int(len(brain.post)),
            "observations": rows, "active_visits": visits,
            "deliveries": deliveries, "seconds": seconds,
        }, indent=2) + "\n", encoding="utf-8")
        print(f"\nwritten {a.out}")


if __name__ == "__main__":
    main()
