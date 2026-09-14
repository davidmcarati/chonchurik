"""A golden record of the kernel's state, so a rewrite can be proved equal.

    python -m tools.kernel_equivalence --write reference.npz
    python -m tools.kernel_equivalence --check reference.npz

The kernel mutates nineteen arrays. A port that gets eighteen of them right and
one of them subtly wrong would still produce plausible spike counts, plausible
trades and plausible evolution -- and every measurement in `docs/validation.md`
taken afterwards would be describing a different animal without saying so.
Comparing spike totals is not enough; this compares every array, elementwise.

What it does:

  --write   runs the CPU kernel over a fixed, offline stimulus and records the
            exact contents of every mutable array after each observation;
  --check   runs it again and asserts the record is reproduced bit for bit.

The second is not a formality. It establishes that the reference is a property
of the code rather than of the machine that produced it, which is the only
thing that makes it usable as a target for a different implementation.

Offline throughout: the fixture sine, no network, no run directory, nothing
written except the file that is asked for.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from stonkfly.config import Settings

# Everything the kernel may change. `weight` is not among brain.fields because
# it belongs to the graph, but the plasticity rule writes to it, so a port that
# left it alone would pass a test that only looked at brain.fields.
EXTRA = ["weight"]
OBSERVATIONS = 6
WARMUP = 4


def snapshot(brain):
    """Copies of every mutable array, keyed by name."""
    names = list(brain.fields) + EXTRA
    return {name: np.asarray(getattr(brain, name)).copy() for name in names}


def differences(a, b):
    """Every array that differs, with enough detail to find the bug."""
    out = []
    for name in sorted(set(a) | set(b)):
        if name not in a or name not in b:
            out.append((name, "missing", None, None))
            continue
        x, y = a[name], b[name]
        if x.shape != y.shape or x.dtype != y.dtype:
            out.append((name, f"{x.dtype}{x.shape} vs {y.dtype}{y.shape}",
                        None, None))
            continue
        if np.array_equal(x, y):
            continue
        bad = int(np.count_nonzero(x != y))
        if np.issubdtype(x.dtype, np.floating):
            gap = float(np.nanmax(np.abs(x.astype(np.float64)
                                         - y.astype(np.float64))))
        else:
            gap = float(np.max(np.abs(x.astype(np.int64) - y.astype(np.int64))))
        out.append((name, f"{bad:,} of {x.size:,} elements", gap,
                    int(np.argmax(x != y))))
    return out


def trace(observations=OBSERVATIONS, warmup=WARMUP):
    """One deterministic pass, recording every array after each observation."""
    from stonkfly.neural.controller import FlyController

    from .evolve.evaluate import CHART_WINDOW
    from .kernel_cost import observe, series

    controller = FlyController(Settings())
    prices = series(CHART_WINDOW + warmup + observations + 2)
    for i in range(warmup):
        observe(controller, prices[:CHART_WINDOW + i + 1])
    frames = []
    for i in range(observations):
        out = observe(controller, prices[:CHART_WINDOW + warmup + i + 1])
        frames.append({
            "arrays": snapshot(controller.brain),
            "side": out["side"],
            "kc": int(out["KC_spikes"]),
            "spikes": int(controller.brain.counts.sum()),
        })
    return controller.brain, frames


def flatten(frames):
    out = {}
    for i, frame in enumerate(frames):
        for name, array in frame["arrays"].items():
            out[f"{i}/{name}"] = array
        out[f"{i}/_side"] = np.array(frame["side"])
        out[f"{i}/_kc"] = np.array(frame["kc"])
        out[f"{i}/_spikes"] = np.array(frame["spikes"])
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--write", type=Path, help="record a reference")
    p.add_argument("--check", type=Path, help="assert a reference is reproduced")
    p.add_argument("--observations", type=int, default=OBSERVATIONS)
    a = p.parse_args()
    if not a.write and not a.check:
        p.error("give --write or --check")

    started = time.perf_counter()
    brain, frames = trace(a.observations)
    print(f"{brain.n:,} neurons, {len(brain.post):,} edges, "
          f"{a.observations} observations in {time.perf_counter() - started:.0f}s")
    for i, f in enumerate(frames):
        print(f"  observation {i}: {f['spikes']:,} spikes, {f['kc']:,} KC, "
              f"proposal {f['side']}")
    flat = flatten(frames)
    print(f"{len(frames[0]['arrays'])} arrays per observation: "
          f"{', '.join(sorted(frames[0]['arrays']))}")

    if a.write:
        a.write.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(a.write, **flat)
        size = a.write.stat().st_size / 1e6
        print(f"\nwritten {a.write} ({size:.1f} MB)")
        print("Check it on this machine before trusting it anywhere: "
              "`--check` must pass\nagainst a fresh process, or the reference "
              "records the machine and not the code.")
        return

    reference = dict(np.load(a.check))
    failures = 0
    for i in range(a.observations):
        want = {k.split("/", 1)[1]: v for k, v in reference.items()
                if k.startswith(f"{i}/") and not k.split("/")[1].startswith("_")}
        got = frames[i]["arrays"]
        bad = differences(want, got)
        if bad:
            failures += 1
            print(f"\nobservation {i}: {len(bad)} array(s) differ")
            for name, detail, gap, where in bad:
                extra = "" if gap is None else f", max |diff| {gap:.3e}, first at {where:,}"
                print(f"  {name:20} {detail}{extra}")
    if failures:
        print(f"\nFAILED: {failures} of {a.observations} observations differ")
        sys.exit(1)
    print(f"\nidentical: all {a.observations} observations, every array, "
          f"every element")


if __name__ == "__main__":
    main()
