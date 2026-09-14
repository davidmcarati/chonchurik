"""Does a fly evaluated on the GPU propose what the same fly proposes on the CPU?

    python -m tools.herd_check --genomes 4 --starts 2 --observations 6

`tools/gpu_port_check.py` proves the kernel reproduces `kernel.cpp` element for
element. That is the hard part and it is not the whole claim: between the
kernel launches sit the retina, the olfactory and gustatory channels, the
memory rule, the accounts and the decoder, and a herd that got any of those
subtly wrong would still produce plausible profits and a plausible champion.

So this runs the same genomes twice -- once through `evolve.evaluate`, the
worker-process path the evolution has been using, and once through
`evolve.herd` -- from the same prices and the same starts, and compares the
rows. Every field has to match: profit, the count of each proposal, the fills,
the Kenyon spikes. Not close; equal.

More than one start in a wave is worth checking rather than assuming. Flies
sharing a start share the chart and the photoreceptor settling, and flies that
do not share one must not share either.

Offline throughout: the fixture series, no network, no run directory, nothing
written.
"""

import argparse
import random
import time

from stonkfly.config import Settings

from .evolve.evaluate import WARMUP, build, evaluate
from .evolve.genome import WILD_TYPE, random_genome

FIELDS = ["profit", "final_equity", "observations", "buy", "sell", "hold",
          "rejected", "kc_spikes", "buy_and_hold", "excess"]


def population(count, seed):
    """The wild type first, so a failure on it is a failure on the default."""
    rng = random.Random(seed)
    return [dict(WILD_TYPE)] + [random_genome(rng) for _ in range(count - 1)]


def compare(cpu, gpu):
    """Every field of one fly. Returns the failures rather than printing."""
    bad = [(name, cpu[name], gpu[name]) for name in FIELDS
           if cpu[name] != gpu[name]]
    for side in ("BUY", "SELL"):
        if cpu["fills"][side] != gpu["fills"][side]:
            bad.append((f"fills/{side}", cpu["fills"][side], gpu["fills"][side]))
    return bad


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--genomes", type=int, default=4)
    p.add_argument("--starts", type=int, default=1)
    p.add_argument("--observations", type=int, default=6)
    p.add_argument("--seed", type=int, default=20260914)
    a = p.parse_args()

    from .evolve.herd import Herd, evaluate_herd
    from .gpu_port_check import assert_no_repeated_targets
    from .evolve.series import fixture_series, starts

    settings = Settings()
    print("building…", flush=True)
    controller, pristine = build(settings)
    genomes = population(a.genomes, a.seed)
    assert_no_repeated_targets(controller.brain.ptr, controller.brain.post)
    prices = fixture_series(2400)
    offsets = starts(prices, a.starts, 100, a.observations + WARMUP)
    tasks = [(i, start) for i in range(len(genomes)) for start in offsets]
    print(f"  {controller.brain.n:,} neurons, {len(genomes)} genomes x "
          f"{len(offsets)} start(s) = {len(tasks)} flies, {a.observations} "
          f"observations after {WARMUP} of warm-up\n")

    started = time.perf_counter()
    herd = Herd(controller, pristine, [genomes[i] for i, _ in tasks],
                settings, [s for _, s in tasks])
    built = time.perf_counter() - started
    started = time.perf_counter()
    gpu = evaluate_herd(herd, prices, a.observations)
    gpu_seconds = time.perf_counter() - started
    print(f"GPU herd of {len(tasks)}: {gpu_seconds:6.1f}s "
          f"({built:.1f}s to build)")
    del herd

    started = time.perf_counter()
    cpu = [evaluate(controller, pristine, genomes[i], prices, start,
                    a.observations, settings) for i, start in tasks]
    cpu_seconds = time.perf_counter() - started
    print(f"CPU one at a time:   {cpu_seconds:6.1f}s   "
          f"({cpu_seconds / max(gpu_seconds, 1e-9):.1f}x one core)\n")

    failures = 0
    for (i, start), one, many in zip(tasks, cpu, gpu):
        bad = compare(one, many)
        label = f"{'wild type' if i == 0 else f'genome {i}'} @ {start}"
        if bad:
            failures += 1
            print(f"{label}: {len(bad)} field(s) differ")
            for name, want, got in bad:
                print(f"    {name:16} cpu {want!r:>22}   gpu {got!r}")
        else:
            print(f"{label:22} identical — profit {one['profit']:+.4f}, "
                  f"{one['buy']}/{one['sell']}/{one['hold']} B/S/H, "
                  f"{one['kc_spikes']:,} KC")
    if failures:
        raise SystemExit(f"\nFAILED: {failures} of {len(cpu)} flies differ")
    print(f"\nIDENTICAL: all {len(cpu)} flies, every field")


if __name__ == "__main__":
    main()
