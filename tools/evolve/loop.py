"""The evolution loop, its screening tier, and the protections against itself.

With fourteen free parameters and dozens of generations, a profitable fly will
be found. That is not in question and it is not evidence of anything. Every
rule below exists to keep the difference visible between "a combination of
parameters suited this stretch of history" and "the fly learned to trade":

  - the test segment is evaluated exactly once, at the very end;
  - fitness is the median over independent chronological starts, so one lucky
    start cannot carry a genome;
  - four baselines are run on the same segments and the same account rules;
  - the kill criterion is declared here, in code, before any run;
  - the whole final population is reported out of sample, not just the winner.
"""

import json
import time
from pathlib import Path

from . import genome as G
from .evaluate import WARMUP, ceiling, degenerate
from .pool import run_baselines, run_one
from .series import starts

SCREEN_OBSERVATIONS, SCREEN_STARTS = 20, 1
FULL_OBSERVATIONS, FULL_STARTS = 50, 5
SURVIVOR_FRACTION = 1 / 3
ELITES = 2
# Declared before any evolution: a champion that does not beat every baseline
# on the held-out test segment is reported as noise, not as a result.
BASELINES = ["buy_and_hold", "all_cash", "random", "wild_type"]


def median(values):
    ordered = sorted(values)
    n = len(ordered)
    if not n:
        return float("-inf")
    mid = n // 2
    return ordered[mid] if n % 2 else 0.5 * (ordered[mid - 1] + ordered[mid])


def fitness(rows):
    """Profit, as the median over independent starts. Nothing else."""
    return median([r["profit"] for r in rows])


def evaluate_population(executor, population, prices, observations, count, window):
    """One (genome, start) task per future; the pool decides the packing."""
    # The warm-up is consumed from the same segment, so a start that leaves
    # room only for the scored observations would silently score a short run.
    offsets = starts(prices, count, window, observations + WARMUP)
    futures = {}
    for index, individual in enumerate(population):
        for start in offsets:
            futures[executor.submit(
                run_one, (individual["genome"], prices, start, observations)
            )] = (index, start)
    rows = [[] for _ in population]
    for future, (index, _) in futures.items():
        rows[index].append(future.result())
    return rows


def screen(executor, population, prices, window):
    """Cheap pass that removes flies that cannot act before paying for them."""
    rows = evaluate_population(
        executor, population, prices, SCREEN_OBSERVATIONS, SCREEN_STARTS, window
    )
    for individual, row in zip(population, rows):
        individual["screen"] = row[0]
        individual["degenerate"] = degenerate(row[0])
        individual["screen_fitness"] = (
            float("-inf") if individual["degenerate"] else fitness(row)
        )
    alive = [i for i in population if not i["degenerate"]]
    keep = max(ELITES, int(len(population) * SURVIVOR_FRACTION))
    if not alive:
        # Every genome proposed one thing for every observation, or none of
        # them fired at all. Carrying the whole generation forward would let
        # the loop crash or, worse, quietly restart from random genomes and
        # lose the record that this happened.
        alive = sorted(population, key=lambda i: -i["screen"]["profit"])[:ELITES]
        for individual in alive:
            individual["screen_fitness"] = individual["screen"]["profit"]
        return alive, len(population)
    alive.sort(key=lambda i: -i["screen_fitness"])
    return alive[:keep], len(population) - len(alive)


def next_generation(survivors, rng, size):
    """Elitism plus mutated crossover. Parents are chosen by rank, not by
    fitness value, so one enormous lucky profit cannot dominate the pool."""
    children = [{"genome": dict(s["genome"])} for s in survivors[:ELITES]]
    while len(children) < size:
        a, b = rng.choice(survivors), rng.choice(survivors)
        children.append(
            {"genome": G.mutate(G.crossover(a["genome"], b["genome"], rng), rng)}
        )
    return children


def save(path, state):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial")
    temporary.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def evolve(executor, segments, rng, generations, size, out, resume=None):
    """Generations on the train segment only. Validation is looked at; test is
    not opened here at all."""
    window = 100
    state = resume or {"generation": 0, "history": [], "population": None}
    population = state["population"] or [
        {"genome": G.random_genome(rng)} for _ in range(size)
    ]
    for generation in range(state["generation"], generations):
        started = time.time()
        survivors, dropped = screen(executor, population, segments["train"], window)
        rows = evaluate_population(
            executor, survivors, segments["train"], FULL_OBSERVATIONS,
            FULL_STARTS, window,
        )
        for individual, row in zip(survivors, rows):
            individual["starts"] = row
            individual["fitness"] = fitness(row)
        survivors.sort(key=lambda i: -i["fitness"])
        best = survivors[0]
        state["history"].append({
            "generation": generation,
            "evaluated": len(population),
            "degenerate": dropped,
            "best_fitness": best["fitness"],
            "best_id": G.identity(best["genome"]),
            "median_fitness": median([i["fitness"] for i in survivors]),
            "seconds": round(time.time() - started, 1),
        })
        print(f"  generation {generation:3}  {len(population)} evaluated, "
              f"{dropped} degenerate, best {best['fitness']:+.4f}, median "
              f"{state['history'][-1]['median_fitness']:+.4f}  "
              f"({state['history'][-1]['seconds']:.0f}s)", flush=True)
        population = next_generation(survivors, rng, size)
        state["generation"] = generation + 1
        state["population"] = population
        state["survivors"] = survivors
        # Written every generation: an interrupted night is a paused run, not
        # a lost one.
        save(out / "population.json", state)
    return state


def judge(executor, champion, population, segments, rng, out, seed):
    """The single pass over the test segment, and the pre-declared verdict."""
    window = 100
    results = {}
    index = next(
        i for i, p in enumerate(population)
        if G.identity(p["genome"]) == G.identity(champion["genome"])
    )
    for name in ["validation", "test"]:
        prices = segments[name]
        offsets = starts(prices, FULL_STARTS, window, FULL_OBSERVATIONS + WARMUP)
        rows = evaluate_population(
            executor, population, prices, FULL_OBSERVATIONS, FULL_STARTS, window
        )
        base = [
            executor.submit(
                run_baselines, (prices, start, FULL_OBSERVATIONS, seed + i)
            )
            for i, start in enumerate(offsets)
        ]
        collected = [f.result() for f in base]
        results[name] = {
            "champion": fitness(rows[index]),
            # Without this a champion that never filled an order reads as a
            # flat 0.0000, which is indistinguishable from one that traded and
            # broke even. On a segment with negative expectancy, profit
            # fitness correctly selects doing nothing, and the report has to
            # say which of the two happened.
            "champion_activity": {
                "fills": sum(
                    r["fills"]["BUY"] + r["fills"]["SELL"] for r in rows[index]
                ),
                "proposals": {
                    k: sum(r[k] for r in rows[index]) for k in ["buy", "sell", "hold"]
                },
                "rejected": sum(r["rejected"] for r in rows[index]),
            },
            "population": sorted(fitness(r) for r in rows),
            "baselines": {
                b: median([c[b]["profit"] for c in collected]) for b in BASELINES
            },
            "starts": offsets,
            # Reported, never compared against: nothing beats perfect
            # foresight. It says how much was there to take at all.
            "perfect_foresight_ceiling": median(
                [ceiling(prices, o, FULL_OBSERVATIONS)["ceiling"] for o in offsets]
            ),
            "ceiling_detail": ceiling(prices, offsets[0], FULL_OBSERVATIONS),
        }
    verdict = decide(results["test"])
    if results["test"]["perfect_foresight_ceiling"] <= 0:
        verdict = (
            f"there was nothing to take: a trader with perfect foresight makes "
            f"{results['test']['perfect_foresight_ceiling']:+.4f} on this test "
            f"segment, so no policy can profit here and the champion's "
            f"{results['test']['champion']:+.4f} says nothing about the fly. "
            f"Sample at a coarser interval or evaluate over more observations "
            f"until the ceiling clears the "
            f"{results['test']['ceiling_detail']['round_trip_cost_percent']:.1f}% "
            f"round-trip cost. Original verdict: " + verdict
        )
    report = {
        "champion_id": G.identity(champion["genome"]),
        "champion_genome": G.describe(champion["genome"]),
        "train_fitness": champion.get("fitness"),
        "results": results,
        "protocol": {
            "warmup_observations": WARMUP,
            "observations_per_evaluation": FULL_OBSERVATIONS,
            "starts_per_evaluation": FULL_STARTS,
            "fitness": "median profit in quote currency over independent chronological starts",
            "test_segment_evaluations": 1,
            "kill_criterion": "A champion that does not beat every one of "
                              f"{', '.join(BASELINES)} on the test segment is "
                              "reported as noise.",
            "execution": "Explicit paper simulator with budget, inventory, order size and fee. NOT the production guard: no cooldown, spread, quote-age or STOP check. A champion is not a validated trading result until re-run through `python -m stonkfly run`.",
        },
        "verdict": verdict,
    }
    save(out / "champion.json", report)
    return report


def decide(test):
    """The criterion, applied. Written before the numbers existed."""
    beaten = {b: test["champion"] > v for b, v in test["baselines"].items()}
    table = ", ".join(
        f"{b} {v:+.4f}{' (beaten)' if beaten[b] else ''}"
        for b, v in test["baselines"].items()
    )
    activity = test.get("champion_activity", {})
    if activity and not activity["fills"]:
        return (f"the champion never filled an order: "
                f"{activity['proposals']['buy']} BUY, "
                f"{activity['proposals']['sell']} SELL, "
                f"{activity['proposals']['hold']} HOLD, all "
                f"{activity['rejected']} orders rejected for budget or "
                f"inventory. Its {test['champion']:+.4f} is the score for "
                f"doing nothing, not for trading well -- on a segment where "
                f"trading loses money, profit fitness selects inaction and "
                f"correctly so. Read the perfect-foresight ceiling before "
                f"reading this as a result")
    if all(beaten.values()):
        return (f"champion {test['champion']:+.4f} on the held-out test "
                f"segment, against {table}. It beat every declared baseline. "
                f"That is one pass over one segment with a search of fourteen "
                f"parameters behind it; it is a result to try to break, not a "
                f"trading strategy")
    missed = [b for b, ok in beaten.items() if not ok]
    return (f"champion {test['champion']:+.4f} on the held-out test segment, "
            f"against {table}. It failed to beat {', '.join(missed)}, so by "
            f"the criterion declared before this run it is NOISE: the search "
            f"found parameters that suited the training stretch of history and "
            f"nothing that survives out of sample")
