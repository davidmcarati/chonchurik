"""The evolution loop, its screening tier, and the protections against itself.

With fourteen free parameters and dozens of generations, a profitable fly will
be found. That is not in question and it is not evidence of anything. Every
rule below exists to keep the difference visible between "a combination of
parameters suited this stretch of history" and "the fly learned to trade":

  - the test segment is evaluated exactly once, at the very end;
  - fitness is the median over independent chronological starts, so one lucky
    start cannot carry a genome;
  - fitness is profit *over buying and holding the same window*, so a rising
    market cannot be mistaken for skill, while the kill criterion stays on
    absolute profit against the baselines -- the two must not be the same
    number, or the search would be graded by the thing it optimises;
  - four baselines are run on the same segments and the same account rules;
  - the kill criterion is declared here, in code, before any run;
  - the whole final population is reported out of sample, not just the winner.
"""

import json
import time
from pathlib import Path

from . import genome as G
from .evaluate import WARMUP, ceiling, degenerate
from .series import starts

SCREEN_STARTS, FULL_STARTS = 1, 5
FULL_OBSERVATIONS = 50


def screen_length(observations):
    """Long enough to be a filter, short enough to be worth skipping ahead.

    A quarter of the full evaluation. Screening a hold-based fly over twenty
    observations would rank the survivors on noise: `tools.evolve.survey`
    reports the median number of bars before price moves far enough to cover
    the round trip, and an evaluation shorter than that has no room for a
    single trade whatever the sampling interval.
    """
    return max(20, observations // 4)
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
    """Profit over buying and holding, median over independent starts.

    Selection only. The pre-declared kill criterion is unchanged and still
    compares absolute profit against the four baselines; see profit_fitness.

    Absolute profit was the original objective and it was the wrong one. In a
    window where price rises, the profit-maximising policy is maximum
    exposure, so the search converges on being buy-and-hold minus fees -- and
    the criterion then asks that same fly to beat buy-and-hold. Generation 0 on
    real candles reached +0.1866 absolute and lost to the benchmark at
    all five starts. Subtracting the benchmark takes the market's drift out of
    what is selected and leaves the timing.
    """
    return median([r["excess"] for r in rows])


def profit_fitness(rows):
    """Absolute profit, median over starts. What the baselines are judged on."""
    return median([r["profit"] for r in rows])


def median_row(rows):
    """The one evaluation sitting at the median of excess.

    Reported instead of three separately-taken medians. Those are each correct
    and they do not subtract: on the first run of this, median profit was
    -1.5711 and median buy-and-hold -0.3698, a difference of -1.2013, while
    the median excess was -0.3503 -- three different starts supplying the
    three middles. A reader subtracting the printed numbers would conclude the
    arithmetic was broken. One real start's profit and benchmark do subtract,
    exactly, to the fitness beside them.

    With an even number of starts `median` averages the middle two and no
    single row is the median; the upper middle is reported and the fitness is
    then the only exact number on the line.
    """
    return sorted(rows, key=lambda r: r["excess"])[len(rows) // 2]


def evaluate_population(runner, population, prices, observations, count, window):
    """Every genome at every start. A runner is the pool or one GPU."""
    # The warm-up is consumed from the same segment, so a start that leaves
    # room only for the scored observations would silently score a short run.
    offsets = starts(prices, count, window, observations + WARMUP)
    return runner.evaluate([i["genome"] for i in population], prices, offsets,
                           observations)


def screen(runner, population, prices, window, observations):
    """Cheap pass that removes flies that cannot act before paying for them."""
    rows = evaluate_population(
        runner, population, prices, screen_length(observations),
        SCREEN_STARTS, window,
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
        alive = sorted(population, key=lambda i: -i["screen"]["excess"])[:ELITES]
        for individual in alive:
            individual["screen_fitness"] = individual["screen"]["excess"]
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


def evolve(runner, segments, rng, generations, size, out, resume=None,
           observations=FULL_OBSERVATIONS):
    """Generations on the train segment only. Validation is looked at; test is
    not opened here at all."""
    window = 100
    state = resume or {"generation": 0, "history": [], "population": None}
    population = state["population"] or [
        {"genome": G.random_genome(rng)} for _ in range(size)
    ]
    for generation in range(state["generation"], generations):
        started = time.time()
        survivors, dropped = screen(
            runner, population, segments["train"], window, observations
        )
        rows = evaluate_population(
            runner, survivors, segments["train"], observations,
            FULL_STARTS, window,
        )
        for individual, row in zip(survivors, rows):
            individual["starts"] = row
            individual["fitness"] = fitness(row)
            individual["profit"] = profit_fitness(row)
            individual["buy_and_hold"] = median([r["buy_and_hold"] for r in row])
            individual["median_start"] = median_row(row)
        survivors.sort(key=lambda i: -i["fitness"])
        best = survivors[0]
        state["history"].append({
            "generation": generation,
            "evaluated": len(population),
            "degenerate": dropped,
            "best_fitness": best["fitness"],
            "best_profit": best["profit"],
            "best_buy_and_hold": best["buy_and_hold"],
            "best_median_start": {
                k: best["median_start"][k]
                for k in ["profit", "buy_and_hold", "excess"]
            },
            "best_id": G.identity(best["genome"]),
            "median_fitness": median([i["fitness"] for i in survivors]),
            "seconds": round(time.time() - started, 1),
        })
        mid = best["median_start"]
        print(f"  generation {generation:3}  {len(population)} evaluated, "
              f"{dropped} degenerate, best {best['fitness']:+.4f} over "
              f"benchmark (at that start: profit {mid['profit']:+.4f}, "
              f"buy+hold {mid['buy_and_hold']:+.4f}), population median "
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


def judge(runner, champion, population, segments, rng, out, seed,
          source="fixture", observations=FULL_OBSERVATIONS):
    """The single pass over the test segment, and the pre-declared verdict."""
    window = 100
    results = {}
    index = next(
        i for i, p in enumerate(population)
        if G.identity(p["genome"]) == G.identity(champion["genome"])
    )
    for name in ["validation", "test"]:
        prices = segments[name]
        offsets = starts(prices, FULL_STARTS, window, observations + WARMUP)
        rows = evaluate_population(
            runner, population, prices, observations, FULL_STARTS, window
        )
        collected = runner.baselines(prices, offsets, observations, seed)
        results[name] = {
            # Absolute profit: the only thing comparable to the baselines, and
            # what the criterion below is applied to. Selection used `excess`;
            # grading must not, or the search would be graded by its own
            # objective.
            "champion": profit_fitness(rows[index]),
            "champion_excess": fitness(rows[index]),
            "champion_buy_and_hold": median(
                [r["buy_and_hold"] for r in rows[index]]
            ),
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
            "population": sorted(profit_fitness(r) for r in rows),
            "population_excess": sorted(fitness(r) for r in rows),
            "baselines": {
                b: median([c[b]["profit"] for c in collected]) for b in BASELINES
            },
            "starts": offsets,
            # Reported, never compared against: nothing beats perfect
            # foresight. It says how much was there to take at all.
            "perfect_foresight_ceiling": median(
                [ceiling(prices, o, observations)["ceiling"] for o in offsets]
            ),
            "ceiling_detail": ceiling(prices, offsets[0], observations),
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
            # The fly was evolved on candles at this spacing. A live run
            # sampling at a different interval shows it a different world, so
            # Settings.interval_seconds has to match what is recorded here.
            "source": source,
            "warmup_observations": WARMUP,
            "observations_per_evaluation": observations,
            "screen_observations": screen_length(observations),
            "starts_per_evaluation": FULL_STARTS,
            "fitness": "SELECTION: median of (profit - buy_and_hold on the "
                       "same window) in quote currency, over independent "
                       "chronological starts. GRADING: absolute median profit "
                       "against the four baselines. These are deliberately "
                       "different quantities; grading on the selected quantity "
                       "would make the criterion unfalsifiable.",
            "fitness_changed": "Generation 0 of the first run on real "
                               "candles "
                               "selected on absolute profit and produced a "
                               "champion at +0.1866 that proposed BUY at 96% "
                               "of observations and lost to buy-and-hold at "
                               "5 of 5 starts. In a rising window, maximum "
                               "exposure maximises profit, so the objective "
                               "was pushing the population towards the "
                               "baseline the criterion requires beating. The "
                               "criterion did not move; the objective did.",
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
