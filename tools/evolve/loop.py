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

DOT = "\u00b7"
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
    """The readout's information coefficient, median over independent starts.

    Selection only. The pre-declared kill criterion is unchanged and still
    compares absolute profit against the four baselines; see profit_fitness.

    Two objectives were tried before this one and both selected something
    nobody asked for.

    Absolute profit came first. In a window where price rises the
    profit-maximising policy is maximum exposure, so the search converged on
    being buy-and-hold minus fees, and the criterion then asked that same fly
    to beat buy-and-hold.

    Excess over buy-and-hold came second, and it removed the drift but not the
    fees. A round trip costs the fee twice, so at chance-level direction every
    trade loses, and fitness became a function of how little a fly traded:
    correlation -0.97 with the number of sells across the surviving
    population, and a champion proposing BUY at 94 to 100 observations of 100
    with two of its five starts scoring an excess of exactly zero, because it
    had bought everything it could afford and was holding.

    Underneath both was something neither could see. The readout carries a
    constant that differs by genome -- -17.15 Hz for the wild type against
    +9.05, +8.60 and +4.42 for three evolved ones, while the spread stayed
    between 4.76 and 6.34 for every one of them. Three generations moved that
    constant 26 Hz and left the variation untouched. The search was selecting
    the sign of an offset.

    An information coefficient demeans the signal, so a constant contributes
    exactly nothing to it whatever its size or sign, and what is left is the
    part that moves. It is also free of the fee, which is what made the money
    objectives measure trade count instead of skill. Whether the thing it
    selects can then pay for itself is a separate question, asked once, at the
    end, on money, by the kill criterion.
    """
    return median([r["readout_ic"] for r in rows])


def excess_fitness(rows):
    """Profit over buying and holding. Reported, no longer selected on."""
    return median([r["excess"] for r in rows])


def profit_fitness(rows):
    """Absolute profit, median over starts. What the baselines are judged on."""
    return median([r["profit"] for r in rows])


def median_row(rows):
    """The one evaluation sitting at the median of excess.

    Excess and not fitness: this line exists so a reader can see a real start's
    profit and benchmark subtract to the excess printed beside them, and the
    fitness is now a correlation that does not subtract from anything.

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


def evaluate_population(runner, population, prices, observations, count,
                        window, bars=None):
    """Every genome at every start. A runner is the pool or one GPU."""
    # The warm-up is consumed from the same segment, so a start that leaves
    # room only for the scored observations would silently score a short run.
    offsets = starts(prices, count, window, observations + WARMUP)
    return runner.evaluate([i["genome"] for i in population], prices, offsets,
                           observations, bars)


def screen(runner, population, prices, window, observations, bars=None):
    """Cheap pass that removes flies that cannot act before paying for them."""
    rows = evaluate_population(
        runner, population, prices, screen_length(observations),
        SCREEN_STARTS, window, bars,
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
    kept = alive[:keep]
    # An elite has already been evaluated at every start and carried forward
    # unchanged, so screening it again decides its fate on one start of a
    # quarter the length -- the noisiest measurement in the run. On the first
    # fixture run that dropped the previous generation's best and the reported
    # fitness went backwards, which elitism exists to make impossible: a
    # deterministic kernel re-scores an unchanged genome to the same number,
    # so a fall can only mean the genome is gone.
    # From the whole population, not from `alive`: the same short window
    # can also call an elite degenerate, and a genome that filled no order
    # over twenty observations is not the same claim as one that filled
    # none over a hundred at five starts, which is what it already passed.
    promoted = [i for i in population if i.get("elite") and i not in kept]
    return kept + promoted, len(population) - len(alive)


def next_generation(survivors, rng, size):
    """Elitism plus mutated crossover. Parents are chosen by rank, not by
    fitness value, so one enormous lucky profit cannot dominate the pool."""
    children = [{"genome": dict(s["genome"]), "elite": True}
                for s in survivors[:ELITES]]
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


def holdout(runner, survivors, segments, observations, bars, window):
    """The elites, re-scored on a segment the search never selects on.

    The instrument against luck, and the reason it runs every generation
    rather than once at the end. Twice now a number rose convincingly on train
    and was nothing out of sample: a champion whose fitness climbed from
    -0.5171 to -0.0680 over three generations and turned out to be
    buy-and-hold, and a readout that reached an information coefficient of
    0.2408 with z 3.18 on train and 0.049 with z 0.55 on validation. Both took
    hours to find out. Two curves printed side by side say it in the second
    generation: if they move together it is an edge, and if they separate it
    is luck being fitted.

    Only the elites, because it costs a wave and it is a diagnostic rather
    than a selection -- nothing here changes who breeds.
    """
    if not survivors or "validation" not in segments:
        return None
    rows = evaluate_population(
        runner, survivors[:ELITES], segments["validation"], observations,
        FULL_STARTS, window, None if bars is None else bars["validation"],
    )
    return [fitness(row) for row in rows]


def evolve(runner, segments, rng, generations, size, out, resume=None,
           observations=FULL_OBSERVATIONS, bars=None):
    """Generations on the train segment only. Validation is looked at once a
    generation and never selected on; test is not opened here at all."""
    window = 100
    state = resume or {"generation": 0, "history": [], "population": None}
    population = state["population"] or [
        {"genome": G.random_genome(rng)} for _ in range(size)
    ]
    for generation in range(state["generation"], generations):
        started = time.time()
        runner.announce(f"generation {generation} {DOT} screening")
        survivors, dropped = screen(
            runner, population, segments["train"], window, observations,
            None if bars is None else bars["train"],
        )
        runner.announce(f"generation {generation} {DOT} full evaluation")
        rows = evaluate_population(
            runner, survivors, segments["train"], observations,
            FULL_STARTS, window, None if bars is None else bars["train"],
        )
        for individual, row in zip(survivors, rows):
            individual["starts"] = row
            individual["fitness"] = fitness(row)
            individual["excess"] = excess_fitness(row)
            individual["profit"] = profit_fitness(row)
            individual["buy_and_hold"] = median([r["buy_and_hold"] for r in row])
            individual["median_start"] = median_row(row)
        survivors.sort(key=lambda i: -i["fitness"])
        best = survivors[0]
        runner.announce(f"generation {generation} {DOT} validation")
        held = holdout(runner, survivors, segments, observations, bars, window)
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
            "best_excess": best["excess"],
            "median_fitness": median([i["fitness"] for i in survivors]),
            # The same elites on a segment nothing here selects on. Read the
            # two side by side: together is an edge, apart is luck.
            "holdout_fitness": held,
            "seconds": round(time.time() - started, 1),
        })
        mid = best["median_start"]
        out_of_sample = (f"{max(held):+.4f}" if held else "  --  ")
        print(f"  generation {generation:3}  {len(population)} evaluated, "
              f"{dropped} degenerate, best ic {best['fitness']:+.4f} train / "
              f"{out_of_sample} holdout, population median "
              f"{state['history'][-1]['median_fitness']:+.4f}, excess "
              f"{best['excess']:+.4f} (at that start: profit "
              f"{mid['profit']:+.4f}, buy+hold {mid['buy_and_hold']:+.4f})  "
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
          source="fixture", observations=FULL_OBSERVATIONS, bars=None):
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
        runner.announce(f"{name} {DOT} the single pass")
        rows = evaluate_population(
            runner, population, prices, observations, FULL_STARTS, window,
            None if bars is None else bars[name],
        )
        runner.announce(f"{name} {DOT} baselines")
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
    # The report describes the champion; this *is* the champion. Fourteen
    # numbers, nothing else, the shape `stonkfly.genome.load` takes -- so the
    # thing twelve generations were spent finding can be run:
    #     python -m stonkfly run --genome runs/<run>/champion-genome.json
    save(out / "champion-genome.json",
         {k: champion["genome"][k] for k in sorted(G.SPACE)})
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
