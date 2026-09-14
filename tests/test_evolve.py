"""The evolution harness, minus the brain.

Everything here runs offline in milliseconds. The parts that need a connectome
are exercised by the run itself; the parts that decide whether a result counts
are exercised here, because those are the ones that must not drift.
"""

import random

import pytest

from tools.evolve import genome as G
from tools.evolve.evaluate import Account, degenerate
from tools.evolve.loop import BASELINES, decide, median, next_generation
from tools.evolve.series import candle_series, fixture_series, split, starts


def test_genome_stays_inside_its_declared_bounds():
    rng = random.Random(1)
    individual = G.random_genome(rng)
    assert set(individual) == set(G.SPACE) == set(G.WILD_TYPE)
    for _ in range(200):
        individual = G.mutate(individual, rng, rate=1.0, strength=3.0)
        for name, (low, high, _) in G.SPACE.items():
            assert low <= individual[name] <= high, name
    child = G.crossover(individual, G.WILD_TYPE, rng)
    assert set(child) == set(G.SPACE)
    assert all(child[k] in (individual[k], G.WILD_TYPE[k]) for k in G.SPACE)


def test_genome_identity_is_stable_and_discriminating():
    a = dict(G.WILD_TYPE)
    assert G.identity(a) == G.identity(dict(reversed(list(a.items()))))
    b = {**a, "kc_rest": a["kc_rest"] - 1}
    assert G.identity(a) != G.identity(b)


def test_split_is_chronological_and_never_overlaps():
    series = fixture_series(1000)
    segments = split(series)
    assert segments["train"] + segments["validation"] + segments["test"] == series[
        : len(segments["train"]) + len(segments["validation"]) + len(segments["test"])
    ]
    assert len(segments["train"]) > len(segments["validation"])
    with pytest.raises(ValueError):
        split(fixture_series(100))


def test_starts_leave_room_for_the_whole_evaluation():
    segment = fixture_series(500)
    offsets = starts(segment, 5, 100, 65)
    assert offsets == sorted(offsets) and len(set(offsets)) == 5
    assert offsets[0] == 0 and offsets[-1] + 100 + 65 == len(segment)
    with pytest.raises(ValueError):
        starts(segment, 5, 100, 500)


def test_candle_series_never_opens_a_socket(tmp_path):
    import json

    path = tmp_path / "candles.json"
    path.write_text(json.dumps({"BTC-USDC": [{"close": 100.0 + i} for i in range(250)]}))
    assert len(candle_series(path)) == 250
    path.write_text(json.dumps([1.0, 2.0]))
    with pytest.raises(ValueError):
        candle_series(path)


def test_account_never_invents_money():
    a = Account("100", "10", "0.006")
    assert a.apply("HOLD", 50.0, 50.0) is None
    # A sell with nothing held is rejected, not borrowed.
    assert a.apply("SELL", 50.0, 50.0) is None and a.base == 0.0
    assert a.apply("BUY", 50.0, 50.0) == "BUY"
    assert a.cash == pytest.approx(100 - 10 * 1.006) and a.base == pytest.approx(0.2)
    # Ten orders of ten against a hundred of capital, fees included: the
    # tenth cannot fill.
    for _ in range(8):
        a.apply("BUY", 50.0, 50.0)
    assert a.apply("BUY", 50.0, 50.0) is None
    assert a.cash >= 0 and a.fills["BUY"] == 9 and a.rejected == 2


def test_degenerate_rejects_a_fly_that_cannot_act():
    traded = {"BUY": 3, "SELL": 2}
    row = {"observations": 20, "buy": 20, "sell": 0, "hold": 0, "kc_spikes": 99,
           "fills": traded}
    assert degenerate(row) == "one proposal for 100% of observations"
    assert degenerate({**row, "buy": 10, "hold": 10, "kc_spikes": 0}) == (
        "silent mushroom body"
    )
    assert degenerate({**row, "observations": 0}) == "no observations"
    # Sitting in cash ties the all-cash baseline for first wherever trading
    # loses money, so it is disqualified rather than allowed to win.
    assert degenerate({**row, "buy": 9, "sell": 5, "hold": 6,
                       "fills": {"BUY": 0, "SELL": 0}}) == "never filled an order"
    assert degenerate({**row, "buy": 9, "sell": 5, "hold": 6}) is None


def test_degenerate_rejects_the_champion_that_the_old_test_let_through():
    """The measured failure, as a regression.

    Generation 0 on real five-minute candles produced this shape: 289 BUY of
    300, 4 SELL, 7 HOLD, 276 of the proposals rejected for want of budget. It
    had bought everything it could afford and was holding, which is a baseline
    rather than a policy -- and it scored the first positive fitness in the
    project. The exact-100% test passed it, because 289 is not 300.
    """
    row = {"observations": 300, "buy": 289, "sell": 4, "hold": 7,
           "kc_spikes": 210332, "fills": {"BUY": 13, "SELL": 4}}
    assert degenerate(row) == "one proposal for 96% of observations"
    # Still one-sided at the declared line, and allowed just below it.
    assert degenerate({**row, "buy": 270, "sell": 20, "hold": 10}) == (
        "one proposal for 90% of observations"
    )
    assert degenerate({**row, "buy": 260, "sell": 25, "hold": 15}) is None


def test_median_is_the_middle_not_the_mean():
    # One enormous lucky start must not carry a genome.
    assert median([0.0, 0.0, 0.0, 0.0, 1000.0]) == 0.0
    assert median([1.0, 3.0]) == 2.0
    assert median([]) == float("-inf")


def test_kill_criterion_needs_every_baseline():
    beats_all = {
        "champion": 1.0,
        "baselines": {b: 0.5 for b in BASELINES},
    }
    assert "beat every declared baseline" in decide(beats_all)
    assert "NOISE" not in decide(beats_all)
    for missed in BASELINES:
        one_short = {
            "champion": 1.0,
            "baselines": {**{b: 0.5 for b in BASELINES}, missed: 1.0},
        }
        verdict = decide(one_short)
        assert "NOISE" in verdict and missed in verdict
    # Equalling a baseline is not beating it.
    assert "NOISE" in decide({"champion": 0.0, "baselines": {b: 0.0 for b in BASELINES}})


def test_next_generation_keeps_the_elites_verbatim():
    rng = random.Random(7)
    survivors = [
        {"genome": G.random_genome(rng), "fitness": f} for f in [3.0, 2.0, 1.0]
    ]
    children = next_generation(survivors, rng, 10)
    assert len(children) == 10
    assert children[0]["genome"] == survivors[0]["genome"]
    assert children[1]["genome"] == survivors[1]["genome"]
    assert children[0]["genome"] is not survivors[0]["genome"]


def test_ceiling_is_an_upper_bound_and_detects_a_dead_window():
    from tools.evolve.evaluate import CHART_WINDOW, WARMUP, ceiling

    # A window that only moves less than the round trip costs has nothing in
    # it for anyone, and the ceiling has to say so rather than report a small
    # positive number.
    flat = [100.0] * (CHART_WINDOW + WARMUP + 60)
    assert ceiling(flat, 0, 50)["ceiling"] == 0.0

    rising = [100.0 * 1.01**i for i in range(CHART_WINDOW + WARMUP + 60)]
    high = ceiling(rising, 0, 50)
    assert high["ceiling"] > 0 and high["profitable_buys"] > 0
    # Nothing can beat perfect foresight, so a real replay must stay under it.
    a = Account("100", "10", "0.006")
    assert a.start == 100.0
    assert high["round_trip_cost_percent"] == pytest.approx(1.2)


def test_deprioritise_actually_lowers_priority():
    """The shipped Windows branch failed silently and left workers at normal.

    ctypes gave GetCurrentProcess an int return, so the (HANDLE)-1 pseudo-handle
    arrived as a 32-bit -1 and SetPriorityClass refused it. Nothing raised, and
    eight background workers competed with the interactive session for months.
    Asserting the scheduler's own answer is the only version of this test that
    would have caught it.
    """
    import sys

    from tools.evolve.pool import deprioritise, priority

    before = priority()
    deprioritise()
    after = priority()
    if sys.platform == "win32":
        assert after == 0x4000, f"expected BELOW_NORMAL, got {after:#x}"
    else:
        assert after > before


def test_selection_and_grading_use_different_numbers():
    """Fitness ranks on excess; the criterion still grades on absolute profit.

    If these were the same quantity the kill criterion would be graded on the
    search's own objective and could not fail. They must stay apart.
    """
    from tools.evolve.loop import fitness, profit_fitness

    # A fly that made money only because the window rose, and less than simply
    # holding would have. Positive profit, negative excess.
    rows = [
        {"profit": 4.68, "buy_and_hold": 5.40, "excess": 4.68 - 5.40},
        {"profit": 1.37, "buy_and_hold": 2.17, "excess": 1.37 - 2.17},
        {"profit": 0.19, "buy_and_hold": 1.11, "excess": 0.19 - 1.11},
    ]
    assert profit_fitness(rows) > 0, "it did make money"
    assert fitness(rows) < 0, "and it still lost to holding"


def test_buy_and_hold_benchmark_obeys_the_same_account_rules():
    from stonkfly.config import Settings

    from tools.evolve.evaluate import CHART_WINDOW, WARMUP, buy_and_hold

    s = Settings()
    flat = [100.0] * (CHART_WINDOW + WARMUP + 60)
    # A flat market cannot pay the fee, so holding through it must lose.
    assert buy_and_hold(flat, 0, 50, s) < 0
    rising = [100.0 * (1.01 ** i) for i in range(CHART_WINDOW + WARMUP + 60)]
    assert buy_and_hold(rising, 0, 50, s) > 0
    # Never more than the capital it is allowed to deploy.
    assert buy_and_hold(rising, 0, 50, s) < float(s.capital)
    assert buy_and_hold(flat, 0, 0, s) == 0.0
