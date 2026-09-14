import os

import numpy as np
import pandas as pd
import pytest

from stonkfly.config import Settings
from stonkfly.neural.controller import Decoder
from stonkfly.neural.rule import advance
from stonkfly.reinforcement import reinforcement


def test_fixed_neuron_decoder():
    a = pd.DataFrame(
        {"type": ["DNp20", "DNp20", "DNpe017"], "somaSide": ["L", "R", "L"]}
    )
    d = Decoder(np.array([1, 2, 3]), a, 2)
    assert d.decode(np.array([0, 10, 0]), 0.5)["side"] == "HOLD"
    assert d.decode(np.array([0, 10, 1]), 0.5)["side"] == "BUY"
    assert d.decode(np.array([10, 0, 1]), 0.5)["side"] == "SELL"
    assert d.decode(np.array([4, 4, 1]), 0.5)["side"] == "HOLD"


@pytest.mark.parametrize(
    "equity,expected",
    [("100.03", "reward"), ("99.97", "aversive"), ("100.001", "none"), ("100", "none")],
)
def test_explicit_feedback(equity, expected):
    assert reinforcement(equity, "100", ".01")[0] == expected


def trace_protocol(order, frozen=False):
    k = np.zeros(2)
    d = np.zeros(1)
    u = np.zeros(2)
    w = np.zeros(2)
    gain = np.ones((1, 2))
    for phase in order:
        for _ in range(20):
            kh = np.array([20.0, 0.0]) if phase == "cue" else np.zeros(2)
            dh = np.array([30.0]) if phase == "reinforce" else np.zeros(1)
            advance(k, d, u, w, kh, dh, gain, 0.01, 0.001, frozen=frozen)
    return w


def test_memory_rule_temporal_specificity():
    paired = trace_protocol(["cue", "reinforce"])
    reverse = trace_protocol(["reinforce", "cue"])
    assert paired[0] < 0 and reverse[0] > 0
    assert paired[1] == 0 and reverse[1] == 0  # Unactivated input is unchanged.
    assert np.array_equal(trace_protocol(["cue", "reinforce"], True), np.zeros(2))


@pytest.mark.skipif(
    os.environ.get("STONKFLY_FULL_TEST") != "1",
    reason="Downloads/uses full MaleCNS; explicit integration test",
)
def test_full_graph_sensory_reinforcement_checkpoint(tmp_path):
    from stonkfly.data import verify
    from stonkfly.neural.controller import FlyController

    assert verify()["neurons"] == 166700
    c = FlyController(Settings())
    assert len(c.brain.post) == 25582938 and len(c.brain.circuit["edges"]) == 7835
    assert len(c.brain.retina) == 3335 and len(c.brain.r8) == 811
    from stonkfly.display import market_frame

    # The mushroom body is reached through the olfactory channel, so that is
    # what this drives. A white field is kept as an explicit control because
    # it used to be the whole test: at the reconstructed excitation/inhibition
    # ratio it activated a third of the Kenyon population, which looked like
    # sensory drive and was saturation. It no longer reaches memory, and that
    # is a measured consequence of the inhibitory gain, not a regression.
    history = [100.0 * 1.002**i for i in range(80)]
    chart = market_frame("BTC-USDC", history, 100.0, 100.1)
    white = np.full((180, 320, 3), 255, np.uint8)
    c.brain.reset()
    assert c.observe(white, "none")["KC_spikes"] < 100
    for _ in range(3):
        c.observe(chart, "none", history)
    c.save(tmp_path / "before.npz")
    before = c.brain.weight[c.brain.circuit["edges"]].copy()
    reward = c.observe(chart, "reward", history)
    assert reward["reward_spikes"] > 0 and reward["stimulus_ms"] == 200
    assert reward["ORN_spikes"] > 0 and reward["sugar_spikes"] == 0
    assert reward["KC_spikes"] > 0 and reward["memory"]["changed_edges"] > 0
    reward_weights = c.brain.weight[c.brain.circuit["edges"]].copy()
    c.restore(tmp_path / "before.npz")
    control = c.observe(chart, "none", history)
    assert not np.array_equal(reward_weights, c.brain.weight[c.brain.circuit["edges"]])
    c.restore(tmp_path / "before.npz")
    c.brain.weights_frozen = True
    c.observe(chart, "reward", history)
    assert np.array_equal(before, c.brain.weight[c.brain.circuit["edges"]])
    c.brain.weights_frozen = False
    c.restore(tmp_path / "before.npz")
    loss = c.observe(chart, "aversive", history)
    assert loss["aversive_spikes"] > 0 and loss["stimulus_ms"] == 200
    assert np.isfinite(c.brain.weight).all()
    # The sugar channel only opens when an account is supplied.
    c.restore(tmp_path / "before.npz")
    tasted = c.observe(chart, "none", history, None, ("105", "100"))
    assert tasted["satiety"] > 0.9 and tasted["sugar_spikes"] > 0
    print({"reward": reward, "unpaired_control": control, "loss": loss})


def odor_annotation(cells_per_glomerulus=3, glomeruli=53):
    names = [f"ORN_G{i:02d}" for i in range(glomeruli)]
    return pd.DataFrame(
        {"type": [n for n in names for _ in range(cells_per_glomerulus)] + ["KC"]}
    )


def test_olfactory_features_are_bounded_and_json_safe():
    import json

    from stonkfly.neural.olfaction import FEATURES, features

    flat = features([100.0] * 80)
    assert flat["volatility"] == 0.0
    # Volatility normalisation must make the channels independent of the
    # sampling interval: the same shape of market, scaled up, reads the same.
    slow = [100.0 * 1.001**i for i in range(80)]
    fast = [100.0 * 1.010**i for i in range(80)]
    a, b = features(slow), features(fast)
    for key in ["trend_fast", "trend_slow", "range_position"]:
        assert a[key] == pytest.approx(b[key], abs=0.02), key
    for key in ["trend_fast", "trend_slow", "range_position", "acceleration"]:
        assert flat[key] == pytest.approx(0.5)
    for history in [[], [100.0], [100.0, 101.0], [100.0 * 1.01**i for i in range(80)]]:
        values = features(history)
        assert set(values) == set(FEATURES)
        # events.jsonl and the ledger both refuse NaN and numpy scalars.
        assert json.loads(json.dumps(values, allow_nan=False)) == values
        assert all(0.0 <= v <= 1.0 for v in values.values())


def test_olfactory_code_is_sparse_separable_and_free_of_repeated_indices():
    from stonkfly.neural.olfaction import (BARS, CHANNELS, Olfaction,
                                           features)

    o = Olfaction(odor_annotation())
    assert len(o.names) == 53 and len(o.indices) == 159
    # drive[ix] += amplitude is plain fancy indexing: a repeated index would
    # overwrite instead of summing, silently losing one feature's drive.
    assert len(np.unique(o.indices)) == len(o.indices)
    assert sum(len(b) for b in o.bands) == len(o.names)
    assert len({int(i) for b in o.bands for i in b}) == len(o.names)

    rising = [100.0 * 1.002**i for i in range(80)]
    falling = [100.0 * 0.998**i for i in range(80)]
    codes = [o.activation(features(h)) > 0 for h in [rising, falling]]
    for code in codes:
        assert 0 < code.sum() <= 4 * len(CHANNELS)
    assert not np.array_equal(*codes)
    # What actually bounds the footprint is the tuning width against the
    # floor, not the number of channels: a Gaussian of sigma s clears a floor
    # f within s*sqrt(-2 ln f) glomeruli, which at 0.9 and 0.1 is 1.93 -- so
    # three whole glomeruli, or four when the peak falls between two. Adding a
    # channel narrows every band and must not widen any peak.
    for code in codes:
        for band in o.bands:
            assert code[band].sum() <= 4
    pulse, values = o.stimulation(rising)
    indices, current = pulse
    assert len(indices) == len(current) and current.max() <= o.current
    # Without klines the whole-bar channels are absent rather than neutral, so
    # a reader can tell "no data" from "this bar read one half".
    assert set(values) == set(CHANNELS) - set(BARS)
    # The trade tag is an efference copy: it must move the code on its own,
    # with the market held exactly still.
    bought = o.activation(o.stimulation(rising, "BUY")[1]) > 0
    sold = o.activation(o.stimulation(rising, "SELL")[1]) > 0
    held = o.activation(o.stimulation(rising, "HOLD")[1]) > 0
    assert not np.array_equal(bought, sold)
    assert np.array_equal(held, o.activation(o.stimulation(rising)[1]) > 0)
    with pytest.raises(ValueError):
        o.stimulation(rising, "LONG")

    # Klines fill the remaining three, and they move the code with the price
    # history held exactly still -- which is the whole reason they exist.
    def bar(close, taker):
        return {"high": close * 1.002, "low": close * 0.998, "close": close,
                "volume": 100.0, "taker_buy_base": taker}

    bought = [bar(c, 85.0) for c in rising]
    sold = [bar(c, 15.0) for c in rising]
    assert set(o.stimulation(rising, None, bought)[1]) == set(CHANNELS)
    aggressive = o.activation(o.stimulation(rising, None, bought)[1]) > 0
    passive = o.activation(o.stimulation(rising, None, sold)[1]) > 0
    assert not np.array_equal(aggressive, passive)
    # And a bar channel only ever touches its own band.
    plain = o.activation(o.stimulation(rising)[1]) > 0
    outside = np.concatenate([b for b, name in zip(o.bands, CHANNELS)
                              if name not in BARS])
    assert np.array_equal(aggressive[outside], plain[outside])


def test_satiety_is_bounded_and_rests_at_half():
    from stonkfly.neural.gustation import Gustation, satiety

    assert satiety("100", "100") == pytest.approx(0.5)
    assert satiety("105", "100") > 0.9
    assert satiety("95", "100") < 0.1
    # A missing, zeroed or unparseable reference must read as resting, not as
    # maximal loss: the sugar channel would otherwise invent a starving fly on
    # the first observation of every run.
    for bad in [None, "0", "", "abc", float("nan")]:
        assert satiety("100", bad) == 0.5
        assert satiety(bad, "100") == 0.5 or bad == "0"
    # An account that really is empty is a real total loss, not missing data.
    assert satiety("0", "100") == 0.0

    g = Gustation(np.array([3, 7, 11], dtype=np.int32))
    (indices, current), value = g.stimulation("100", "100")
    assert list(indices) == [3, 7, 11]
    assert float(current) == pytest.approx(g.floor + g.span * 0.5) and value == 0.5
    # Graded in BOTH directions: a loss has to be a smaller current than
    # resting, not the same silence as a bigger loss.
    drives = [
        float(g.stimulation(e, "100")[0][1]) for e in ["95", "98", "100", "102", "105"]
    ]
    assert drives == sorted(drives) and drives[0] < drives[2] < drives[-1]
    assert min(drives) >= g.floor and max(drives) <= g.floor + g.span
    with pytest.raises(RuntimeError):
        Gustation(np.array([3, 3], dtype=np.int32))
