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
    white = np.full((180, 320, 3), 255, np.uint8)
    for _ in range(3):
        c.observe(white, "none")
    c.save(tmp_path / "before.npz")
    before = c.brain.weight[c.brain.circuit["edges"]].copy()
    reward = c.observe(white, "reward")
    assert reward["reward_spikes"] > 0 and reward["stimulus_ms"] == 200
    assert reward["KC_spikes"] > 0 and reward["memory"]["changed_edges"] > 0
    reward_weights = c.brain.weight[c.brain.circuit["edges"]].copy()
    c.restore(tmp_path / "before.npz")
    control = c.observe(white, "none")
    assert not np.array_equal(reward_weights, c.brain.weight[c.brain.circuit["edges"]])
    c.restore(tmp_path / "before.npz")
    c.brain.weights_frozen = True
    c.observe(white, "reward")
    assert np.array_equal(before, c.brain.weight[c.brain.circuit["edges"]])
    c.restore(tmp_path / "before.npz")
    loss = c.observe(white, "aversive")
    assert loss["aversive_spikes"] > 0 and loss["stimulus_ms"] == 200
    assert np.isfinite(c.brain.weight).all()
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
    for key in ["trend_fast", "trend_slow", "range_position", "acceleration"]:
        assert flat[key] == pytest.approx(0.5)
    for history in [[], [100.0], [100.0, 101.0], [100.0 * 1.01**i for i in range(80)]]:
        values = features(history)
        assert set(values) == set(FEATURES)
        # events.jsonl and the ledger both refuse NaN and numpy scalars.
        assert json.loads(json.dumps(values, allow_nan=False)) == values
        assert all(0.0 <= v <= 1.0 for v in values.values())


def test_olfactory_code_is_sparse_separable_and_free_of_repeated_indices():
    from stonkfly.neural.olfaction import FEATURES, Olfaction, features

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
        assert 0 < code.sum() <= 3 * len(FEATURES)
    assert not np.array_equal(*codes)
    pulse, values = o.stimulation(rising)
    indices, current = pulse
    assert len(indices) == len(current) and current.max() <= o.current
    assert set(values) == set(FEATURES)
