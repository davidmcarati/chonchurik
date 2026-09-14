"""A champion has to be loadable, and has to arrive intact.

Twelve generations of search are worth nothing if the winner cannot be put
back into a fly. `runs/<run>/champion-genome.json` is the fourteen numbers and
nothing else; this checks that the file round-trips, that a file which is not
one is rejected loudly rather than half-applied, and that applying it actually
moves the fly.
"""

import json

import pytest

from stonkfly.genome import INERT, SPACE, WILD_TYPE, apply, load


def write(tmp_path, genome):
    path = tmp_path / "genome.json"
    path.write_text(json.dumps(genome), encoding="utf-8")
    return path


def test_the_wild_type_round_trips(tmp_path):
    assert load(write(tmp_path, dict(WILD_TYPE))) == {
        k: float(v) for k, v in WILD_TYPE.items()
    }


def test_every_declared_parameter_has_a_wild_type_and_bounds():
    assert set(SPACE) == set(WILD_TYPE)
    for name, (low, high, kind) in SPACE.items():
        assert low < high, name
        assert kind in ("linear", "log"), name
        assert low <= WILD_TYPE[name] <= high, name


def test_a_missing_parameter_is_refused_by_name(tmp_path):
    genome = dict(WILD_TYPE)
    del genome["eta"]
    with pytest.raises(ValueError, match="missing eta"):
        load(write(tmp_path, genome))


def test_an_unknown_parameter_is_refused_by_name(tmp_path):
    genome = {**WILD_TYPE, "wingspan": 1.0}
    with pytest.raises(ValueError, match="unknown wingspan"):
        load(write(tmp_path, genome))


def test_a_value_outside_the_declared_bounds_is_refused(tmp_path):
    genome = {**WILD_TYPE, "inhibitory_gain": 9.0}
    with pytest.raises(ValueError, match="outside the declared bounds"):
        load(write(tmp_path, genome))


def test_something_that_is_not_an_object_is_refused(tmp_path):
    path = tmp_path / "genome.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(ValueError, match="not a genome"):
        load(path)


def test_the_dead_gene_is_named_as_dead_rather_than_hidden():
    # It is in the genome and the Decoder never reads it. Repairing that means
    # re-running the search; pretending otherwise would be worse than either.
    assert "decoder_threshold_hz" in SPACE
    assert "decoder_threshold_hz" in INERT


class Brain:
    """Just enough of one to see whether the assignments land."""

    def __init__(self):
        import numpy as np

        self.circuit = {"kc": np.array([0, 1])}
        self.rest = np.full(4, -52.0, np.float32)
        self.initial = {"v": np.full(4, -52.0, np.float32)}
        self.tonic = np.zeros(4, np.float32)
        self.lamina = np.array([2, 3])
        self.dan_baseline_hz = np.zeros(2)
        self.inhibitory_edges = np.array([0, 1])
        self.weight = np.array([-1.0, -2.0, 3.0, 4.0], np.float32)
        self.inhibitory_gain = 1.0
        self.adaptation_jump = self.adaptation_tau = self.eta = None


class Channel:
    pass


class Controller:
    def __init__(self):
        self.brain = Brain()
        self.olfaction = Channel()
        self.gustation = Channel()


def test_applying_a_genome_moves_the_fly():
    import numpy as np

    c = Controller()
    pristine = c.brain.weight[c.brain.inhibitory_edges].copy()
    genome = {**WILD_TYPE, "kc_rest": -70.0, "lamina_bias": 20.0,
              "inhibitory_gain": 2.0, "eta": 0.003, "odor_sigma": 1.1}
    report = apply(c, genome, pristine)

    assert list(c.brain.rest) == [-70.0, -70.0, -52.0, -52.0]
    assert list(c.brain.initial["v"]) == [-70.0, -70.0, -52.0, -52.0]
    # drive[lamina] is set to the wild-type bias and then tonic is added, so
    # the offset is what carries the genome's value.
    assert list(c.brain.tonic) == [0.0, 0.0, 8.0, 8.0]
    assert c.brain.eta == 0.003
    assert c.brain.inhibitory_gain == 2.0
    assert np.array_equal(c.brain.weight[:2], pristine * 2.0)
    assert c.brain.weight[2] == 3.0 and c.brain.weight[3] == 4.0
    assert c.olfaction.sigma == 1.1
    assert "decoder_threshold_hz" not in report["applied"]


def test_applying_twice_does_not_compound_the_gain():
    # The failure this guards is silent: without the pristine magnitudes,
    # the second apply divides out the *first* genome's gain, and float32 does
    # not survive that round trip.
    import numpy as np

    c = Controller()
    pristine = c.brain.weight[c.brain.inhibitory_edges].copy()
    for gain in (2.0, 1.5, 2.4, 1.3):
        apply(c, {**WILD_TYPE, "inhibitory_gain": gain}, pristine)
        assert np.array_equal(c.brain.weight[:2], pristine * np.float32(gain))
