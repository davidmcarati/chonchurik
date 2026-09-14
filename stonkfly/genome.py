"""The declared free parameters of the model, and how to put them into a fly.

These fourteen numbers are the only things an evolution is allowed to change.
Wiring is never one of them: AGENTS.md forbids pruning the retained graph, and
none of these adds, removes, re-signs or re-routes an edge. They are resting
potentials, adaptation constants, gains, currents and one readout threshold,
each a documented model choice that existed before any search did, each with
the bounds it is searched inside.

The declaration lives here rather than in `tools/` because it is a statement
about the animal, not about the search, and because `apply` below is what a
live run uses to become a particular fly. `tools/evolve/genome.py` imports it
and adds the sampling, mutation and crossover a search needs.

A genome is a plain dict of name to float. `runs/<run>/champion-genome.json`
is exactly that and nothing else, so it can be handed to
`python -m stonkfly run --genome`.
"""

import json

# name -> (low, high, kind). "log" samples multiplicatively, which is right for
# a gain that can plausibly be half or double, and wrong for a voltage.
SPACE = {
    # The single parameter measured to move the whole chain in and out of
    # saturation. Bounded to the range where a graded regime exists at all.
    "inhibitory_gain": (1.3, 2.4, "linear"),
    "kc_rest": (-75.0, -50.0, "linear"),
    "adaptation_jump": (0.0, 40.0, "linear"),
    "adaptation_tau": (50.0, 600.0, "log"),
    "lamina_bias": (2.0, 24.0, "linear"),
    "eta": (0.0002, 0.005, "log"),
    "dan_baseline_hz": (0.0, 20.0, "linear"),
    "odor_current": (8.0, 60.0, "log"),
    "odor_sigma": (0.3, 1.6, "log"),
    "odor_floor": (0.05, 0.6, "log"),
    "satiety_floor": (0.0, 16.0, "linear"),
    "satiety_span": (4.0, 48.0, "log"),
    "pulse_current": (10.0, 120.0, "log"),
    "decoder_threshold_hz": (0.5, 12.0, "log"),
}
WILD_TYPE = {
    "inhibitory_gain": 1.9,
    "kc_rest": -60.0,
    "adaptation_jump": 8.0,
    "adaptation_tau": 200.0,
    "lamina_bias": 12.0,
    "eta": 0.001,
    "dan_baseline_hz": 0.0,
    "odor_current": 30.0,
    "odor_sigma": 0.9,
    "odor_floor": 0.1,
    "satiety_floor": 8.0,
    "satiety_span": 24.0,
    "pulse_current": 40.0,
    "decoder_threshold_hz": 2.0,
}

# `decoder_threshold_hz` is in the genome and does nothing. The Decoder reads
# its threshold when the controller is built and never consults Settings
# again, so every evolution run so far has searched a parameter it could not
# feel. `apply` reproduces that rather than repairing it: a champion run with
# a threshold its own selection never used is a different fly from the one
# that won, and the comparison the whole exercise rests on would be void.
# Repairing it means re-running the search, not changing this line.
INERT = ("decoder_threshold_hz",)


def load(path):
    """A genome from disk, checked against the declared bounds."""
    genome = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(genome, dict):
        raise ValueError(f"{path} is not a genome: expected an object of "
                         f"{len(SPACE)} numbers")
    missing = sorted(set(SPACE) - set(genome))
    extra = sorted(set(genome) - set(SPACE))
    if missing or extra:
        raise ValueError(
            f"{path} is not a genome for this model"
            + (f"; missing {', '.join(missing)}" if missing else "")
            + (f"; unknown {', '.join(extra)}" if extra else "")
        )
    for name, (low, high, _) in SPACE.items():
        value = genome[name]
        if not isinstance(value, (int, float)) or not low <= value <= high:
            raise ValueError(
                f"{path}: {name} is {value!r}, outside the declared bounds "
                f"[{low}, {high}]. Bounds are declared up front and are not "
                f"widened after seeing results."
            )
    return {name: float(genome[name]) for name in SPACE}


def apply(controller, genome, pristine=None):
    """Make an already-built fly into this particular fly.

    Every assignment here is one an evolution worker makes too; the search and
    a live run must produce the same animal from the same numbers, so there is
    one implementation and `tools/evolve/evaluate.configured` wraps this rather
    than repeating it.

    `pristine` is the inhibitory magnitudes as the graph was reconstructed,
    recovered once from a brain that has had the gain applied exactly once.
    Pass it whenever this is called more than once on the same brain: the gain
    multiplies those magnitudes, and re-deriving them by dividing out the
    *previous* genome's gain is a float32 round trip the value does not
    survive, so a worker reused across a generation would drift a little
    further with every fly it saw. Omitted is correct only for a brain
    straight out of the constructor, which is the live case.
    """
    brain = controller.brain
    kc = brain.circuit["kc"]
    if pristine is None:
        pristine = pristine_inhibitory(brain)
    brain.rest[kc] = genome["kc_rest"]
    brain.initial["v"][kc] = genome["kc_rest"]
    brain.adaptation_jump = genome["adaptation_jump"]
    brain.adaptation_tau = genome["adaptation_tau"]
    brain.eta = genome["eta"]
    brain.dan_baseline_hz[:] = genome["dan_baseline_hz"]
    # drive[lamina] = 12, then drive += tonic, so an offset is exactly an
    # altered bias without reaching into the step signature.
    brain.tonic[brain.lamina] = genome["lamina_bias"] - WILD_TYPE["lamina_bias"]
    brain.inhibitory_gain = genome["inhibitory_gain"]
    brain.weight[brain.inhibitory_edges] = pristine * genome["inhibitory_gain"]
    if controller.olfaction is not None:
        controller.olfaction.current = genome["odor_current"]
        controller.olfaction.sigma = genome["odor_sigma"]
        controller.olfaction.floor = genome["odor_floor"]
    if controller.gustation is not None:
        controller.gustation.floor = genome["satiety_floor"]
        controller.gustation.span = genome["satiety_span"]
    return {"applied": [k for k in SPACE if k not in INERT],
            "inert": list(INERT)}


def pristine_inhibitory(brain):
    """The inhibitory magnitudes with the current gain taken back out.

    Recovered once, from a brain that has had the gain applied exactly once at
    construction. Everything afterwards multiplies these, so the division
    happens once rather than every time a genome is applied.
    """
    return (brain.weight[brain.inhibitory_edges]
            / brain.inhibitory_gain).copy()
