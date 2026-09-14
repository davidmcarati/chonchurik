"""The declared free parameters, and nothing else.

Wiring is never touched. AGENTS.md forbids pruning the retained graph, and
none of these parameters adds, removes, re-signs or re-routes an edge: they
are resting potentials, adaptation constants, gains, currents and one readout
threshold. Every one of them was already a documented model choice before any
evolution existed, and each is recorded with its bounds so a champion can be
read as "these values", not as "a fly that learned something".

Bounds are declared here, up front, and are not widened after seeing results.
"""

import hashlib
import json
import math

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


def clamp(name, value):
    low, high, _ = SPACE[name]
    return float(min(max(value, low), high))


def sample(rng, name):
    low, high, kind = SPACE[name]
    if kind == "log":
        return float(math.exp(rng.uniform(math.log(low), math.log(high))))
    return float(rng.uniform(low, high))


def random_genome(rng):
    return {name: sample(rng, name) for name in SPACE}


def mutate(genome, rng, rate=0.3, strength=0.25):
    """Perturb a minority of genes by a fraction of their declared range."""
    out = dict(genome)
    for name, (low, high, kind) in SPACE.items():
        if rng.random() >= rate:
            continue
        if kind == "log":
            out[name] = clamp(name, genome[name] * math.exp(rng.gauss(0, strength)))
        else:
            out[name] = clamp(name, genome[name] + rng.gauss(0, strength * (high - low)))
    return out


def crossover(a, b, rng):
    """Uniform crossover. No gene is linked to another, so nothing is preserved
    by taking contiguous blocks."""
    return {name: (a if rng.random() < 0.5 else b)[name] for name in SPACE}


def identity(genome):
    """Stable id for caching and for naming a champion in a report."""
    text = json.dumps({k: round(genome[k], 9) for k in sorted(SPACE)}, sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def describe(genome):
    """Genome against wild type, for a report a reader can check."""
    return {
        name: {
            "value": round(genome[name], 6),
            "wild_type": WILD_TYPE[name],
            "bounds": list(SPACE[name][:2]),
        }
        for name in SPACE
    }
