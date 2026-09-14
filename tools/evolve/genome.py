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

# Declared in the model, not here: these fourteen numbers are a statement
# about the animal, and a live run has to be able to become one of them. This
# module keeps only what a *search* adds -- sampling, mutation, crossover and
# the identity of a genome in a report.
from stonkfly.genome import SPACE, WILD_TYPE  # noqa: F401


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
