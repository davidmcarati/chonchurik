"""Engineered market-to-odour adapter. This is not fly olfaction.

A real ORN_DA1 answers cVA. Assigning it a price feature has no biological
content at all. What is borrowed here is the architecture, not the chemistry:
MaleCNS v1.0 retains 2,635 olfactory receptor neurons in 53 glomerular
channels, and they reach 3,829 of the 4,064 Kenyon cells in two synapses
through the antennal lobe -- the canonical sparse-coding input the mushroom
body is built around. The visual pathway the experiment used until now has no
such direct route, and measurement showed it delivering whole-field luminance
rather than a combinatorial code (see docs/validation.md).

Feature definitions, window lengths, scales, tuning width and drive current
below are declared model choices. None of them is fitted to returns. Only the
price history enters here; account balance and executed trades are separate
channels with their own disclosure.
"""

import numpy as np

PREFIX = "ORN_"
FEATURES = [
    "trend_fast",
    "trend_slow",
    "volatility",
    "range_position",
    "acceleration",
]
# Observation counts, not minutes: the wall interval is a separate setting.
FAST, SLOW, RANGE = 5, 30, 60
# Deflection scales in log-return units. Declared, not fitted: roughly the
# size of a move this experiment should treat as large.
SCALES = {
    "trend_fast": 0.003,
    "trend_slow": 0.008,
    "volatility": 0.002,
    "acceleration": 0.002,
}
# Tuning width in glomeruli, and the floor below which a glomerulus is left
# unstimulated. Together they put about three glomeruli per feature above
# threshold, which is the sparseness a real antennal lobe delivers.
SIGMA = 0.9
FLOOR = 0.1
CURRENT = 30.0

PARAMETERS = {
    "olfactory_features": FEATURES,
    "olfactory_windows": {"fast": FAST, "slow": SLOW, "range": RANGE},
    "olfactory_scales": SCALES,
    "olfactory_tuning_sigma_glomeruli": SIGMA,
    "olfactory_threshold": FLOOR,
    "olfactory_peak_current": CURRENT,
    "interpretation": "Engineered assignment of price descriptors to glomerular channels. Fixed, alphabetical, content-independent; no odour identity, receptor affinity or concentration is modeled.",
}


def squash(x, signed=True):
    """Bounded, monotone, and centred at 0.5 for a market that is going nowhere."""
    t = float(np.tanh(x))
    return 0.5 * (1 + t) if signed else abs(t)


def features(history):
    """Scale-free descriptors of the price history alone.

    Returns a value in [0, 1] per feature. A flat market sits at 0.5 for the
    signed features and 0 for volatility, so "nothing is happening" is itself
    a distinct odour rather than an absence of input.
    """
    p = np.asarray(history, dtype=float)
    p = p[np.isfinite(p) & (p > 0)]
    if len(p) < 2:
        return {k: (0.0 if k == "volatility" else 0.5) for k in FEATURES}
    r = np.diff(np.log(p))
    fast, slow = r[-FAST:], r[-SLOW:]
    window = p[-RANGE:]
    span = float(window.max() - window.min())
    return {
        "trend_fast": squash(fast.sum() / SCALES["trend_fast"]),
        "trend_slow": squash(slow.sum() / SCALES["trend_slow"]),
        "volatility": squash(float(np.std(slow)) / SCALES["volatility"], signed=False),
        "range_position": (
            float((p[-1] - window.min()) / span) if span > 0 else 0.5
        ),
        "acceleration": squash(
            (fast.mean() - slow.mean()) / SCALES["acceleration"]
        ),
    }


class Olfaction:
    """Fixed assignment of features to glomerular channels.

    Glomeruli are taken in alphabetical order and split into equal contiguous
    bands, one per feature. The assignment cannot depend on what the market
    did, because it is computed before any price is read.
    """

    def __init__(self, annotation, current=CURRENT, sigma=SIGMA, floor=FLOOR):
        kinds = annotation.type.fillna("").astype(str)
        mask = kinds.str.startswith(PREFIX).to_numpy()
        self.names = sorted(set(kinds[mask]))
        if not self.names:
            raise RuntimeError("No olfactory receptor neurons in this graph")
        self.cells = [
            np.flatnonzero((kinds == name).to_numpy()).astype(np.int32)
            for name in self.names
        ]
        self.bands = [
            np.asarray(b, dtype=np.int32)
            for b in np.array_split(np.arange(len(self.names)), len(FEATURES))
        ]
        if min(len(b) for b in self.bands) < 2:
            raise RuntimeError("Too few glomeruli to give every feature a band")
        self.indices = np.concatenate(self.cells).astype(np.int32)
        if len(np.unique(self.indices)) != len(self.indices):
            raise RuntimeError("A receptor neuron belongs to two glomeruli")
        self.current = float(current)
        self.sigma = float(sigma)
        self.floor = float(floor)
        self.report = {
            **PARAMETERS,
            "receptor_neurons": int(len(self.indices)),
            "glomeruli": len(self.names),
            "band_sizes": [int(len(b)) for b in self.bands],
            "assignment": {
                feature: [self.names[i] for i in band]
                for feature, band in zip(FEATURES, self.bands)
            },
            "validated": False,
        }

    def activation(self, values):
        """Per-glomerulus amplitude in [0, 1]. A place code, one peak per feature."""
        out = np.zeros(len(self.names), dtype=np.float32)
        for feature, band in zip(FEATURES, self.bands):
            x = float(np.clip(values[feature], 0.0, 1.0))
            centre = x * (len(band) - 1)
            offset = np.arange(len(band), dtype=np.float32) - centre
            out[band] = np.exp(-0.5 * (offset / self.sigma) ** 2)
        return np.where(out < self.floor, 0.0, out)

    def stimulation(self, history):
        """One pulse for the whole observation, or None when there is no odour.

        Indices stay unique across glomeruli: `drive[ix] += amplitude` is plain
        fancy indexing, so a repeated index would silently overwrite instead of
        summing.
        """
        values = features(history)
        amplitude = self.activation(values)
        live = np.flatnonzero(amplitude > 0)
        if not len(live):
            return None, values
        indices = np.concatenate([self.cells[g] for g in live])
        current = np.concatenate(
            [np.full(len(self.cells[g]), amplitude[g]) for g in live]
        )
        return (indices, (self.current * current).astype(np.float32)), values
