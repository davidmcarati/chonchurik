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

import math

import numpy as np

PREFIX = "ORN_"
FEATURES = [
    "trend_fast",
    "trend_slow",
    "volatility",
    "range_position",
    "acceleration",
]
# An efference copy of the fly's own last executed trade, on its own band of
# glomeruli. It is fed at the NEXT observation, because nothing can be smelled
# before it happens. HOLD, a vetoed proposal and the very first observation
# all share the resting value, so "no trade" is one odour rather than three.
TRADE = "executed_trade"
TRADE_CODE = {None: 0.5, "HOLD": 0.5, "VETO": 0.5, "BUY": 1.0, "SELL": 0.0}
# Descriptors of a whole bar rather than of its close. Everything in FEATURES
# above is a function of one number, the closing price, which is why they
# share a fate: tools/features.py measured all five against the forward return
# on 10,446 hourly bars and found nothing at any horizon.
#
# These three are the readings a close cannot give. `flow` in particular is
# not a function of price at all -- it is the share of the bar's volume that
# was somebody lifting the offer rather than hitting the bid, and it answers a
# different question about the same minute.
#
# Measured before being added, on Binance BTCUSDT, both timeframes and both
# the training and the validation segment: `flow` and `bar_position` carry the
# same sign in all four panels, around -0.03 at one bar and gone by six. That
# is short-horizon reversion, it is small, and it is the only effect any
# descriptor in this file has ever shown. `flow_slow` is the same reading over
# the fast window, so a single bar cannot carry the channel alone.
#
# Three and not six. The glomeruli are a fixed budget -- 53 of them, split
# into equal bands -- so every channel added costs every other channel
# resolution. volume, trade size and bar range were measured alongside these
# and showed nothing, and a place code with four glomeruli to a feature can
# hardly express one.
BARS = ["flow", "flow_slow", "bar_position"]
CHANNELS = FEATURES + BARS + [TRADE]
# Observation counts, not minutes: the wall interval is a separate setting,
# and these are deliberately short. This fly is meant to scalp.
FAST, SLOW, RANGE = 5, 30, 60
# Channels are expressed in units of the market's own realised volatility, not
# in a fixed number of basis points. A trend is how many standard deviations
# of its own horizon the move covers; volatility is a ratio to the longer
# window; position in range is already a fraction.
#
# Fixed scales in log-return units were what this shipped with, and they tie
# the fly to one sampling interval: one-minute bars move 0.044% per bar and
# fifteen-minute bars 0.197%, so a scale calibrated for one reads flat on the
# other and full deflection on a third. Nothing below carries a unit that a
# change of interval could invalidate.
DEVIATIONS = 2.0
VOLATILITY_RATIO = 2.0
# Tuning width in glomeruli, and the floor below which a glomerulus is left
# unstimulated. Together they put about three glomeruli per feature above
# threshold, which is the sparseness a real antennal lobe delivers.
SIGMA = 0.9
FLOOR = 0.1
CURRENT = 30.0

PARAMETERS = {
    "olfactory_channels": CHANNELS,
    "olfactory_bar_channels": BARS,
    "olfactory_windows": {"fast": FAST, "slow": SLOW, "range": RANGE},
    "olfactory_normalisation": "Realised volatility of the same series. Trends are z-scores over their own horizon, volatility is a ratio of the short window to the long one, position in range is a fraction. No channel carries a unit tied to the sampling interval.",
    "olfactory_full_deflection_deviations": DEVIATIONS,
    "olfactory_tuning_sigma_glomeruli": SIGMA,
    "olfactory_threshold": FLOOR,
    "olfactory_peak_current": CURRENT,
    "olfactory_trade_code": {str(k): v for k, v in TRADE_CODE.items()},
    "interpretation": "Engineered assignment of price descriptors, whole-bar descriptors including order flow, and the fly's own last executed trade to glomerular channels. Fixed, alphabetical, content-independent; no odour identity, receptor affinity or concentration is modeled. The whole-bar channels rest unless the caller supplies klines.",
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
    if len(p) < 3:
        return {k: (0.0 if k == "volatility" else 0.5) for k in FEATURES}
    r = np.diff(np.log(p))
    fast, slow = r[-FAST:], r[-SLOW:]
    window = p[-RANGE:]
    span = float(window.max() - window.min())
    # How far this market usually travels over the window in question. A
    # random walk covers about sd * sqrt(n), so a trend is measured in those.
    reference = float(np.std(r[-RANGE:]))

    def trend(returns):
        scale = reference * math.sqrt(len(returns)) * DEVIATIONS
        return squash(float(returns.sum()) / scale) if scale > 0 else 0.5

    return {
        "trend_fast": trend(fast),
        "trend_slow": trend(slow),
        "volatility": (
            squash(float(np.std(fast)) / (reference * VOLATILITY_RATIO), signed=False)
            if reference > 0 else 0.0
        ),
        "range_position": (
            float((p[-1] - window.min()) / span) if span > 0 else 0.5
        ),
        "acceleration": (
            squash(
                (float(fast.mean()) - float(slow.mean()))
                / (reference * DEVIATIONS)
            )
            if reference > 0 else 0.5
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
            for b in np.array_split(np.arange(len(self.names)), len(CHANNELS))
        ]
        if min(len(b) for b in self.bands) < 2:
            raise RuntimeError("Too few glomeruli to give every channel a band")
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
                channel: [self.names[i] for i in band]
                for channel, band in zip(CHANNELS, self.bands)
            },
            "validated": False,
        }

    def activation(self, values):
        """Per-glomerulus amplitude in [0, 1]. A place code, one peak per channel."""
        out = np.zeros(len(self.names), dtype=np.float32)
        for channel, band in zip(CHANNELS, self.bands):
            x = float(np.clip(values.get(channel, 0.5), 0.0, 1.0))
            centre = x * (len(band) - 1)
            offset = np.arange(len(band), dtype=np.float32) - centre
            out[band] = np.exp(-0.5 * (offset / self.sigma) ** 2)
        return np.where(out < self.floor, 0.0, out)

    def stimulation(self, history, executed=None, bars=None):
        """One pulse for the whole observation, or None when there is no odour.

        Indices stay unique across glomeruli: `drive[ix] += amplitude` is plain
        fancy indexing, so a repeated index would silently overwrite instead of
        summing.

        `bars` are whole klines for the same history, newest last. Without
        them the three whole-bar channels sit at their resting value, which is
        the same thing `activation` does for any channel it is not given -- so
        a caller written before those channels existed, or a data source that
        carries closes only, measures what it always measured.
        """
        if executed not in TRADE_CODE:
            raise ValueError(f"Unknown executed trade: {executed!r}")
        values = {**features(history), **bar_features(bars),
                  TRADE: TRADE_CODE[executed]}
        amplitude = self.activation(values)
        live = np.flatnonzero(amplitude > 0)
        if not len(live):
            return None, values
        indices = np.concatenate([self.cells[g] for g in live])
        current = np.concatenate(
            [np.full(len(self.cells[g]), amplitude[g]) for g in live]
        )
        return (indices, (self.current * current).astype(np.float32)), values


def bar_features(bars):
    """The whole-bar descriptors. Empty when there are no bars to read.

    `bars` are dicts with at least high, low, close, volume and
    taker_buy_base, oldest first; `tools/fetch_binance.py` writes them and
    `tools/evolve/series.klines` reads them back. At most FAST of them are
    consulted for the averaged channel and one for the rest, so a caller can
    hand over a short window.

    Returning nothing rather than a neutral dict is deliberate: `activation`
    already rests a channel it is not given, and an explicit 0.5 here would be
    indistinguishable from a bar that genuinely read 0.5.
    """
    if not bars:
        return {}
    last = bars[-1]
    fast = bars[-FAST:]

    def share(bar):
        volume = bar.get("volume", 0.0)
        if volume <= 0:
            return 0.5
        return float(np.clip(bar.get("taker_buy_base", 0.0) / volume, 0.0, 1.0))

    span = last["high"] - last["low"]
    return {
        # Centred on a half in absolute terms, not against the window's own
        # mean: which side was the aggressor is meaningful by itself, and an
        # hour of steady buying must not read neutral because of its own
        # persistence.
        "flow": share(last),
        "flow_slow": float(sum(share(b) for b in fast) / len(fast)),
        "bar_position": (float((last["close"] - last["low"]) / span)
                         if span > 0 else 0.5),
    }
