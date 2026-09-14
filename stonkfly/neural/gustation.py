"""Engineered account-state-to-taste adapter. This is not fly gustation.

LB3c are sugar-sensing gustatory receptor neurons. MaleCNS v1.0 retains 23 of
them; `prepare.py` has always compiled them into graph.npz and nothing has
ever driven them. Feeding account equity to them is an arbitrary engineered
assignment with no biological content: a fly tastes sugar, not a balance.

DISCLOSURE. Until this module the network never received portfolio state. The
chart excluded balances and P&L by construction, and the only profit-derived
input was the reward/aversive dopamine pulse, which is momentary and binary.
With this channel the network receives account equity continuously, as a
graded input, every observation. That is a real widening of what the fly is
told, and every claim about what it learned has to be read against it.

It remains an input. Nothing here selects an action, replaces a proposal or
gives the decoder a profit term; see AGENTS.md on hidden profit-based action
selection. It is disclosed in provenance.json, docs/model.md and
docs/validation.md.
"""

import math

import numpy as np

# A 2% move away from the reference equity spans most of the range. Declared
# model choice, not fitted to returns.
SCALE = 0.02
# Satiety is delivered as a current between these two, not between zero and a
# peak. Measured reason: a cell needs roughly 7 units of drive to reach
# threshold at all, so a channel that rests below it reports gains and nothing
# else -- a 2% loss and a 5% loss both arrive as silence, and so does an
# untouched account. Resting well above threshold makes the channel graded in
# both directions, which is the whole point of a satiety signal.
FLOOR_CURRENT = 8.0
SPAN_CURRENT = 24.0
PARAMETERS = {
    "satiety_scale": SCALE,
    "satiety_floor_current": FLOOR_CURRENT,
    "satiety_span_current": SPAN_CURRENT,
    "reference": "The ledger's accounting anchor, i.e. equity at the previous observation boundary.",
    "interpretation": "Account equity relative to its reference, delivered to LB3c sugar gustatory neurons as a current graded in both directions around a resting level. Engineered assignment; no feeding state, metabolic drive or taste quality is modeled.",
}


def satiety(equity, reference):
    """Equity relative to its reference, squashed into [0, 1].

    Exactly 0.5 when equity matches the reference, so an untouched account is
    a definite resting taste rather than an absence of input. Returns the
    resting value for any reference that cannot be divided by, which keeps a
    zeroed or missing balance from reading as maximal loss.
    """
    try:
        value, base = float(equity), float(reference)
    except (TypeError, ValueError):
        return 0.5
    if not math.isfinite(value) or not math.isfinite(base) or base <= 0:
        return 0.5
    return 0.5 * (1 + math.tanh((value / base - 1) / SCALE))


class Gustation:
    """Drives the retained LB3c population with one graded current."""

    def __init__(self, sugar, floor=FLOOR_CURRENT, span=SPAN_CURRENT):
        self.indices = np.asarray(sugar, dtype=np.int32)
        if not len(self.indices) or len(np.unique(self.indices)) != len(self.indices):
            raise RuntimeError("Invalid sugar gustatory population")
        self.floor = float(floor)
        self.span = float(span)
        self.report = {
            **PARAMETERS,
            "sugar_neurons": int(len(self.indices)),
            "validated": False,
        }

    def stimulation(self, equity, reference):
        """One pulse for the whole observation, plus the value delivered."""
        value = satiety(equity, reference)
        return (self.indices, np.float32(self.floor + self.span * value)), value
