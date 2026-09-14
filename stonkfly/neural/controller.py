"""Only sensory input and engineered reinforcement enter the network.

The chart reaches the retina and the price history reaches the olfactory
receptors. No market policy, no profit term and no price rule touches the
proposal the network produces.
"""

import hashlib

import numpy as np

from .common import annotations
from .gustation import Gustation
from .olfaction import Olfaction
from .visual import VisualMemoryBrain


class Decoder:
    def __init__(self, ids, annotation, threshold):
        types = annotation.type.fillna("")
        sides = annotation.somaSide.fillna("")
        self.left = np.flatnonzero(types.eq("DNp20") & sides.eq("L"))
        self.right = np.flatnonzero(types.eq("DNp20") & sides.eq("R"))
        self.gate = np.flatnonzero(types.eq("DNpe017"))
        if not len(self.left) or not len(self.right) or not len(self.gate):
            raise RuntimeError("Missing annotated BCI outputs")
        self.threshold = threshold
        self.identities = {
            k: [str(ids[i]) for i in getattr(self, k)]
            for k in ["left", "right", "gate"]
        }

    def decode(self, counts, seconds):
        # Mean rates prevent side population size from creating a built-in bias.
        left = float(np.mean(counts[self.left]) / seconds)
        right = float(np.mean(counts[self.right]) / seconds)
        difference = right - left
        gate = int(counts[self.gate].sum())
        side = (
            "HOLD"
            if not gate or abs(difference) < self.threshold
            else "BUY"
            if difference > 0
            else "SELL"
        )
        return {
            "side": side,
            "left_hz": left,
            "right_hz": right,
            "difference_hz": difference,
            "gate_spikes": gate,
            "cell_ids": self.identities,
        }


class FlyController:
    def __init__(self, settings, odor=True, taste=True):
        self.s = settings
        self.brain = VisualMemoryBrain()
        self.brain.weights_frozen = not settings.learning
        a = annotations(self.brain.ids)
        self.decoder = Decoder(self.brain.ids, a, settings.decoder_threshold_hz)
        # Off only for diagnostics that need the pre-olfactory control arm.
        self.olfaction = Olfaction(a) if odor else None
        self.gustation = Gustation(self.brain.sugar) if taste else None

    def observe(self, rgb, reinforcement, history=None, executed=None, account=None):
        """One market observation.

        `history` opens the olfactory channel, `account` the gustatory one, and
        `executed` names the fly's own last filled trade so it can smell it.
        Each stays shut when its argument is absent, so a caller written before
        a channel existed still measures what it measured then.
        """
        if reinforcement not in ("none", "reward", "aversive"):
            raise ValueError("Unknown reinforcement")
        b = self.brain
        # Odour and taste are present for the whole observation; the
        # reinforcement pulse is not. They are separate entries in the list.
        odor, smelled = (None, None)
        if self.olfaction is not None and history is not None:
            odor, smelled = self.olfaction.stimulation(history, executed)
        taste, tasted = (None, None)
        if self.gustation is not None and account is not None:
            taste, tasted = self.gustation.stimulation(*account)
        counts = np.zeros(b.n, dtype=np.int32)
        wall = 0.0
        remaining = round(self.s.neural_ms / b.dt)
        pulse = round(self.s.pulse_ms / b.dt) if reinforcement != "none" else 0
        delivered = 0
        while remaining:
            n = min(remaining, round(self.s.neural_bin_ms / b.dt))
            if pulse:
                n = min(n, pulse)
            stimulus = [x for x in (odor, taste) if x is not None]
            if pulse:
                stimulus.append((b.circuit[reinforcement], self.s.pulse_current))
            c, elapsed = b.rgb_step(
                rgb, n * b.dt, learning=self.s.learning, stimulation=stimulus or None
            )
            counts += c
            wall += elapsed
            remaining -= n
            if pulse:
                delivered += n
                pulse -= n
        b.counts[:] = counts
        return {
            **self.decoder.decode(counts, self.s.neural_ms / 1000),
            "brain_ms": b.sim_ms,
            "compute_seconds": wall,
            "stimulus": reinforcement,
            "stimulus_ms": delivered * b.dt,
            "odor": smelled,
            "satiety": tasted,
            "sugar_spikes": (
                0 if self.gustation is None
                else int(counts[self.gustation.indices].sum())
            ),
            "ORN_spikes": (
                0 if self.olfaction is None
                else int(counts[self.olfaction.indices].sum())
            ),
            "reward_spikes": int(counts[b.circuit["reward"]].sum()),
            "aversive_spikes": int(counts[b.circuit["aversive"]].sum()),
            "KC_spikes": int(counts[b.circuit["kc"]].sum()),
            "total_spikes": int(counts.sum()),
            "spike_sha256": hashlib.sha256(counts.tobytes()).hexdigest(),
            "input_sha256": hashlib.sha256(np.asarray(rgb).tobytes()).hexdigest(),
            "memory": b.memory(),
        }

    def save(self, path):
        self.brain.checkpoint(path)

    def restore(self, path):
        self.brain.restore(path)
