"""Only sensory input and engineered reinforcement enter the network.

The chart reaches the retina and the price history reaches the olfactory
receptors. No market policy, no profit term and no price rule touches the
proposal the network produces.
"""

import collections
import hashlib

import numpy as np

from .common import annotations
from .gustation import Gustation
from .olfaction import Olfaction
from .visual import VisualMemoryBrain


# How many past observations the resting difference is taken over. Long
# enough that one loud observation cannot move it, short enough to follow a
# network whose adaptation and plastic weights are still settling. The median
# rather than the mean for the same reason.
BASELINE = 60


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

    def decode(self, counts, seconds, threshold=None, baseline=0.0):
        """`threshold` overrides the one this was built with.

        It has to be passed, not remembered, because the threshold is a
        declared free parameter and an evolution changes it per genome by
        replacing the controller's Settings. Reading it once at construction
        meant every search so far optimised a number the fly could not feel;
        `FlyController.observe` now hands over the live one on every call.

        `baseline` is the fly's own resting difference, subtracted before the
        comparison. Measured: the wild type's difference sits at -17.15 Hz
        with a spread of 5.57, and three evolved genomes sit at +9.05, +8.60
        and +4.42 with spreads of 5.77, 4.76 and 6.34. The offset differs by
        26 Hz between genomes; the variation does not differ at all. Compared
        against an absolute threshold declared over 0.5 to 12 Hz, that offset
        decides everything and the market decides nothing -- the wild type
        proposed SELL at 80 observations of 80, and the fittest evolved genome
        BUY at 64 of 80, because their constants had opposite signs.

        Subtracting it compares the fly against itself. It is a normalisation
        and not a policy: it never sees a return, a profit or a price, only
        this readout's own recent level, and only from observations already
        past. Passing zero reproduces every result recorded before it existed.
        """
        # Mean rates prevent side population size from creating a built-in bias.
        left = float(np.mean(counts[self.left]) / seconds)
        right = float(np.mean(counts[self.right]) / seconds)
        difference = right - left
        relative = difference - float(baseline)
        gate = int(counts[self.gate].sum())
        limit = self.threshold if threshold is None else float(threshold)
        side = (
            "HOLD"
            if not gate or abs(relative) < limit
            else "BUY"
            if relative > 0
            else "SELL"
        )
        return {
            "side": side,
            "left_hz": left,
            "right_hz": right,
            "difference_hz": difference,
            "relative_hz": relative,
            "baseline_hz": float(baseline),
            "gate_spikes": gate,
            "threshold_hz": limit,
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
        # The fly's own recent readout, for the resting difference the decoder
        # subtracts. Strictly past observations, and cleared by `reset`.
        self.readouts = collections.deque(maxlen=BASELINE)

    def reset(self):
        """Back to a fly that has seen nothing. Use this, not `brain.reset()`.

        The readout history is part of what a fly carries between
        observations, so a caller that reset only the brain would start the
        next run, or the next genome, with the previous one's resting
        difference and would not be able to tell.
        """
        self.readouts.clear()
        self.brain.reset()

    def baseline(self):
        """The resting difference: the median of what this fly has seen.

        Zero until it has seen anything, which is what the decoder did before
        this existed.
        """
        return float(np.median(self.readouts)) if self.readouts else 0.0

    def observe(self, rgb, reinforcement, history=None, executed=None,
                account=None, bars=None):
        """One market observation.

        `history` opens the olfactory channel, `account` the gustatory one, and
        `executed` names the fly's own last filled trade so it can smell it.
        Each stays shut when its argument is absent, so a caller written before
        a channel existed still measures what it measured then.

        `bars` are whole klines for the same history. They add the three
        whole-bar odour channels; without them those rest, which is what every
        caller and every recorded diagnostic did before they existed.
        """
        if reinforcement not in ("none", "reward", "aversive"):
            raise ValueError("Unknown reinforcement")
        b = self.brain
        # Odour and taste are present for the whole observation; the
        # reinforcement pulse is not. They are separate entries in the list.
        odor, smelled = (None, None)
        if self.olfaction is not None and history is not None:
            odor, smelled = self.olfaction.stimulation(history, executed, bars)
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
        decoded = self.decoder.decode(counts, self.s.neural_ms / 1000,
                                      self.s.decoder_threshold_hz,
                                      self.baseline())
        # After deciding, never before: the resting difference this
        # observation was judged against is made only of earlier ones.
        self.readouts.append(decoded["difference_hz"])
        return {
            **decoded,
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
