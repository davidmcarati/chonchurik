"""A whole generation of flies on one GPU, advanced in lockstep.

One CUDA block runs one fly's whole observation, and the batch fills the card.
The kernel is `stonkfly/neural/kernel.cu`, which `tools/gpu_port_check.py`
proves identical to `kernel.cpp` element for element on all eighteen mutable
arrays, so a fly evaluated here is the same animal as a fly evaluated by a
worker process -- not an approximation of one. `tools/herd_check.py` is the
end-to-end version of that claim: same genome, same prices, same proposals.

Only the integration moved. It was 95.2% of an observation, measured, and the
other 4.8% still runs where it ran before, through the same code:

  * the sensory front end is `VisualMemoryBrain.rgb_bin` and
    `MemoryBrain.prepare_drive`, called once per fly per 10 ms bin on the one
    CPU brain the herd borrows. Those two were split out of `rgb_step` and
    `_neural_step` for exactly this, with the bodies unchanged;
  * the memory rule is `stonkfly.neural.rule.advance`, called once per fly per
    bin on that fly's own traces. Rewriting it batched would be faster and
    would be a second implementation of the thing the results depend on.

What the herd owns per fly is the state the kernel touches -- membrane, memory
traces, spike queue, active list -- and a private copy of the weights, because
the plasticity rule writes to them and `inhibitory_gain` is evolved. That copy
is 102 MB, and it is what caps the batch rather than anything about speed:
throughput is already flat from 84 blocks upward.

Nothing here decides a trade. The herd reports what the decoder says and the
accounts do the rest, exactly as `evaluate.replay` does for one fly.
"""

import json
import time
from pathlib import Path

import numpy as np

from stonkfly.display import market_frame
from stonkfly.neural.brain import PARAMETERS
from stonkfly.neural.olfaction import FAST
from stonkfly.reinforcement import reinforcement

from .evaluate import CHART_WINDOW, PRODUCT, WARMUP, Account, buy_and_hold
from .genome import WILD_TYPE
from .series import quotes

KERNEL = (Path(__file__).resolve().parents[2] / "stonkfly" / "neural"
          / "kernel.cu")
BLOCK = 256
# Double-dashed: NVRTC silently ignores an option it does not recognise, so a
# single dash would leave contraction on while looking as though it were off,
# and the port's bit-exactness rests on it being off.
OPTIONS = ("--fmad=false", "--prec-div=true", "--prec-sqrt=true")
TABLE = 1 << 20

# Appended to the kernel source rather than compiled on its own, so `Cell` here
# is the struct the kernel declares and cannot drift from it.
SUPPORT = r"""
extern "C" __global__ void set_drive(Cell* cell, const float* drive,
                                     const long long total) {
  const long long k = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  if (k < total) cell[k].drive = drive[k];
}

extern "C" __global__ void clear_counts(Cell* cell, const long long total) {
  const long long k = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  if (k < total) cell[k].counts = 0;
}

// One row per fly of the spike counts the host actually reads: the decoder's
// cells, the Kenyon cells, and the two populations the memory rule needs.
// Downloading all 166,700 every bin would be 56 MB for 12,000 useful ones.
extern "C" __global__ void gather_counts(const Cell* cell, const int* index,
                                         const int k, const int n,
                                         const int batch, int* out) {
  const long long t = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  if (t >= (long long)k * batch) return;
  const int b = (int)(t / k), j = (int)(t % k);
  out[t] = cell[(long long)b * n + index[j]].counts;
}

extern "C" __global__ void spread_weights(float* weight,
                                          const long long stride,
                                          const float* __restrict__ pristine,
                                          const long long nedges,
                                          const int batch) {
  const long long t = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  if (t >= nedges * batch) return;
  weight[(t / nedges) * stride + (t % nedges)] = pristine[t % nedges];
}

// The genome's inhibitory gain, applied the way evaluate.py applies it: the
// pristine magnitude times the gain, never the current value divided and
// multiplied back, because float32 does not survive that round trip.
extern "C" __global__ void scale_inhibitory(
    float* weight, const long long stride,
    const long long* __restrict__ edge, const float* __restrict__ pristine,
    const float* __restrict__ gain, const long long m, const int batch) {
  const long long t = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  if (t >= m * batch) return;
  const int b = (int)(t / m);
  const long long p = t % m;
  weight[(long long)b * stride + edge[p]] = pristine[p] * gain[b];
}

extern "C" __global__ void set_plastic(
    float* weight, const long long stride,
    const long long* __restrict__ edge, const float* __restrict__ baseline,
    const int nplastic, const int batch) {
  const long long t = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  if (t >= (long long)nplastic * batch) return;
  const int b = (int)(t / nplastic), p = (int)(t % nplastic);
  weight[(long long)b * stride + edge[p]] = baseline[p];
}

// `weight[edges] = baseline_plastic * (1 + memory_w)` as numpy computes it: a
// float32 baseline times a float64 sum, in double, rounded once on the way
// back into a float32 array.
extern "C" __global__ void apply_memory(
    float* weight, const long long stride,
    const long long* __restrict__ edge, const float* __restrict__ baseline,
    const double* __restrict__ w, const int nplastic, const int batch) {
  const long long t = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  if (t >= (long long)nplastic * batch) return;
  const int b = (int)(t / nplastic), p = (int)(t % nplastic);
  weight[(long long)b * stride + edge[p]] =
      (float)((double)baseline[p] * (1.0 + w[t]));
}
"""

CELL = np.dtype([("last", "<i8"), ("v", "<f4"), ("g", "<f4"),
                 ("adaptation", "<f4"), ("drive", "<f4"), ("counts", "<i4"),
                 ("refractory", "<i2"), ("flags", "u1"), ("pad", "u1")])
SLOW = np.dtype([("eligibility", "<f8"), ("eligibility_last", "<i8"),
                 ("modulation_last", "<i8"), ("modulation", "<f4"),
                 ("pad", "<f4")])
FIXED = np.dtype([("rest", "<f4"), ("kc", "u1"), ("modulatory", "u1"),
                  ("pad", "u1"), ("dan", "i1")])
assert (CELL.itemsize, SLOW.itemsize, FIXED.itemsize) == (32, 32, 8)


def capacity(free_bytes, n, nedges, slots, reserve=0.15):
    """How many flies fit, leaving a margin for the allocator to work in."""
    per = ((CELL.itemsize + SLOW.itemsize + 8 + 4 * slots) * n
           + 4 * nedges + 4 * TABLE)
    return max(1, int((free_bytes * (1 - reserve)) // per))


_MODULE = {}


def module(block=BLOCK):
    """Compiled once per process. NVRTC takes about three seconds, and a
    generation builds a new herd for every wave."""
    import cupy as cp

    if block not in _MODULE:
        _MODULE[block] = cp.RawModule(
            code=KERNEL.read_text(encoding="utf-8") + SUPPORT,
            options=OPTIONS + (f"-D BLOCK={block}",))
    return _MODULE[block]


def cuda_headers():
    """NVRTC has to find the CUDA headers; the pip wheels hide them."""
    import os
    import site

    if os.environ.get("CUDA_PATH"):
        return
    for root in site.getsitepackages():
        candidate = Path(root) / "nvidia" / "cuda_runtime"
        if (candidate / "include" / "cuda_runtime.h").exists():
            os.environ["CUDA_PATH"] = str(candidate)
            return


def standing_pulses(controller, history, executed, account, bars=None):
    """The odour and taste currents for one observation, as the controller
    assembles them.

    The filter is the point. `Olfaction.stimulation` returns no pulse at all
    when the genome's floor and width leave every glomerulus at zero, and
    `FlyController.observe` drops it with `[x for x in (odor, taste) if x is
    not None]`. The herd did not, and an evolution died on the first random
    genome that produced no odour -- `cannot unpack non-iterable NoneType`
    inside `prepare_drive`, four minutes into generation 0.
    """
    out = []
    if controller.olfaction is not None and history is not None:
        odor = controller.olfaction.stimulation(history, executed, bars)[0]
        if odor is not None:
            out.append(odor)
    if controller.gustation is not None and account is not None:
        taste = controller.gustation.stimulation(*account)[0]
        if taste is not None:
            out.append(taste)
    return out


class Herd:
    """The flies of one wave, all replaying the same chronological start."""

    def __init__(self, controller, pristine_inhibitory, genomes, settings,
                 starts=None, measure=False):
        cuda_headers()
        import cupy as cp

        from tools.gpu_port_check import tables

        self.cp = cp
        self.controller = controller
        self.settings = settings
        self.genomes = [dict(g) for g in genomes]
        self.batch = len(self.genomes)
        # A fly is a genome at a chronological start, and a wave of them can
        # hold several starts at once -- the full tier evaluates every survivor
        # at five. Flies sharing a start share the chart, so they share the
        # rendering and the photoreceptor settling; what they never share is
        # the account, and with it the reinforcement sign, the satiety current
        # and the odour of the last fill.
        self.starts = list(starts) if starts is not None else [0] * self.batch
        if len(self.starts) != self.batch:
            raise ValueError("one start per genome")
        brain = self.brain = controller.brain
        n = self.n = brain.n
        c = self.circuit = brain.circuit

        dt = self.dt = brain.dt
        self.ticks = round(settings.neural_ms / dt)
        self.bin = round(settings.neural_bin_ms / dt)
        self.pulse = round(settings.pulse_ms / dt)
        if self.ticks % self.bin or self.pulse % self.bin:
            raise ValueError(
                "The herd advances every fly over the same bins, which needs "
                "neural_ms and pulse_ms to be whole numbers of neural_bin_ms; "
                f"these are {settings.neural_ms}, {settings.pulse_ms} and "
                f"{settings.neural_bin_ms}."
            )
        self.delay = int(round(1.8 / dt))
        self.rfc = int(round(2.2 / dt))
        self.slots = self.delay + 1

        built = module()
        self.kernel = built.get_function("memory_advance_batch")
        self.helper = {name: built.get_function(name) for name in
                       ["set_drive", "clear_counts", "gather_counts",
                        "spread_weights", "scale_inhibitory", "set_plastic",
                        "apply_memory"]}

        self.tau_elig = np.float32(PARAMETERS["trace_kc_seconds"] * 1000)
        self._tables = tables
        av, ag, _, am, ae = tables(dt, brain.adaptation_tau,
                                   float(self.tau_elig))
        self.av, self.ag, self.am, self.ae = (cp.asarray(x)
                                              for x in (av, ag, am, ae))

        fixed = np.zeros(n, FIXED)
        fixed["rest"] = brain.rest
        fixed["kc"] = c["kc_mask"]
        fixed["modulatory"] = brain.modulation_mask
        fixed["dan"] = c["dan_index"]
        self.fixed = cp.asarray(fixed.view(np.uint8))

        self.ptr = cp.asarray(brain.ptr)
        self.post = cp.asarray(brain.post)
        self.nedges = len(brain.post)
        self.nplastic = len(c["edges"])
        self.edges = cp.asarray(c["edges"])
        self.pre = cp.asarray(c["pre"])
        self.baseline = cp.asarray(brain.baseline_plastic)
        self.gain = cp.asarray(c["gain"])
        self.inhibitory_edges = cp.asarray(brain.inhibitory_edges)
        self.pristine_inhibitory = cp.asarray(pristine_inhibitory)
        self.clean_weight = cp.asarray(np.asarray(brain.weight))
        self.scratch = cp.zeros(1, cp.int32)

        # Everything the host reads back per bin, in one gather.
        d = controller.decoder
        parts = [np.asarray(c["pre"], np.int32), np.asarray(c["dan"], np.int32),
                 d.left.astype(np.int32), d.right.astype(np.int32),
                 d.gate.astype(np.int32), np.asarray(c["kc"], np.int32)]
        self.cuts = np.cumsum([0] + [len(p) for p in parts])
        self.index = cp.asarray(np.concatenate(parts))
        self.gathered = cp.zeros(self.batch * int(self.cuts[-1]), cp.int32)

        # Off by default: timing the kernel means waiting for it, which is
        # exactly what a run should not do.
        self.measure = measure
        self.cost = dict.fromkeys(["sensory", "upload", "kernel", "rule"], 0.0)

        self._allocate()
        self._physiology()
        self.reset()

    def _mark(self, bucket, since):
        if not self.measure:
            return None
        self.cp.cuda.Stream.null.synchronize()
        now = time.perf_counter()
        self.cost[bucket] += now - since
        return now

    # -- device memory ------------------------------------------------------

    def _allocate(self):
        cp, B, n = self.cp, self.batch, self.n
        self.cell = cp.zeros(B * n * CELL.itemsize, cp.uint8)
        self.slow = cp.zeros(B * n * SLOW.itemsize, cp.uint8)
        self.previous_drive = cp.zeros(B * n, cp.float32)
        self.queue = cp.zeros(B * self.slots * n, cp.int32)
        self.queue_count = cp.zeros(B * self.slots, cp.int32)
        self.active = cp.zeros(B * n, cp.int32)
        self.nactive = cp.zeros(B, cp.int32)
        self.clock = cp.zeros(B, cp.int64)
        self.weight = cp.zeros(B * self.nedges, cp.float32)
        self.drive = cp.zeros(B * n, cp.float32)
        self.drive_host = np.zeros((B, n), np.float32)
        self.memory_w_device = cp.zeros(B * self.nplastic, cp.float64)

    def _physiology(self):
        """The four evolved parameters the kernel itself has to know."""
        cp = self.cp

        def column(name):
            return cp.asarray(
                np.array([g[name] for g in self.genomes], np.float32))

        self.eta = column("eta")
        self.adaptation_jump = column("adaptation_jump")
        self.adaptation_tau = column("adaptation_tau")
        self.kc_rest = column("kc_rest")
        self.inhibitory_gain = column("inhibitory_gain")
        # aa tabulates exp(-dt*i/adaptation_tau), so it is one table per fly.
        # Built by the host's own compiler: this device's exp disagrees with it
        # on 3% of the entries, which is enough to move a spike.
        self.aa = cp.asarray(np.concatenate([
            self._tables(self.dt, g["adaptation_tau"], float(self.tau_elig))[2]
            for g in self.genomes]))

    # -- state --------------------------------------------------------------

    def reset(self):
        """Every fly back to its own starting brain. `MemoryBrain.reset`."""
        cp, B, n = self.cp, self.batch, self.n
        brain = self.brain
        brain.reset()
        initial = brain.initial

        cells = np.zeros((B, n), CELL)
        rest = np.tile(np.asarray(initial["v"]), (B, 1))
        for b, genome in enumerate(self.genomes):
            rest[b][self.circuit["kc"]] = np.float32(genome["kc_rest"])
        cells["v"] = rest
        cells["g"] = np.asarray(initial["g"])
        cells["adaptation"] = np.asarray(initial["adaptation"])
        cells["drive"] = np.asarray(initial["drive"])
        cells["counts"] = np.asarray(initial["counts"])
        cells["refractory"] = np.asarray(initial["refractory"])
        cells["flags"] = np.asarray(initial["active_flag"])
        cells["last"] = np.asarray(initial["last"])
        self.cell[...] = cp.asarray(cells.reshape(-1).view(np.uint8))

        slow = np.zeros((B, n), SLOW)
        slow["eligibility"] = np.asarray(initial["eligibility"])
        slow["eligibility_last"] = np.asarray(initial["eligibility_last"])
        slow["modulation"] = np.asarray(initial["modulation"])
        slow["modulation_last"] = np.asarray(initial["modulation_last"])
        self.slow[...] = cp.asarray(slow.reshape(-1).view(np.uint8))

        self.previous_drive[...] = cp.asarray(
            np.tile(np.asarray(initial["previous_drive"]), B))
        self.queue[...] = cp.asarray(
            np.tile(np.asarray(initial["queue"]).ravel(), B))
        self.queue_count[...] = cp.asarray(
            np.tile(np.asarray(initial["queue_count"]).ravel(), B))
        self.active[...] = cp.asarray(np.tile(np.asarray(initial["active"]), B))
        self.nactive[...] = cp.asarray(
            np.tile(np.asarray(initial["nactive"]).astype(np.int32), B))
        self.clock[...] = 0
        self.drive[...] = 0

        # Weights: the pristine graph, this genome's inhibitory gain, and the
        # plastic edges back at baseline. Built on the device, because B copies
        # of 102 MB is not something to push across the bus.
        self._launch("spread_weights", self.nedges * B,
                     (self.weight, np.int64(self.nedges), self.clean_weight,
                      np.int64(self.nedges), np.int32(B)))
        m = len(self.inhibitory_edges)
        self._launch("scale_inhibitory", m * B,
                     (self.weight, np.int64(self.nedges),
                      self.inhibitory_edges, self.pristine_inhibitory,
                      self.inhibitory_gain, np.int64(m), np.int32(B)))
        self._launch("set_plastic", self.nplastic * B,
                     (self.weight, np.int64(self.nedges), self.edges,
                      self.baseline, np.int32(self.nplastic), np.int32(B)))

        # Retinal and photoreceptor settling belongs to the chart, so it is
        # kept per start rather than per fly.
        self.optics = {
            start: [np.asarray(initial["luminance"]).copy(),
                    np.asarray(initial["r8_light"]).copy()]
            for start in set(self.starts)
        }
        self.groups = {}
        for b, start in enumerate(self.starts):
            self.groups.setdefault(start, []).append(b)

        # The memory rule's own state stays on the host, where the rule runs.
        self.rate_kc = np.tile(initial["rate_kc"], (B, 1))
        self.rate_dan = np.tile(initial["rate_dan"], (B, 1))
        self.memory_u = np.tile(initial["memory_u"], (B, 1))
        self.memory_w = np.tile(initial["memory_w"], (B, 1))

    def _launch(self, name, total, args):
        blocks = int((int(total) + BLOCK - 1) // BLOCK)
        if blocks:
            self.helper[name]((blocks,), (BLOCK,), args)

    # -- one observation ----------------------------------------------------

    def observe(self, frames, histories, kinds, executed, accounts,
                bars=None):
        """One market observation for every fly. `FlyController.observe`."""
        B, n = self.batch, self.n
        brain = self.brain
        seconds = self.bin * self.dt / 1000
        interval = self.bin * self.dt

        standing, pulses = [], []
        for b, genome in enumerate(self.genomes):
            self._sensory_parameters(genome)
            standing.append(standing_pulses(
                self.controller, histories[b], executed[b], accounts[b],
                None if bars is None else bars[b]))
            # Odour and taste are present for the whole observation; the
            # reinforcement pulse is not, and is appended per bin below.
            pulses.append(None if kinds[b] == "none" else
                          (self.circuit[kinds[b]],
                           np.float32(genome["pulse_current"])))

        left = self.ticks
        remaining_pulse = [self.pulse if p is not None else 0 for p in pulses]
        totals = np.zeros((B, int(self.cuts[-1])), np.int64)
        while left:
            clock = time.perf_counter() if self.measure else None
            for start, members in self.groups.items():
                settled, light = self.optics[start]
                head = members[0]
                brain.luminance[:] = settled
                brain.r8_light[:] = light
                samples, r8 = brain.rgb_bin(frames[head], interval, None)
                # rgb_bin advanced the photoreceptors once for this chart;
                # prepare_drive advances `luminance`, so every fly on this
                # chart has to start from the same unadvanced copy.
                light[:] = brain.r8_light
                before = brain.luminance.copy()
                for b in members:
                    brain.luminance[:] = before
                    brain.tonic[brain.lamina] = (
                        self.genomes[b]["lamina_bias"]
                        - WILD_TYPE["lamina_bias"])
                    extra = list(standing[b])
                    if remaining_pulse[b]:
                        extra.append(pulses[b])
                    brain.prepare_drive(samples, interval,
                                        stimulation=extra + r8)
                    self.drive_host[b] = brain.drive
                settled[:] = brain.luminance
            clock = self._mark("sensory", clock)
            self.drive.set(self.drive_host.reshape(-1))
            self._launch("set_drive", B * n,
                         (self.cell, self.drive, np.int64(B * n)))
            self._launch("clear_counts", B * n, (self.cell, np.int64(B * n)))
            clock = self._mark("upload", clock)
            self._advance(self.bin)
            clock = self._mark("kernel", clock)
            totals += self._rule(seconds)
            clock = self._mark("rule", clock)
            remaining_pulse = [max(0, x - self.bin) for x in remaining_pulse]
            left -= self.bin

        window = self.settings.neural_ms / 1000
        return [self._decode(totals[b], window) for b in range(B)]

    def _sensory_parameters(self, genome):
        """The genome's front end, on the one CPU brain the herd borrows."""
        o, t = self.controller.olfaction, self.controller.gustation
        if o is not None:
            o.current = genome["odor_current"]
            o.sigma = genome["odor_sigma"]
            o.floor = genome["odor_floor"]
        if t is not None:
            t.floor = genome["satiety_floor"]
            t.span = genome["satiety_span"]

    def _advance(self, steps):
        self.kernel((self.batch,), (BLOCK,), (
            np.int32(self.n), np.int32(self.batch), self.ptr, self.post,
            self.weight, np.int64(self.nedges), self.cell, self.slow,
            self.fixed, self.previous_drive, self.queue, self.queue_count,
            self.clock, np.int32(steps), np.float32(self.dt), self.active,
            self.nactive, np.int32(self.nplastic), self.edges, self.pre,
            self.baseline, self.gain, self.eta, self.adaptation_jump,
            self.adaptation_tau, self.kc_rest, self.tau_elig,
            np.float32(PARAMETERS["minimum_fraction"]),
            # The shipped path hands the kernel learning=False: the memory that
            # matters is written by rule.advance on the host, not by the
            # kernel's own plasticity.
            np.int32(0),
            self.av, self.ag, self.aa, self.am, self.ae,
            np.int32(self.delay), np.int32(self.rfc), np.int32(self.slots),
            self.scratch,
        ))

    def _rule(self, seconds):
        """`MemoryBrain.step`'s other half of a bin, once per fly.

        A loop, deliberately. This is 27% of a herd's time and the obvious fix
        is to batch it, so: measured, a batched `advance` over (flies, edges)
        arrays is *slower* -- 60.3s against 39.9s for the same work -- because
        one fly's traces are 63 kB and stay in cache while the herd's are 5 MB
        and do not.

        Nor is it call overhead. It is two matrix-vector products against
        `gain`, which is float32 while the rates are float64, so numpy promotes
        a 1 MB matrix on every one of them. 0.209 ms each, and nothing exact is
        faster: a contiguous transpose, einsum and np.dot are all either the
        same speed or a different answer. Handing `advance` a float64 `gain`
        is 7.5x quicker and changes the result by one unit in the last place,
        which is a change to the model and not a change to this file.
        """
        from stonkfly.neural.rule import advance

        cp, B = self.cp, self.batch
        k = int(self.cuts[-1])
        self._launch("gather_counts", k * B,
                     (self.cell, self.index, np.int32(k), np.int32(self.n),
                      np.int32(B), self.gathered))
        counts = cp.asnumpy(self.gathered).reshape(B, k)
        frozen = self.brain.weights_frozen
        gain = self.circuit["gain"]
        for b in range(B):
            advance(
                self.rate_kc[b], self.rate_dan[b], self.memory_u[b],
                self.memory_w[b],
                counts[b, self.cuts[0]:self.cuts[1]] / seconds,
                counts[b, self.cuts[1]:self.cuts[2]] / seconds
                - self.brain.dan_baseline_hz,
                gain, seconds, self.genomes[b]["eta"],
                self.settings.learning, frozen,
            )
        if not frozen:
            self.memory_w_device.set(self.memory_w.reshape(-1))
            self._launch("apply_memory", self.nplastic * B,
                         (self.weight, np.int64(self.nedges), self.edges,
                          self.baseline, self.memory_w_device,
                          np.int32(self.nplastic), np.int32(B)))
        return counts.astype(np.int64)

    def _decode(self, row, window):
        """`Decoder.decode`, on the gathered rows rather than all 166,700.

        The threshold is the controller's, not the genome's. `configured()`
        puts `decoder_threshold_hz` into a replaced Settings, and the Decoder
        read its threshold when the controller was built and never looks at
        Settings again -- so that gene does nothing in the CPU path either.
        Matching the CPU is the point of this class; the gene is reported as
        dead rather than quietly brought to life here.
        """
        a, b, c, e = (int(self.cuts[i]) for i in (2, 3, 4, 5))
        left = float(np.mean(row[a:b]) / window)
        right = float(np.mean(row[b:c]) / window)
        difference = right - left
        gate = int(row[c:e].sum())
        threshold = self.controller.decoder.threshold
        side = ("HOLD" if not gate or abs(difference) < threshold
                else "BUY" if difference > 0 else "SELL")
        return {
            "side": side,
            "left_hz": left,
            "right_hz": right,
            "difference_hz": difference,
            "gate_spikes": gate,
            "KC_spikes": int(row[e:].sum()),
        }


def replay_herd(herd, prices, observations, report=None, record=False,
                bars=None):
    """`evaluate.replay` for every fly at once, one account each.

    The chart is the price series and not anything a fly did, so flies sharing
    a start share it. What diverges is the account, and with it the
    reinforcement sign, the satiety current and the odour of the last fill.
    """
    settings = herd.settings
    B, starts = herd.batch, herd.starts
    unique = sorted(set(starts))
    need = CHART_WINDOW + WARMUP + observations
    short = [s for s in unique if s + need > len(prices)]
    if short:
        # `series.starts` never produces one of these. The CPU path would run
        # the affected fly short and report it as a full evaluation, which is a
        # wrong number rather than a missing one.
        raise ValueError(
            f"starts {short} leave fewer than {need} prices; the run would be "
            f"scored over fewer observations than it claims"
        )

    accounts = [Account(settings.capital, settings.order_limit,
                        settings.paper_fee) for _ in range(B)]
    history = {s: list(prices[s:s + CHART_WINDOW]) for s in unique}
    cursor = {s: s + CHART_WINDOW for s in unique}
    herd.reset()
    anchors = [str(a.start) for a in accounts]
    executed = [None] * B
    sides = [[] for _ in range(B)]
    # The continuous readout behind each proposal, kept only when asked. The
    # side is that number put through a threshold and a gate, and the two can
    # disagree about whether the brain knew anything: `tools/ic.py` measures
    # both, so a dead proposal stream can be told apart from a live readout
    # that the decoder threw away.
    readout = [[] for _ in range(B)]
    kc = [0] * B
    for step in range(WARMUP + observations):
        frame, quote, window = {}, {}, {}
        for start in unique:
            c = cursor[start]
            # Only the fast window is ever read, and it ends at the bar
            # whose close the fly is being shown. Handing over more would
            # not change the odour; handing over one more would be a look
            # at a bar that has not finished.
            if bars is not None:
                window[start] = bars[max(0, c - FAST + 1):c + 1]
            price = prices[c]
            bid, ask = quotes(price)
            quote[start] = (bid, ask, price)
            frame[start] = market_frame(PRODUCT, history[start], bid, ask)
        equities = [accounts[b].equity(quote[starts[b]][0]) for b in range(B)]
        kinds = [reinforcement(str(equities[b]), anchors[b], "0.01")[0]
                 for b in range(B)]
        out = herd.observe(
            [frame[starts[b]] for b in range(B)],
            [history[starts[b]] for b in range(B)],
            kinds, executed,
            [(str(equities[b]), anchors[b]) for b in range(B)],
            None if bars is None else [window[starts[b]] for b in range(B)],
        )
        for b in range(B):
            bid, ask, _ = quote[starts[b]]
            anchors[b] = str(equities[b])
            kc[b] += out[b]["KC_spikes"]
            if step >= WARMUP:
                sides[b].append(out[b]["side"])
                if record:
                    readout[b].append((out[b]["difference_hz"],
                                       out[b]["gate_spikes"]))
                executed[b] = accounts[b].apply(out[b]["side"], bid, ask)
            else:
                # Warm-up runs the network but never the account, so a fly is
                # judged from a charged mushroom body and a full balance.
                executed[b] = None
                accounts[b].cash, accounts[b].base = accounts[b].start, 0.0
        if report is not None:
            report(step, WARMUP + observations,
                   [accounts[b].equity(quote[starts[b]][0]) - accounts[b].start
                    for b in range(B)])
        for start in unique:
            history[start].append(quote[start][2])
            cursor[start] += 1

    rows = []
    for b in range(B):
        bid = quotes(prices[cursor[starts[b]] - 1])[0]
        final = accounts[b].equity(bid)
        rows.append({
            "profit": float(final - accounts[b].start),
            "final_equity": float(final),
            "observations": len(sides[b]),
            "buy": sides[b].count("BUY"),
            "sell": sides[b].count("SELL"),
            "hold": sides[b].count("HOLD"),
            "fills": dict(accounts[b].fills),
            "rejected": accounts[b].rejected,
            "kc_spikes": int(kc[b]),
        })
        if record:
            # Opt-in because a row is saved into the population state after
            # every generation, and a proposal per observation per fly per
            # start would add tens of thousands of strings to a file that is
            # read back on every resume. Only a diagnostic asks for them.
            rows[-1]["sides"] = list(sides[b])
            rows[-1]["difference_hz"] = [d for d, _ in readout[b]]
            rows[-1]["gate_spikes"] = [g for _, g in readout[b]]
    return rows


def evaluate_herd(herd, prices, observations, report=None):
    """`evaluate.evaluate` for every fly: profit, and profit over benchmark."""
    rows = replay_herd(herd, prices, observations, report)
    benchmark = {}
    for row, start in zip(rows, herd.starts):
        if start not in benchmark:
            benchmark[start] = buy_and_hold(prices, start, observations,
                                            herd.settings)
        row["buy_and_hold"] = benchmark[start]
        row["excess"] = row["profit"] - benchmark[start]
    return rows


class HerdRunner:
    """What the worker pool is to the CPU, one card is to a whole generation.

    Same interface as `pool.PoolRunner`, and the same rows out of it. The batch
    is taken from free device memory unless it is given: a fly costs 102 MB of
    weights and that is what limits it. There is nothing to gain by squeezing
    past eighty-four either way -- throughput stopped improving there.
    """

    def __init__(self, settings, batch=None, controller=None, pristine=None,
                 out=None):
        from .evaluate import build

        # Where the heartbeat goes. A generation writes nothing to disk until
        # it ends, which is half an hour of a watcher having to infer from CPU
        # load whether anything is happening. This writes one small file per
        # observation instead, so the question stops being a guess.
        # One process feeding a card is still one core at 100%, and the
        # policy for this project is that an evolution left running does not
        # make the machine unpleasant to use. Same call the worker pool makes.
        from .pool import deprioritise

        deprioritise()
        self.out = out
        self.stage = ""
        self.wave = (0, 0)
        self.beat = 0.0
        cuda_headers()
        import cupy as cp

        self.cp = cp
        self.settings = settings
        if controller is None:
            controller, pristine = build(settings)
        self.controller, self.pristine = controller, pristine
        brain = controller.brain
        # The delivery loop parallelises one source's edges without atomics,
        # which is sound only because no neuron lists the same target twice.
        # Checked once here rather than once per wave.
        from tools.gpu_port_check import assert_no_repeated_targets

        assert_no_repeated_targets(brain.ptr, brain.post)
        slots = int(round(1.8 / brain.dt)) + 1
        free = cp.cuda.Device().mem_info[0]
        room = capacity(free, brain.n, len(brain.post), slots)
        self.room = room
        self.batch = int(batch or min(room, 84))
        if self.batch > room:
            raise ValueError(
                f"a batch of {self.batch} does not fit in the "
                f"{free / 1e9:.1f} GB free on the device; {room} do"
            )

    def announce(self, stage):
        """What the loop is doing, for the heartbeat to say out loud."""
        self.stage = stage

    def _report(self, flies, starts, observations, warmup):
        """A callback that writes `progress.json`, throttled to twice a second.

        Atomic: written beside the target and renamed, so a watcher polling it
        never reads half a file. Never raises -- a run must not die because a
        status file could not be written.
        """
        if self.out is None:
            return None
        began = [None]

        def report(step, total, equity):
            now = time.time()
            if began[0] is None:
                began[0] = now
            if now - self.beat < 0.5 and step + 1 < total:
                return
            self.beat = now
            done = (step + 1) * flies
            state = {
                "stage": self.stage,
                "wave": self.wave[0], "waves": self.wave[1],
                "flies": flies,
                "starts": sorted(set(starts)),
                "observation": step + 1, "observations": total,
                "warmup": warmup,
                "rate": done / max(1e-9, now - began[0]),
                "profit": [round(float(x), 6) for x in equity],
                "updated": now,
            }
            try:
                path = Path(self.out) / "progress.json"
                temporary = path.with_suffix(".partial")
                temporary.write_text(json.dumps(state), encoding="utf-8")
                temporary.replace(path)
            except OSError:
                pass

        return report

    def describe(self):
        name = self.cp.cuda.runtime.getDeviceProperties(0)["name"]
        if isinstance(name, bytes):
            name = name.decode()
        return f"{name}, {self.batch} flies a wave"

    def _wave(self, genomes, starts, prices, observations):
        herd = Herd(self.controller, self.pristine, genomes, self.settings,
                    starts)
        try:
            return evaluate_herd(
                herd, prices, observations,
                self._report(len(genomes), starts, observations, WARMUP))
        finally:
            del herd
            self.cp.get_default_memory_pool().free_all_blocks()

    def evaluate(self, genomes, prices, offsets, observations):
        """Rows per genome, one per start, in the order the offsets came in."""
        tasks = [(i, start) for i in range(len(genomes)) for start in offsets]
        rows = [[] for _ in genomes]
        waves = (len(tasks) + self.batch - 1) // self.batch
        for index, cut in enumerate(range(0, len(tasks), self.batch), start=1):
            self.wave = (index, waves)
            wave = tasks[cut:cut + self.batch]
            done = self._wave([genomes[i] for i, _ in wave],
                              [s for _, s in wave], prices, observations)
            for (i, _), row in zip(wave, done):
                rows[i].append(row)
        return rows

    def baselines(self, prices, offsets, observations, seed):
        """One dict per start. Three of the four never touch the brain."""
        import random

        from .evaluate import (Account, fixed_proposal, random_proposal,
                               replay)
        from .genome import WILD_TYPE

        out = [{} for _ in offsets]
        for i, start in enumerate(offsets):
            rng = random.Random(seed + i)
            for name, propose in [("buy_and_hold", fixed_proposal("BUY")),
                                  ("all_cash", fixed_proposal("HOLD")),
                                  ("random", random_proposal(rng))]:
                out[i][name] = replay(
                    self.controller, prices, start, observations, propose,
                    Account(self.settings.capital, self.settings.order_limit,
                            self.settings.paper_fee),
                )
        # The fourth is a real fly, so it goes through the card like the rest.
        wild = self._wave([dict(WILD_TYPE)] * len(offsets), list(offsets),
                          prices, observations)
        for i, row in enumerate(wild):
            out[i]["wild_type"] = row
        return out
