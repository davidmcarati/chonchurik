"""Run kernel.cu and kernel.cpp from one state and compare every array.

    python -m tools.gpu_port_check --steps 100
    python -m tools.gpu_port_check --steps 5000 --batch 4

Both kernels advance the same brain from the same starting state by the same
number of ticks. Afterwards every array the kernel may touch is compared
elementwise, by the same comparator `tools.kernel_equivalence` uses. A port
that agrees on spike counts and disagrees on one voltage is a different model,
and this is what says so.

Nothing here is wired into a run: it builds its own brain, never writes a run
directory and never touches the CPU kernel the model actually uses.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from stonkfly.config import Settings

BLOCK = 256
# Double-dashed: NVRTC silently ignores an option it does not recognise, so a
# single dash here would leave contraction and denormal flushing on while
# looking as though they were off.
OPTIONS = ("--fmad=false", "--prec-div=true",
           "--prec-sqrt=true")
SOURCE = Path(__file__).resolve().parent.parent / "stonkfly" / "neural" / "kernel.cu"


def find_cuda_headers():
    import os
    import site

    if os.environ.get("CUDA_PATH"):
        return
    for root in site.getsitepackages():
        candidate = Path(root) / "nvidia" / "cuda_runtime"
        if (candidate / "include" / "cuda_runtime.h").exists():
            os.environ["CUDA_PATH"] = str(candidate)
            return


def assert_no_repeated_targets(ptr, post, samples=4000, seed=0):
    """The delivery loop parallelises one source's edges without atomics.

    That is only safe because no neuron lists the same target twice. Checked
    here rather than assumed, because if it were ever false the kernel would
    lose additions silently and look merely slightly different.
    """
    rng = np.random.default_rng(seed)
    n = len(ptr) - 1
    for i in rng.choice(n, min(samples, n), replace=False):
        row = post[ptr[i]:ptr[i + 1]]
        if row.size and np.unique(row).size != row.size:
            raise AssertionError(
                f"neuron {i} lists a target twice; the delivery loop needs "
                "atomics and the port is unsound as written"
            )


MUTABLE = ["v", "g", "refractory", "previous_drive", "queue", "queue_count",
           "counts", "active", "active_flag", "nactive", "last", "eligibility",
           "eligibility_last", "modulation", "modulation_last", "adaptation",
           "weight"]


def snapshot(brain, clock):
    out = {k: np.asarray(getattr(brain, k)).copy() for k in MUTABLE}
    out["clock"] = np.asarray([clock], dtype=np.int64)
    return out


TABLE_SOURCE = """// The decay tables exactly as kernel.cpp builds them, same compiler, same
// libm. numpy's exp differs from MSVC's in 40% of the 1024 entries -- by one
// unit in the last place, which is enough to make every voltage downstream
// disagree and look like a race in the port.
#include <cmath>
#define ENTRIES (1 << 20)
extern "C" __declspec(dllexport) void build(float dt, float tau_a,
                                            float tau_elig,
                                            float* av, float* ag, float* aa,
                                            double* am, double* ae) {
  for (int i = 0; i < ENTRIES; i++) {
    av[i] = std::exp(-dt * i / 20.f);
    ag[i] = std::exp(-dt * i / 5.f);
    aa[i] = std::exp(-dt * i / tau_a);
    // The kernel decays these two inline rather than from a table; the
    // argument is still -dt times a whole number of ticks, so it tabulates
    // exactly the same way and stops the device's exp from disagreeing.
    //
    // Double, unlike the three above. The voltage decays are multiplied into
    // a float and rounded once, so a float table is exactly what the host
    // computes. These two are not: eligibility is a double throughout, and
    // modulation is a float times the double result. Rounding the table to
    // float first rounds twice, and the extra rounding survives.
    am[i] = std::exp(-dt * i / 100.f);
    ae[i] = std::exp(-dt * i / tau_elig);
  }
}
"""


def tables(dt, adaptation_tau, tau_elig):
    """Built by the kernel's own compiler, not by numpy.

    The GPU has to be handed the same numbers the CPU kernel computes for
    itself, and `np.exp(...).astype(float32)` is not those numbers: it rounds
    twice, through float64, and disagrees with MSVC's `std::exp` in 40% of the
    entries. Every one of those is one unit in the last place, and every
    voltage that touches one inherits it.
    """
    import ctypes
    import shutil
    import subprocess
    import tempfile

    from stonkfly.neural.brain import msvc_environment

    out = Path(tempfile.mkdtemp(prefix="stonkfly-tables-"))
    (out / "tab.cpp").write_text(TABLE_SOURCE, encoding="utf-8")
    env = msvc_environment()
    if sys.platform == "win32":
        cl = shutil.which("cl", path=env["PATH"] if env else None)
        argv = [cl, "/nologo", "/O2", "/std:c++17", "/EHsc", "/LD", "tab.cpp",
                "/Fe:tab.dll"]
        library = out / "tab.dll"
    else:
        argv = ["c++", "-O3", "-std=c++17", "-shared", "-fPIC", "tab.cpp",
                "-o", "tab.so"]
        library = out / "tab.so"
    subprocess.run(argv, cwd=out, env=env, check=True, capture_output=True)
    lib = ctypes.CDLL(str(library))
    arrays = [np.zeros(1 << 20, np.float32) for _ in range(3)]
    arrays += [np.zeros(1 << 20, np.float64) for _ in range(2)]
    lib.build(ctypes.c_float(dt), ctypes.c_float(adaptation_tau),
              ctypes.c_float(tau_elig),
              *[x.ctypes.data_as(ctypes.c_void_p) for x in arrays])
    return tuple(arrays)


# The kernel keeps a neuron's state in one 32-byte struct rather than in a
# dozen parallel arrays, because a scattered read costs a whole 32-byte sector
# whatever it asks for, and a visit to a cell used to pay for twelve of them to
# use four bytes of each. These are those structs as numpy sees them; the sizes
# are asserted below, since a padding surprise would silently misread
# everything without changing a single number in the source.
CELL = np.dtype([("last", "<i8"), ("v", "<f4"), ("g", "<f4"),
                 ("adaptation", "<f4"), ("drive", "<f4"), ("counts", "<i4"),
                 ("refractory", "<i2"), ("flags", "u1"), ("pad", "u1")])
SLOW = np.dtype([("eligibility", "<f8"), ("eligibility_last", "<i8"),
                 ("modulation_last", "<i8"), ("modulation", "<f4"),
                 ("pad", "<f4")])
FIXED = np.dtype([("rest", "<f4"), ("kc", "u1"), ("modulatory", "u1"),
                  ("pad", "u1"), ("dan", "i1")])
assert (CELL.itemsize, SLOW.itemsize, FIXED.itemsize) == (32, 32, 8)

# Which numpy array fills which field. `drive` is read-only during a launch but
# rides along, because a visit needs it and a separate array would cost the
# sector this whole arrangement exists to save.
CELL_FIELDS = {"last": "last", "v": "v", "g": "g", "adaptation": "adaptation",
               "drive": None, "counts": "counts", "refractory": "refractory",
               "flags": "active_flag"}
SLOW_FIELDS = {"eligibility": "eligibility",
               "eligibility_last": "eligibility_last",
               "modulation_last": "modulation_last",
               "modulation": "modulation"}


def pack(dtype, fields, state, n, extra=None):
    out = np.zeros(n, dtype)
    for field, name in fields.items():
        if name is not None:
            out[field] = np.asarray(state[name])
    for field, value in (extra or {}).items():
        out[field] = value
    return out


def unpack(packed, fields, state, out):
    """Back into the arrays the comparator expects, at their own dtypes."""
    for field, name in fields.items():
        if name is not None:
            out[name] = packed[field].astype(np.asarray(state[name]).dtype)


def run_gpu(cp, kernel, brain, state, steps, batch, settings, block=BLOCK,
            learning=0):
    """One launch; the whole tick loop lives inside the kernel."""
    from stonkfly.neural.brain import PARAMETERS

    c = brain.circuit
    n = brain.n
    dt = np.float32(brain.dt)
    tau_a = np.float32(brain.adaptation_tau)
    tau_elig = np.float32(PARAMETERS["trace_kc_seconds"] * 1000)
    av, ag, aa, am, ae = (
        cp.asarray(x)
        for x in tables(brain.dt, brain.adaptation_tau, float(tau_elig)))
    # The resting potential of a Kenyon cell is an evolved parameter and the
    # resting potential of everything else is not, so the kernel takes one
    # number and the mask, rather than an array per genome.
    kc_rest = float(np.asarray(brain.rest)[c["kc"][0]])
    delay = int(round(1.8 / brain.dt))
    rfc = int(round(2.2 / brain.dt))
    slots = delay + 1

    def tile(array):
        """Genome-major: the kernel offsets each genome by a whole array.

        Always raveled first. `np.tile` on a 2-D array with a scalar repeat
        tiles the last axis, so `queue`, which numpy holds as (slots, n), came
        out interleaved by row -- genome 0's slot 0 beside genome 1's slot 0 --
        and every delivery after that read another genome's spikes. Tiled on
        the device, because at a useful batch the host copy alone is gigabytes.
        """
        return cp.tile(cp.asarray(np.ascontiguousarray(array).ravel()), batch)

    cells = pack(CELL, CELL_FIELDS, state, n,
                 {"drive": np.asarray(brain.drive)})
    slow = pack(SLOW, SLOW_FIELDS, state, n)
    fixed = np.zeros(n, FIXED)
    fixed["rest"] = brain.rest
    fixed["kc"] = c["kc_mask"]
    fixed["modulatory"] = brain.modulation_mask
    fixed["dan"] = c["dan_index"]

    d = {
        "cell": tile(cells.view(np.uint8)),
        "slow": tile(slow.view(np.uint8)),
        "previous_drive": tile(state["previous_drive"]),
        "queue": tile(state["queue"]),
        "queue_count": tile(state["queue_count"]),
        "active": tile(state["active"]),
        # One copy for everyone while the plasticity rule is off, which is
        # 102 MB instead of 102 MB a genome and the difference between
        # fitting 84 genomes on the card and fitting several hundred.
        "weight": (tile(state["weight"]) if learning
                   else cp.asarray(np.asarray(state["weight"]))),
        "nactive": tile(np.asarray(state["nactive"]).astype(np.int32)),
        "clock": cp.asarray(np.full(batch, state["clock"][0], np.int64)),
    }

    args = (
        np.int32(n), np.int32(batch), cp.asarray(brain.ptr),
        cp.asarray(brain.post), d["weight"],
        np.int64(len(brain.post) if learning else 0),
        d["cell"], d["slow"],
        cp.asarray(fixed.view(np.uint8)), d["previous_drive"], d["queue"],
        d["queue_count"], d["clock"], np.int32(steps), dt, d["active"],
        d["nactive"], np.int32(len(c["edges"])), cp.asarray(c["edges"]),
        cp.asarray(c["pre"]), cp.asarray(brain.baseline_plastic),
        cp.asarray(c["gain"]),
        # Per genome, because the evolution moves them. Here every genome is
        # the same brain, which is the point: the comparison is against the
        # CPU kernel, and the CPU kernel has one physiology.
        cp.full(batch, brain.eta, np.float32),
        cp.full(batch, brain.adaptation_jump, np.float32),
        cp.full(batch, brain.adaptation_tau, np.float32),
        cp.full(batch, kc_rest, np.float32),
        tau_elig, np.float32(PARAMETERS["minimum_fraction"]),
        np.int32(learning),
        av, ag, cp.tile(aa, batch), am, ae,
        np.int32(delay), np.int32(rfc), np.int32(slots),
        cp.zeros(1, cp.int32),
    )
    cp.cuda.Stream.null.synchronize()
    started = time.perf_counter()
    kernel((batch,), (block,), args)
    cp.cuda.Stream.null.synchronize()
    seconds = time.perf_counter() - started

    # Genome 0's slice, back into the arrays the comparator expects.
    out = {}
    unpack(cp.asnumpy(d["cell"])[:n * CELL.itemsize].view(CELL),
           CELL_FIELDS, state, out)
    unpack(cp.asnumpy(d["slow"])[:n * SLOW.itemsize].view(SLOW),
           SLOW_FIELDS, state, out)
    out["previous_drive"] = cp.asnumpy(d["previous_drive"])[:n]
    q = np.asarray(state["queue"])
    out["queue"] = cp.asnumpy(d["queue"])[:q.size].reshape(q.shape)
    qc = np.asarray(state["queue_count"])
    out["queue_count"] = cp.asnumpy(d["queue_count"])[:qc.size].reshape(qc.shape)
    out["active"] = cp.asnumpy(d["active"])[:n]
    out["nactive"] = cp.asnumpy(d["nactive"])[:1].astype(state["nactive"].dtype)
    out["weight"] = cp.asnumpy(d["weight"])[:len(state["weight"])]
    out["clock"] = cp.asnumpy(d["clock"])[:1]
    return out, seconds


def run_cpu(brain, steps, learning=0):
    """The shipped kernel, on exactly the state and drive the GPU was given."""
    from stonkfly.neural.brain import PARAMETERS

    c = brain.circuit
    clock = np.asarray([brain.cursor], dtype=np.int64)
    arrays = [brain.ptr, brain.post, brain.weight, brain.v, brain.g,
              brain.refractory, brain.drive, brain.previous_drive, brain.queue,
              brain.queue_count, clock]
    started = time.perf_counter()
    brain.advance(
        brain.n, *[x.ctypes.data for x in arrays], steps, brain.dt,
        *[getattr(brain, k).ctypes.data for k in
          ["counts", "active", "active_flag", "nactive", "last"]],
        c["kc_mask"].ctypes.data, c["dan_index"].ctypes.data,
        brain.eligibility.ctypes.data, brain.eligibility_last.ctypes.data,
        len(c["edges"]), c["edges"].ctypes.data, c["pre"].ctypes.data,
        brain.baseline_plastic.ctypes.data, c["gain"].ctypes.data, brain.eta,
        PARAMETERS["trace_kc_seconds"] * 1000, PARAMETERS["minimum_fraction"],
        learning, brain.modulation.ctypes.data, brain.modulation_last.ctypes.data,
        brain.modulation_mask.ctypes.data, brain.rest.ctypes.data,
        brain.adaptation.ctypes.data, brain.adaptation_jump,
        brain.adaptation_tau,
    )
    elapsed = time.perf_counter() - started
    brain.cursor = int(clock[0])
    return elapsed


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--reinforce", choices=("none", "reward", "aversive"),
                   default="none",
                   help="drive the dopaminergic cells hard enough to fire for "
                        "the whole comparison, so the plasticity rule actually "
                        "runs. Nothing in an offline fixture makes a DAN "
                        "spike, and with no DAN the two weight arrays agree "
                        "because neither kernel writes to them")
    p.add_argument("--dan-current", type=float, default=400.0,
                   help="how hard. This is a stress test of a code path, not "
                        "the model's pulse: the model uses 40, and at 40 these "
                        "cells sit at -108 mV under inhibition and never reach "
                        "threshold in an observation")
    p.add_argument("--learning", action="store_true",
                   help="run the plasticity rule on both sides. Off, the "
                        "weight arrays agree because neither kernel writes to "
                        "them, which proves nothing about the rule")
    p.add_argument("--warmup", type=int, default=2,
                   help="observations run on the CPU before the comparison, so "
                        "the state is a used brain rather than a fresh one")
    a = p.parse_args()

    find_cuda_headers()
    import cupy as cp

    from stonkfly.neural.controller import FlyController

    from .evolve.evaluate import CHART_WINDOW
    from .kernel_cost import observe, series
    from .kernel_equivalence import differences

    print("building…", flush=True)
    controller = FlyController(Settings())
    brain = controller.brain
    assert_no_repeated_targets(brain.ptr, brain.post)
    print(f"  {brain.n:,} neurons, {len(brain.post):,} edges; no neuron "
          f"repeats a target")

    prices = series(CHART_WINDOW + a.warmup + 2)
    for i in range(a.warmup):
        observe(controller, prices[:CHART_WINDOW + i + 1])

    if a.reinforce != "none":
        cells = brain.circuit[a.reinforce]
        brain.drive[cells] += np.float32(a.dan_current)
        print(f"driving {len(cells)} {a.reinforce} cells at {a.dan_current:g} "
              f"so the plasticity rule has something to run on")

    brain.counts.fill(0)
    before = snapshot(brain, brain.cursor)
    print(f"starting from clock {brain.cursor:,}, "
          f"{int(brain.nactive[0]):,} active\n")

    kernel = cp.RawKernel(SOURCE.read_text(encoding="utf-8"),
                          "memory_advance_batch", options=OPTIONS)
    gpu, gpu_seconds = run_gpu(cp, kernel, brain, before, a.steps, a.batch,
                               controller.s, learning=int(a.learning))
    print(f"GPU {a.steps} ticks x {a.batch} genome(s): {gpu_seconds:.3f}s")

    for name, value in before.items():
        if name != "clock":
            np.asarray(getattr(brain, name))[...] = value
    brain.cursor = int(before["clock"][0])
    # brain.advance directly, not _neural_step: the latter recomputes `drive`
    # from the frame before advancing, so the two kernels would run on
    # different input and every difference afterwards would be the harness's.
    cpu_seconds = run_cpu(brain, a.steps, int(a.learning))
    cpu = snapshot(brain, brain.cursor)
    print(f"CPU {a.steps} ticks x 1 genome:      {cpu_seconds:.3f}s"
          f"   ({cpu_seconds / max(gpu_seconds, 1e-9) * a.batch:.1f}x per genome)\n")

    # Without this line "identical" could mean the rule never ran: an
    # untouched weight array agrees with an untouched weight array.
    moved = int(np.count_nonzero(cpu["weight"] != before["weight"]))
    spikes = int(cpu["counts"].sum())
    print(f"the run itself: {spikes:,} spikes, {moved:,} weights moved"
          f"{'' if a.learning else '   (plasticity off)'}")

    bad = differences(cpu, gpu)
    if not bad:
        print(f"IDENTICAL: all {len(cpu)} arrays, every element")
        return
    print(f"{len(bad)} of {len(cpu)} arrays differ:")
    for name, detail, gap, where in bad:
        extra = "" if gap is None else f"   max |diff| {gap:.3e}   first at {where:,}"
        print(f"  {name:18} {detail}{extra}")


if __name__ == "__main__":
    main()
