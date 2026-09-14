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
OPTIONS = ("--fmad=false", "--ftz=false", "--prec-div=true",
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
extern "C" __declspec(dllexport) void build(float dt, float tau_a,
                                            float* av, float* ag, float* aa) {
  for (int i = 0; i < 1024; i++) {
    av[i] = std::exp(-dt * i / 20.f);
    ag[i] = std::exp(-dt * i / 5.f);
    aa[i] = std::exp(-dt * i / tau_a);
  }
}
"""


def tables(dt, adaptation_tau):
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
    arrays = [np.zeros(1024, np.float32) for _ in range(3)]
    lib.build(ctypes.c_float(dt), ctypes.c_float(adaptation_tau),
              *[x.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
                for x in arrays])
    return tuple(arrays)


def run_gpu(cp, kernel, brain, state, steps, batch, settings):
    """One launch; the whole tick loop lives inside the kernel."""
    from stonkfly.neural.brain import PARAMETERS

    c = brain.circuit
    dt = np.float32(brain.dt)
    tau_a = np.float32(brain.adaptation_tau)
    av, ag, aa = (cp.asarray(x) for x in tables(brain.dt, brain.adaptation_tau))
    delay = int(round(1.8 / brain.dt))
    rfc = int(round(2.2 / brain.dt))
    slots = delay + 1

    def tile(name, dtype=None):
        a = np.asarray(state[name])
        a = a.astype(dtype) if dtype else a
        return cp.asarray(np.tile(a, (batch,) + (1,) * (a.ndim - 1))
                          if a.ndim else np.repeat(a, batch))

    d = {name: tile(name) for name in
         ["v", "g", "refractory", "previous_drive", "counts", "active",
          "last", "eligibility", "eligibility_last", "modulation",
          "modulation_last", "adaptation", "weight"]}
    d["flags"] = cp.asarray(np.tile(state["active_flag"], batch))
    d["queue"] = cp.asarray(np.tile(state["queue"], batch))
    d["queue_count"] = cp.asarray(np.tile(state["queue_count"], batch))
    d["nactive"] = cp.asarray(np.tile(state["nactive"], batch).astype(np.int32))
    d["clock"] = cp.asarray(np.full(batch, state["clock"][0], np.int64))
    drive = cp.asarray(np.tile(np.asarray(brain.drive), batch))

    args = (
        np.int32(brain.n), np.int32(batch), cp.asarray(brain.ptr),
        cp.asarray(brain.post), d["weight"], d["v"], d["g"],
        d["refractory"], drive, d["previous_drive"], d["queue"],
        d["queue_count"], d["clock"], np.int32(steps), dt, d["counts"],
        d["active"], d["flags"], d["nactive"], d["last"],
        cp.asarray(c["kc_mask"]), cp.asarray(c["dan_index"].astype(np.int8)),
        d["eligibility"], d["eligibility_last"], np.int32(len(c["edges"])),
        cp.asarray(c["edges"]), cp.asarray(c["pre"]),
        cp.asarray(brain.baseline_plastic), cp.asarray(c["gain"]),
        np.float32(brain.eta),
        np.float32(PARAMETERS["trace_kc_seconds"] * 1000),
        np.float32(PARAMETERS["minimum_fraction"]), np.int32(0),
        d["modulation"], d["modulation_last"], cp.asarray(brain.modulation_mask),
        cp.asarray(brain.rest), d["adaptation"], np.float32(brain.adaptation_jump),
        tau_a, av, ag, aa, np.int32(delay), np.int32(rfc), np.int32(slots),
        cp.zeros(1, cp.int32),
    )
    cp.cuda.Stream.null.synchronize()
    started = time.perf_counter()
    kernel((batch,), (BLOCK,), args)
    cp.cuda.Stream.null.synchronize()
    seconds = time.perf_counter() - started

    n = brain.n
    out = {}
    for name in ["v", "g", "refractory", "previous_drive", "counts", "active",
                 "last", "eligibility", "eligibility_last", "modulation",
                 "modulation_last", "adaptation"]:
        out[name] = cp.asnumpy(d[name])[:n]
    out["active_flag"] = cp.asnumpy(d["flags"])[:n]
    out["queue"] = cp.asnumpy(d["queue"])[:len(state["queue"])]
    out["queue_count"] = cp.asnumpy(d["queue_count"])[:len(state["queue_count"])]
    out["nactive"] = cp.asnumpy(d["nactive"])[:1].astype(state["nactive"].dtype)
    out["weight"] = cp.asnumpy(d["weight"])[:len(state["weight"])]
    out["clock"] = cp.asnumpy(d["clock"])[:1]
    return out, seconds


def run_cpu(brain, steps):
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
        0, brain.modulation.ctypes.data, brain.modulation_last.ctypes.data,
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

    brain.counts.fill(0)
    before = snapshot(brain, brain.cursor)
    print(f"starting from clock {brain.cursor:,}, "
          f"{int(brain.nactive[0]):,} active\n")

    kernel = cp.RawKernel(SOURCE.read_text(encoding="utf-8"),
                          "memory_advance_batch", options=OPTIONS)
    gpu, gpu_seconds = run_gpu(cp, kernel, brain, before, a.steps, a.batch,
                               controller.s)
    print(f"GPU {a.steps} ticks x {a.batch} genome(s): {gpu_seconds:.3f}s")

    for name, value in before.items():
        if name != "clock":
            np.asarray(getattr(brain, name))[...] = value
    brain.cursor = int(before["clock"][0])
    # brain.advance directly, not _neural_step: the latter recomputes `drive`
    # from the frame before advancing, so the two kernels would run on
    # different input and every difference afterwards would be the harness's.
    cpu_seconds = run_cpu(brain, a.steps)
    cpu = snapshot(brain, brain.cursor)
    print(f"CPU {a.steps} ticks x 1 genome:      {cpu_seconds:.3f}s"
          f"   ({cpu_seconds / max(gpu_seconds, 1e-9) * a.batch:.1f}x per genome)\n")

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
