"""A bounded worker pool that leaves the machine usable.

Eight workers by default, not one per thread. Each holds its own brain --
about a gigabyte -- and runs below normal priority, so an evolution left
running in the background yields to whatever the machine is actually for.
Raise it with --workers for an unattended night; the default is chosen for a
machine someone is sitting at.
"""

import os
import sys
from concurrent.futures import ProcessPoolExecutor

DEFAULT_WORKERS = 8
_STATE = {}


def deprioritise():
    """Below-normal scheduling priority, on either platform.

    The Windows branch declares its argument types. Without them ctypes gives
    GetCurrentProcess a C `int` return, so the pseudo-handle (HANDLE)-1 comes
    back as a 32-bit -1, SetPriorityClass rejects it and returns 0, and the
    process quietly keeps running at normal priority -- which is what shipped,
    and what made "eight workers that leave the machine usable" untrue on the
    one platform this repository is developed on. A failure here is raised
    rather than ignored: a background run that silently competes with the
    interactive session is the whole thing this function exists to prevent.
    """
    if sys.platform == "win32":
        import ctypes

        BELOW_NORMAL_PRIORITY_CLASS = 0x00004000
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        kernel32.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        kernel32.SetPriorityClass.restype = ctypes.c_int
        ok = kernel32.SetPriorityClass(
            kernel32.GetCurrentProcess(), BELOW_NORMAL_PRIORITY_CLASS
        )
        if not ok:
            raise ctypes.WinError(ctypes.get_last_error())
    else:
        try:
            os.nice(10)
        except OSError:
            pass


def priority():
    """What the scheduler actually thinks, for a test to check."""
    if sys.platform == "win32":
        import ctypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        kernel32.GetPriorityClass.argtypes = [ctypes.c_void_p]
        kernel32.GetPriorityClass.restype = ctypes.c_uint32
        return int(kernel32.GetPriorityClass(kernel32.GetCurrentProcess()))
    return os.getpriority(os.PRIO_PROCESS, 0)


def initialise(settings):
    """One brain per worker process, built once and reconfigured per genome."""
    from .evaluate import build, graph_snapshot

    deprioritise()
    # The kernel is single-threaded; letting BLAS also fan out would put every
    # worker in competition with every other one for the same cores.
    for name in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"]:
        os.environ.setdefault(name, "1")
    controller, pristine = build(settings)
    _STATE["controller"] = controller
    _STATE["pristine"] = pristine
    _STATE["settings"] = settings
    _STATE["graph"] = graph_snapshot(controller.brain)


def run_one(task):
    """Evaluate one genome at one chronological start, inside a worker."""
    from .evaluate import evaluate, graph_unchanged

    genome, prices, start, observations = task
    controller = _STATE["controller"]
    row = evaluate(
        controller, _STATE["pristine"], genome, prices, start, observations,
        _STATE["settings"],
    )
    # A genome that leaked into the graph would quietly contaminate every
    # later genome this worker sees, and the contamination would look like
    # evolution working.
    if not graph_unchanged(controller.brain, _STATE["graph"]):
        raise AssertionError("A genome modified the retained graph")
    return row


def run_baselines(task):
    import random

    from .evaluate import baselines

    prices, start, observations, seed = task
    return baselines(
        _STATE["controller"], _STATE["pristine"], prices, start, observations,
        _STATE["settings"], random.Random(seed),
    )


def pool(settings, workers=DEFAULT_WORKERS):
    return ProcessPoolExecutor(
        max_workers=workers, initializer=initialise, initargs=(settings,)
    )


class PoolRunner:
    """The worker pool behind the interface the GPU herd also offers.

    The evolution asks a runner for rows and does not care where they came
    from, which is the only reason a card can be swapped in for the pool
    without the loop knowing. `tools/herd_check.py` is what says the two
    runners agree; this class is just the pool, unchanged, wearing the shape.
    """

    def __init__(self, executor, workers=DEFAULT_WORKERS):
        self.executor = executor
        self.workers = workers

    def describe(self):
        return f"{self.workers} worker processes at below-normal priority"

    def evaluate(self, genomes, prices, offsets, observations):
        """One (genome, start) task per future; the pool decides the packing."""
        futures = {}
        for index, individual in enumerate(genomes):
            for start in offsets:
                futures[self.executor.submit(
                    run_one, (individual, prices, start, observations)
                )] = index
        rows = [[] for _ in genomes]
        for future, index in futures.items():
            rows[index].append(future.result())
        return rows

    def baselines(self, prices, offsets, observations, seed):
        futures = [
            self.executor.submit(run_baselines,
                                 (prices, start, observations, seed + i))
            for i, start in enumerate(offsets)
        ]
        return [f.result() for f in futures]
