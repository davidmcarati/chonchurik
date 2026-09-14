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
    """Below-normal scheduling priority, on either platform."""
    if sys.platform == "win32":
        import ctypes

        BELOW_NORMAL_PRIORITY_CLASS = 0x00004000
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        ctypes.windll.kernel32.SetPriorityClass(handle, BELOW_NORMAL_PRIORITY_CLASS)
    else:
        try:
            os.nice(10)
        except OSError:
            pass


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
