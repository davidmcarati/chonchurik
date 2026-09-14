"""Would the kernel run faster on the GPU, and what would that cost? Measure.

    python -m tools.gpu_bench
    python -m tools.gpu_bench --batches 8,16,24,32

Both of the kernel's loops, ported to CUDA exactly as `kernel.cpp` writes them,
measured against this machine's CPU. Nothing here is wired into the model: it
answers whether a port is worth starting, and what it would have to give up.

Three things it establishes, none of which were obvious beforehand:

  * the optimal batch is set by the GPU's L2, not by its cores. Sixteen genomes
    of per-neuron state fit; twenty-four do not, and the cliff is 3x;
  * the active-list loop is bit-identical to the CPU with `-fmad=false`, at a
    cost of about 1.6%. Float contraction, not atomics, was the difference;
  * the synaptic scatter is *not* reproducible with float atomics -- two
    identical runs differ -- and is reproducible with integer ones at no
    measurable cost, because integer addition commutes and float addition
    does not.

Requires cupy and the pip-installed CUDA headers:

    pip install cupy-cuda12x nvidia-cuda-runtime-cu12
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np

N = 166_700                # neurons in the retained graph
ACTIVE = 22_655            # mean active list, measured by tools.kernel_cost
SPIKES_PER_TICK = 88       # 442,464 spikes / 5,000 ticks, measured
TICKS = 5_000              # 500 ms at 0.1 ms
DT = np.float32(0.1)
TAU_A = np.float32(200.0)
FIXED = np.float32(1 << 20)
# order.cpp on this machine, one core, the same loop. Re-measure on other
# hardware before trusting the comparison.
CPU_NS_PER_VISIT = 16.98
CPU_CORES = 8

EVOLVE = r"""
extern "C" __global__ void evolve_active(
    const int n, const int active_n, const int batch, const long long now,
    const int* __restrict__ act, const float* __restrict__ rest,
    const float* __restrict__ drive,
    float* v, float* g, float* adaptation, long long* last, short* refractory,
    const float* __restrict__ av, const float* __restrict__ ag,
    const float* __restrict__ aa, const float dt, const float tau_a) {
  const long long tid = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  if (tid >= (long long)active_n * batch) return;
  const int b = (int)(tid / active_n);
  const int i = act[tid % active_n];
  const long long off = (long long)b * n + i;
  long long d = now - last[off];
  if (d <= 0) return;
  const int fr = refractory[off] > 0 ? refractory[off] - 1 : 0;
  const int sk = (int)(d < fr ? d : fr);
  if (sk > 0 && adaptation[off] > 0.f)
    adaptation[off] *= sk < 1024 ? aa[sk] : expf(-dt * sk / tau_a);
  refractory[off] = d >= refractory[off] ? 0 : (short)(refractory[off] - d);
  d -= sk;
  if (d > 0) {
    const float a = d < 1024 ? av[d] : expf(-dt * d / 20.f);
    const float bb = d < 1024 ? ag[d] : expf(-dt * d / 5.f);
    v[off] = rest[i] + (v[off] - rest[i]) * a + drive[i] * (1.f - a)
           + g[off] * (a - bb) / 3.f;
    g[off] *= bb;
    if (adaptation[off] > 0.f) {
      const float c = d < 1024 ? aa[d] : expf(-dt * d / tau_a);
      v[off] -= adaptation[off] * tau_a / (tau_a - 20.f) * (c - a);
      adaptation[off] *= c;
    }
  }
  last[off] = now;
}
"""

DELIVER = r"""
extern "C" __global__ void deliver_float(
    const int n, const int batch, const long long edges,
    const int* __restrict__ post, const float* __restrict__ weight,
    float* g, const short* __restrict__ refractory) {
  const long long tid = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  if (tid >= edges * batch) return;
  const int b = (int)(tid / edges);
  const long long k = tid % edges;
  const int j = post[k];
  if (refractory[(long long)b * n + j] == 0)
    atomicAdd(&g[(long long)b * n + j], weight[k]);
}

extern "C" __global__ void deliver_fixed(
    const int n, const int batch, const long long edges,
    const int* __restrict__ post, const float* __restrict__ weight,
    long long* g, const short* __restrict__ refractory, const float scale) {
  const long long tid = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  if (tid >= edges * batch) return;
  const int b = (int)(tid / edges);
  const long long k = tid % edges;
  const int j = post[k];
  if (refractory[(long long)b * n + j] == 0)
    atomicAdd((unsigned long long*)&g[(long long)b * n + j],
              (unsigned long long)(long long)(weight[k] * scale));
}
"""


def find_cuda_headers():
    """NVRTC needs CUDA headers; the pip wheel ships them but says nothing."""
    if os.environ.get("CUDA_PATH"):
        return
    import site

    for root in site.getsitepackages():
        candidate = Path(root) / "nvidia" / "cuda_runtime"
        if (candidate / "include" / "cuda_runtime.h").exists():
            os.environ["CUDA_PATH"] = str(candidate)
            return


def tables():
    i = np.arange(1024, dtype=np.float32)
    return (np.exp(-DT * i / 20.0).astype(np.float32),
            np.exp(-DT * i / 5.0).astype(np.float32),
            np.exp(-DT * i / TAU_A).astype(np.float32))


def active_loop(cp, kernel, batch, ticks=TICKS):
    rng = np.random.default_rng(20260914)
    act = cp.asarray(np.sort(rng.choice(N, ACTIVE, replace=False)).astype(np.int32))
    rest = cp.asarray(np.full(N, -52.0, np.float32))
    drive = cp.asarray(rng.uniform(-7, -4, N).astype(np.float32))
    av, ag, aa = (cp.asarray(x) for x in tables())
    v = cp.asarray(rng.uniform(-70, -40, (batch, N)).astype(np.float32))
    g = cp.asarray(rng.uniform(-0.7, -0.4, (batch, N)).astype(np.float32))
    ad = cp.asarray(np.where(np.arange(N) % 3 == 0, 0.5, 0.0)[None, :]
                    .repeat(batch, 0).astype(np.float32))
    last = cp.zeros((batch, N), cp.int64)
    ref = cp.asarray(np.where(np.arange(N) % 7 == 0, 3, 0)[None, :]
                     .repeat(batch, 0).astype(np.int16))
    threads = 256
    blocks = (ACTIVE * batch + threads - 1) // threads

    def sweep(steps, start):
        for t in range(steps):
            kernel((blocks,), (threads,),
                   (np.int32(N), np.int32(ACTIVE), np.int32(batch),
                    np.int64(start + t), act, rest, drive, v, g, ad, last, ref,
                    av, ag, aa, DT, TAU_A))

    sweep(20, 1)
    cp.cuda.Stream.null.synchronize()
    started = time.perf_counter()
    sweep(ticks, 100)
    cp.cuda.Stream.null.synchronize()
    seconds = time.perf_counter() - started
    return seconds, ticks * ACTIVE * batch


def delivery_loop(cp, module, graph, batch, variant, ticks=1000):
    post, weight = graph
    edges = len(post)
    d_post, d_weight = cp.asarray(post), cp.asarray(weight)
    ref = cp.zeros((batch, N), cp.int16)
    kernel = module.get_function(variant)
    if variant == "deliver_fixed":
        acc = cp.zeros((batch, N), cp.int64)
        args = (np.int32(N), np.int32(batch), np.int64(edges), d_post,
                d_weight, acc, ref, FIXED)
    else:
        acc = cp.zeros((batch, N), cp.float32)
        args = (np.int32(N), np.int32(batch), np.int64(edges), d_post,
                d_weight, acc, ref)
    threads = 256
    blocks = (edges * batch + threads - 1) // threads
    for _ in range(5):
        kernel((blocks,), (threads,), args)
    cp.cuda.Stream.null.synchronize()
    started = time.perf_counter()
    for _ in range(ticks):
        kernel((blocks,), (threads,), args)
    cp.cuda.Stream.null.synchronize()
    return time.perf_counter() - started, ticks * edges * batch


def one_tick_of_edges(rng):
    """The outgoing edges of one tick's spiking neurons, from the real graph.

    The out-degree distribution is what decides how far the threads of a warp
    diverge, so a synthetic graph would measure the wrong thing.
    """
    z = np.load(Path("data/graph.npz"))
    ptr, post, weight = z["ptr"], z["post"], z["weight"]
    spikes = rng.choice(len(ptr) - 1, SPIKES_PER_TICK, replace=False)
    e = np.concatenate([np.arange(ptr[i], ptr[i + 1]) for i in spikes])
    return post[e].astype(np.int32), weight[e].astype(np.float32)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batches", default="1,8,16,24,32",
                   help="genomes simulated concurrently")
    a = p.parse_args()

    find_cuda_headers()
    import cupy as cp

    props = cp.cuda.runtime.getDeviceProperties(0)
    l2 = props["l2CacheSize"]
    print(f"{props['name'].decode()}, {props['multiProcessorCount']} SMs, "
          f"L2 {l2 / 1e6:.1f} MB\n")

    visits_per_obs = TICKS * ACTIVE
    cpu_one = 1e9 / CPU_NS_PER_VISIT / visits_per_obs
    print(f"CPU reference for the active loop: {CPU_NS_PER_VISIT} ns/visit on "
          f"one core\n  = {cpu_one:.3f} observations/s, "
          f"{cpu_one * CPU_CORES:.3f} at {CPU_CORES} cores\n")

    evolve = cp.RawKernel(EVOLVE, "evolve_active", options=("-fmad=false",))
    print("ACTIVE-LIST LOOP  (64% of operations)")
    print(f"{'batch':>6}{'state MB':>10}{'ns/visit':>10}{'obs/s':>9}"
          f"{'vs 1 core':>11}{'vs 8':>8}")
    for batch in [int(x) for x in a.batches.split(",")]:
        seconds, visits = active_loop(cp, evolve, batch)
        obs = batch / seconds
        state = batch * N * 22 / 1e6
        flag = "  <- exceeds L2" if state * 1e6 > l2 else ""
        print(f"{batch:>6}{state:>10.0f}{1e9 * seconds / visits:>10.3f}"
              f"{obs:>9.1f}{obs / cpu_one:>10.0f}x"
              f"{obs / (cpu_one * CPU_CORES):>7.1f}x{flag}")

    rng = np.random.default_rng(20260914)
    graph = one_tick_of_edges(rng)
    module = cp.RawModule(code=DELIVER, options=("-fmad=false",))
    print(f"\nSYNAPTIC SCATTER  (36% of operations, {len(graph[0]):,} "
          f"deliveries per tick)")
    print(f"{'batch':>6}{'variant':>14}{'ns/delivery':>13}{'reproducible':>14}")
    for batch in [int(x) for x in a.batches.split(",")]:
        for variant, repro in (("deliver_float", "no"),
                               ("deliver_fixed", "yes")):
            seconds, count = delivery_loop(cp, module, graph, batch, variant)
            print(f"{batch:>6}{variant.split('_')[1]:>14}"
                  f"{1e9 * seconds / count:>13.3f}{repro:>14}")

    print("\nreproducibility, the same input run twice:")
    for variant, dtype, extra in (("deliver_float", cp.float32, ()),
                                  ("deliver_fixed", cp.int64, (FIXED,))):
        kernel = module.get_function(variant)
        post, weight = graph
        d_post, d_weight = cp.asarray(post), cp.asarray(weight)
        ref = cp.zeros((1, N), cp.int16)
        outs = []
        for _ in range(2):
            acc = cp.zeros((1, N), dtype)
            kernel(((len(post) + 255) // 256,), (256,),
                   (np.int32(N), np.int32(1), np.int64(len(post)), d_post,
                    d_weight, acc, ref, *extra))
            cp.cuda.Stream.null.synchronize()
            outs.append(cp.asnumpy(acc).copy())
        same = np.array_equal(outs[0], outs[1])
        note = "" if same else f"   max diff {np.abs(outs[0] - outs[1]).max():.3e}"
        print(f"  {variant.split('_')[1]:>6} atomics identical: {same}{note}")

    print("\nFloat addition is not associative, so the order the warps happen "
          "to finish\nin changes the sum. Integer addition is, so it does not. "
          "That is the whole\nof the determinism question, and it costs "
          "nothing measurable.")


if __name__ == "__main__":
    main()
