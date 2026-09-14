// CUDA port of kernel.cpp. NOT FINISHED, and not wired into the model.
//
// Measured against the CPU kernel by tools/gpu_port_check.py from one shared
// state. At batch 1, ten of eighteen arrays come back bit-identical --
// counts, active, queue, queue_count, nactive, refractory, last, weight,
// active_flag, previous_drive -- which is the whole of the ordering logic, and
// the part that was actually at risk. Five differ: v, g, adaptation,
// modulation, eligibility, by at most 2e-3 and usually far less.
//
// That residue is not a race. It is `exp`: the host's libm and the device's
// do not round identically, and every one of those five arrays reaches a path
// that calls it rather than the shared lookup table. Bit-equality across the
// two machines is therefore not reachable while both call their own exp, and
// claiming it here would be a lie the tests would eventually catch.
//
// At batch greater than one, fourteen arrays differ. That IS a bug, in the
// per-genome indexing here or in the harness, and it is unfixed.
//
// The parallelism is across genomes, not inside one brain: one CUDA block runs
// one genome's whole 5,000-tick observation, and the batch fills the device.
// That is what makes exactness affordable. Every place kernel.cpp's result
// depends on the order operations happen in -- the active list, the spike
// queue, the accumulation into g -- keeps that order here:
//
//   * the active list is compacted with an ordered block scan, never an atomic
//     append, so survivors keep their relative positions;
//   * the spike queue is appended the same way, so deliveries happen in the
//     order the active list produced them;
//   * the queue is walked one spiking neuron at a time with a barrier between,
//     so additions to g from different sources cannot interleave;
//   * within one source, threads run its edges in parallel, which is safe only
//     because no neuron in this graph lists the same target twice. That was
//     measured, not assumed, and the host asserts it before using this kernel.
//
// Float addition is not associative, so any of those relaxed into an atomic
// would give a different answer every run and a different animal from the one
// docs/validation.md describes. The ordering machinery is not free: a block
// scan per chunk, and the queue walked one source at a time, is why a single
// block is slower than a single CPU core here and the gain has to come from
// running many genomes at once.
// NVRTC compiles without the host standard library, so no <cstdint> here;
// every integer width below is spelled out instead.

#define BLOCK 256
#define TABLE 1024

__device__ __forceinline__ float table(const float* t, long long d, float dt,
                                       float tau) {
  return d < TABLE ? t[d] : expf(-dt * (float)d / tau);
}

// Ordered exclusive scan over one block. Returns this thread's offset; `total`
// receives the block's sum. Deliberately a plain shared-memory scan: it is
// ordered, it is short, and it does not pull CUB through NVRTC.
__device__ __forceinline__ int block_scan(int value, int* shared, int* total) {
  const int tid = threadIdx.x;
  shared[tid] = value;
  __syncthreads();
  for (int offset = 1; offset < BLOCK; offset <<= 1) {
    int add = tid >= offset ? shared[tid - offset] : 0;
    __syncthreads();
    shared[tid] += add;
    __syncthreads();
  }
  const int inclusive = shared[tid];
  if (tid == BLOCK - 1) *total = inclusive;
  __syncthreads();
  return inclusive - value;
}

struct Brain {
  float* v;
  float* g;
  short* refractory;
  const float* drive;
  float* previous_drive;
  int* queue;
  int* queue_count;
  int* counts;
  int* active;
  unsigned char* flags;
  int* nactive;
  long long* last;
  double* eligibility;
  long long* eligibility_last;
  float* modulation;
  long long* modulation_last;
  float* adaptation;
  float* weight;
};

__device__ void evolve(const Brain& b, int i, long long now, float current,
                       const float* av, const float* ag, const float* aa,
                       float dt, float tau_a, const float* rest) {
  long long d = now - b.last[i];
  if (d <= 0) return;
  const int frozen = b.refractory[i] > 0 ? b.refractory[i] - 1 : 0;
  const int skip = (int)(d < frozen ? d : frozen);
  if (skip > 0 && b.adaptation[i] > 0.f)
    b.adaptation[i] *= table(aa, skip, dt, tau_a);
  b.refractory[i] =
      d >= b.refractory[i] ? 0 : (short)(b.refractory[i] - d);
  d -= skip;
  if (d > 0) {
    const float a = table(av, d, dt, 20.f);
    const float bb = table(ag, d, dt, 5.f);
    b.v[i] = rest[i] + (b.v[i] - rest[i]) * a + current * (1.f - a)
           + b.g[i] * (a - bb) / 3.f;
    b.g[i] *= bb;
    if (b.adaptation[i] > 0.f) {
      const float c = table(aa, d, dt, tau_a);
      b.v[i] -= b.adaptation[i] * tau_a / (tau_a - 20.f) * (c - a);
      b.adaptation[i] *= c;
    }
  }
  b.last[i] = now;
}

extern "C" __global__ void memory_advance_batch(
    const int n, const int batch, const long long* __restrict__ ptr,
    const int* __restrict__ post, float* weight_all,
    float* v_all, float* g_all, short* refractory_all,
    const float* __restrict__ drive_all, float* previous_drive_all,
    int* queue_all, int* queue_count_all, long long* clock_all, const int steps,
    const float dt, int* counts_all, int* active_all, unsigned char* flags_all,
    int* nactive_all, long long* last_all,
    const unsigned char* __restrict__ kc_mask,
    const signed char* __restrict__ dan_index,
    double* eligibility_all, long long* eligibility_last_all,
    const int nplastic, const long long* __restrict__ plastic_edge,
    const int* __restrict__ plastic_pre, const float* __restrict__ baseline_weight,
    const float* __restrict__ dan_gain, const float eta, const float tau_elig_ms,
    const float floor_fraction, const int learning_enabled,
    float* modulation_all, long long* modulation_last_all,
    const unsigned char* __restrict__ modulation_mask,
    const float* __restrict__ rest,
    float* adaptation_all, const float adaptation_jump, const float adaptation_tau,
    const float* __restrict__ av, const float* __restrict__ ag,
    const float* __restrict__ aa, const int delay, const int rfc,
    const int slots, int* scratch_all) {
  const int b = blockIdx.x;
  if (b >= batch) return;
  const long long off = (long long)b * n;
  const int tid = threadIdx.x;

  Brain brain;
  brain.v = v_all + off;
  brain.g = g_all + off;
  brain.refractory = refractory_all + off;
  brain.drive = drive_all + off;
  brain.previous_drive = previous_drive_all + off;
  brain.queue = queue_all + (long long)b * n * slots;
  brain.queue_count = queue_count_all + (long long)b * slots;
  brain.counts = counts_all + off;
  brain.active = active_all + off;
  brain.flags = flags_all + off;
  brain.nactive = nactive_all + b;
  brain.last = last_all + off;
  brain.eligibility = eligibility_all + off;
  brain.eligibility_last = eligibility_last_all + off;
  brain.modulation = modulation_all + off;
  brain.modulation_last = modulation_last_all + off;
  brain.adaptation = adaptation_all + off;
  brain.weight = weight_all + (long long)b * ptr[n];

  long long* clock = clock_all + b;
  (void)scratch_all;

  __shared__ int s_scan[BLOCK];
  __shared__ int s_total;
  __shared__ int s_kept;
  __shared__ int s_qcount;

  // ---- sensory currents, applied after settling the old ones ----
  if (tid == 0) s_kept = *brain.nactive;
  __syncthreads();
  for (int base = 0; base < n; base += BLOCK) {
    const int i = base + tid;
    const bool changed = i < n && brain.drive[i] != brain.previous_drive[i];
    if (changed) {
      evolve(brain, i, *clock - 1, brain.previous_drive[i], av, ag, aa, dt,
             adaptation_tau, rest);
      brain.previous_drive[i] = brain.drive[i];
    }
    const int want = (changed && !brain.flags[i]) ? 1 : 0;
    const int slot = block_scan(want, s_scan, &s_total);
    if (want) {
      brain.flags[i] = 1;
      brain.active[s_kept + slot] = i;
    }
    __syncthreads();
    if (tid == 0) s_kept += s_total;
    __syncthreads();
  }
  if (tid == 0) *brain.nactive = s_kept;
  __syncthreads();

  // ---- the tick loop ----
  for (int t = 0; t < steps; t++) {
    const long long now = *clock;
    const int slot = (int)(now % slots), future = (int)((now + delay) % slots);
    const int original = *brain.nactive;
    if (tid == 0) {
      s_kept = 0;
      s_qcount = brain.queue_count[future];
    }
    __syncthreads();

    for (int base = 0; base < original; base += BLOCK) {
      const int k = base + tid;
      const bool live = k < original;
      int i = -1, fires = 0, keeps = 0;
      if (live) {
        i = brain.active[k];
        evolve(brain, i, now, brain.drive[i], av, ag, aa, dt, adaptation_tau,
               rest);
        fires = (brain.refractory[i] == 0 && brain.v[i] > -45.f) ? 1 : 0;
        const float gap = -45.f - rest[i];
        keeps = (brain.v[i] > -45.f || brain.drive[i] > gap
                 || brain.drive[i] + brain.g[i] > gap) ? 1 : 0;
      }
      // Ordered append to the spike queue.
      int qslot = block_scan(fires, s_scan, &s_total);
      if (fires) {
        brain.queue[(long long)future * n + s_qcount + qslot] = i;
        brain.counts[i]++;
        if (kc_mask[i]) {
          brain.adaptation[i] += adaptation_jump;
          brain.eligibility[i] *= exp(
              -(double)dt * (double)(now - brain.eligibility_last[i])
              / (double)tau_elig_ms);
          brain.eligibility[i] += 1.0;
          brain.eligibility_last[i] = now;
        }
      }
      __syncthreads();
      if (tid == 0) s_qcount += s_total;
      __syncthreads();
      // Ordered compaction of the active list. Reads brain.active[k] before
      // any thread writes brain.active[s_kept + kslot]; the write index never
      // exceeds the read index, so the in-place compaction is safe.
      int kslot = block_scan(keeps, s_scan, &s_total);
      if (live && keeps) brain.active[s_kept + kslot] = i;
      if (live && !keeps) brain.flags[i] = 0;
      __syncthreads();
      if (tid == 0) s_kept += s_total;
      __syncthreads();
    }
    if (tid == 0) {
      *brain.nactive = s_kept;
      brain.queue_count[future] = s_qcount;
    }
    __syncthreads();

    // ---- deliveries, one source at a time so the order of additions to g
    // matches the sequential kernel exactly ----
    const int pending = brain.queue_count[slot];
    for (int q = 0; q < pending; q++) {
      const int i = brain.queue[(long long)slot * n + q];
      const long long first = ptr[i], stop = ptr[i + 1];
      if (modulation_mask[i]) {
        for (long long e = first + tid; e < stop; e += BLOCK) {
          const int j = post[e];
          brain.modulation[j] *= expf(
              -dt * (float)(now - brain.modulation_last[j]) / 100.f);
          brain.modulation[j] += fabsf(brain.weight[e]) / .275f;
          brain.modulation_last[j] = now;
        }
        __syncthreads();
        if (learning_enabled && dan_index[i] >= 0) {
          for (int p = tid; p < nplastic; p += BLOCK) {
            const int pre = plastic_pre[p];
            const double trace = brain.eligibility[pre] * exp(
                -(double)dt * (double)(now - brain.eligibility_last[pre])
                / (double)tau_elig_ms);
            const float gain = dan_gain[dan_index[i] * nplastic + p];
            const long long edge = plastic_edge[p];
            const float candidate =
                brain.weight[edge] * expf(-eta * gain * (float)trace);
            const float lower = baseline_weight[p] * floor_fraction;
            brain.weight[edge] = candidate > lower ? candidate : lower;
          }
          __syncthreads();
        }
        continue;
      }
      // No neuron in this graph lists the same target twice, so the threads
      // of this loop touch distinct j and need no atomics. Asserted on the
      // host before the kernel is used.
      for (long long base = first; base < stop; base += BLOCK) {
        const long long e = base + tid;
        int j = -1, want = 0;
        if (e < stop) {
          j = post[e];
          evolve(brain, j, now, brain.drive[j], av, ag, aa, dt, adaptation_tau,
                 rest);
          if (brain.refractory[j] == 0) {
            brain.g[j] += brain.weight[e];
            want = brain.flags[j] ? 0 : 1;
          }
        }
        const int aslot = block_scan(want, s_scan, &s_total);
        if (want) {
          brain.flags[j] = 1;
          brain.active[*brain.nactive + aslot] = j;
        }
        __syncthreads();
        if (tid == 0) *brain.nactive += s_total;
        __syncthreads();
      }
    }
    __syncthreads();
    if (tid == 0) brain.queue_count[slot] = 0;
    __syncthreads();

    const int reset = brain.queue_count[future];
    for (int q = tid; q < reset; q += BLOCK) {
      const int i = brain.queue[(long long)future * n + q];
      brain.v[i] = rest[i];
      brain.g[i] = 0.f;
      brain.refractory[i] = (short)rfc;
    }
    __syncthreads();
    if (tid == 0) (*clock)++;
    __syncthreads();
  }

  // ---- materialise every cell at the observation boundary ----
  for (int i = tid; i < n; i += BLOCK)
    evolve(brain, i, *clock - 1, brain.drive[i], av, ag, aa, dt, adaptation_tau,
           rest);
}
