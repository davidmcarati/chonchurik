// CUDA port of kernel.cpp. Bit-identical to it, and not yet wired in.
//
// Measured against the CPU kernel by tools/gpu_port_check.py from one shared
// state: after 5,000 ticks -- a whole observation -- every one of the eighteen
// arrays the kernel may touch comes back identical, element for element.
//
// The parallelism is across genomes, not inside one brain: one CUDA block runs
// one genome's whole observation, and the batch fills the device. That is what
// makes exactness affordable. Every place kernel.cpp's result depends on the
// order operations happen in -- the active list, the spike queue, the
// accumulation into g -- keeps that order here:
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
// docs/validation.md describes.
//
// WHAT COSTS WHAT, measured rather than guessed. This kernel is bound by the
// number of scattered memory accesses it makes, and by nothing else. The
// first version of it spent its whole budget on the wrong thing:
//
//   * past 32 blocks, more blocks bought nothing -- 7 ms a genome per 200
//     ticks at 32, at 64 and at 96, on a card with 84 SMs. Not short of
//     parallelism.
//   * 512 threads a block was exactly as fast as 256. Not short of warps to
//     hide latency with.
//   * the card serves 14.8 billion random four-byte lookups a second,
//     measured; that kernel was making 12.0 billion of them. 81% of the
//     ceiling, and there is no more to have.
//
// So the whole game is accesses per neuron touched. A visit used to read seven
// arrays and write five, and each of those is a separate 32-byte sector
// fetched to use four bytes of it. They are one struct now, sized and aligned
// to exactly one sector, so a visit costs one fetch instead of twelve, and the
// block scan went from sixteen barriers to two. Nothing about the arithmetic
// changed; only where the bytes live.
//
// That is 2.47 ms a genome per 200 ticks, 2.8x the old kernel, and 26.6x one
// CPU core at batch 84. The ceiling moved but did not go away: throughput is
// flat from 84 blocks to 420 -- one resident block per SM to the three the
// register count allows, and the two beyond that queueing -- so the card is
// again saturated on memory and not on anything this kernel can schedule
// differently. Getting past it needs a smaller cell or a sorted active list,
// and a sorted active list is a different animal, not a faster one.
//
// ABOUT exp. The host and this device do not agree on it, and the three
// places it appears need three different answers. Measured, not assumed:
//
//   * `exp(double)` here differs from MSVC's by one unit in the last place on
//     up to 6.2% of arguments, so anything the host keeps at double precision
//     must come from a table the host's own compiler filled. The eligibility
//     trace is a double all the way through, and a device-built table for it
//     disagreed on 1,048,575 entries out of 1,048,576.
//   * anything the host rounds to float absorbs that difference: over
//     4,194,304 draws spanning every scale the plasticity rule reaches,
//     `weight * exp(arg)` narrowed to float disagreed zero times. So the one
//     exponential with a free argument -- the rule's own -- is computed here
//     rather than tabulated, and that is why it is sound.
//   * `expf` is not any of this. It is a different function with a wider
//     error, and spelling the rule with it is what made all 176 weights wrong.
//
// NVRTC compiles without the host standard library, so no <cstdint> here;
// every integer width below is spelled out instead.

#ifndef BLOCK
#define BLOCK 256
#endif
#define WARPS (BLOCK / 32)
// kernel.cpp keeps 1024 entries and calls std::exp past them. The device
// cannot reproduce that call, so the table is extended instead -- built by the
// host's own compiler, it holds exactly what std::exp would have returned, and
// the fallback below stops being reachable for any tick count a run uses.
#define TABLE (1 << 20)

// Everything a visit to a cell reads or writes, in one 32-byte line. The field
// order is the packing, not a preference: eight bytes of `last` first so the
// struct's alignment is natural, then the floats, then the small ones in the
// tail. `drive` never changes during a launch but lives here anyway, because a
// visit needs it and a separate array would cost the sector this whole
// exercise is about saving.
struct __align__(32) Cell {
  long long last;
  float v, g, adaptation, drive;
  int counts;
  short refractory;
  unsigned char flags, pad;
};

// The slow half: eligibility and modulation, touched only when a Kenyon cell
// fires or a modulatory neuron delivers. Four arrays, one sector.
struct __align__(32) Slow {
  double eligibility;
  long long eligibility_last, modulation_last;
  float modulation, pad;
};

// What the graph fixes and a genome cannot change. Shared by every block, so
// all 166,700 of them are 1.3 MB and stay in L2 for the whole launch however
// many genomes are running.
struct __align__(8) Fixed {
  float rest;
  unsigned char kc, modulatory, pad;
  signed char dan;
};

// `kc_rest` is one of the evolved parameters, and it moves the resting
// potential of every Kenyon cell and of nothing else -- `rest` is a constant
// everywhere else in the graph. So a genome changes one number rather than an
// array, and Fixed stays shared across the whole batch.
__device__ __forceinline__ float rest_of(const Fixed& f, float kc_rest) {
  return f.kc ? kc_rest : f.rest;
}

// This device flushes float subnormals to zero in every arithmetic path, and
// --ftz=false does not change it: measured, a subnormal survives a load and
// dies in a multiply. It also lies about them -- `x == 0.f` is true for a
// subnormal, and `(double)x` is -0.0 -- so both the guard and the widening
// have to go through the bits.
//
// The host keeps them, so the kernel has to. A subnormal float is an exact
// integer multiple of 2^-149, the product of two floats is exact in double,
// and the integer is therefore exact: decode the operands bitwise, multiply in
// double, and reassemble. Verified against the host on 20,007 values including
// every magnitude of subnormal, both zeros and the normal/subnormal boundary.
//
// The fast path is one comparison. The slow path runs only when the hardware
// has already returned zero for a product that is not zero.
__device__ __forceinline__ bool is_zero(float x) {
  return (__float_as_uint(x) & 0x7FFFFFFFu) == 0u;
}

__device__ __forceinline__ double as_double(float x) {
  const unsigned bits = __float_as_uint(x);
  if ((bits >> 23) & 0xFFu) return (double)x;
  const double magnitude = (double)(bits & 0x7FFFFFu) * 1.401298464324817e-45;
  return (bits & 0x80000000u) ? -magnitude : magnitude;
}

__device__ __forceinline__ float rebuild(double exact) {
  const double m = rint(fabs(exact) * 7.1362384635298e+44);  // 2^149
  // Above 2^23 multiples of 2^-149 the answer is a normal float, and the
  // ordinary conversion is correct for those. It is reachable here because the
  // device also flushes subnormal *inputs*: an operand read as zero gives a
  // zero product whose true value is perfectly normal.
  if (m >= 8388608.0) return (float)exact;
  return __uint_as_float((exact < 0.0 ? 0x80000000u : 0u) | (unsigned)m);
}

__device__ __forceinline__ float mul_rn(float x, float y) {
  const float p = __fmul_rn(x, y);
  if (!is_zero(p) || is_zero(x) || is_zero(y)) return p;
  return rebuild(as_double(x) * as_double(y));
}

// Addition, subtraction and division need the same treatment for the same
// reason. The reconstruction is exact: for a sum of two floats to land in the
// subnormal range both must be at most 2^-103, since otherwise their spacing
// alone exceeds 2^-126 -- so double holds them and their sum without loss.
__device__ __forceinline__ float add_rn(float x, float y) {
  const float s = __fadd_rn(x, y);
  if (!is_zero(s) || (is_zero(x) && is_zero(y))) return s;
  return rebuild(as_double(x) + as_double(y));
}

__device__ __forceinline__ float sub_rn(float x, float y) {
  const float s = __fsub_rn(x, y);
  if (!is_zero(s) || (is_zero(x) && is_zero(y))) return s;
  return rebuild(as_double(x) - as_double(y));
}

__device__ __forceinline__ float div_rn(float x, float y) {
  const float q = __fdiv_rn(x, y);
  if (!is_zero(q) || is_zero(x)) return q;
  return rebuild(as_double(x) / as_double(y));
}

__device__ __forceinline__ float table(const float* t, long long d, float dt,
                                       float tau) {
  return d < TABLE ? t[d] : expf(-dt * (float)d / tau);
}

// The same decay, in double, for the two places the host keeps double.
//
// `std::exp` takes a double. The voltage decays multiply it into a float and
// round once, so a float table reproduces them exactly; eligibility is a
// double and keeps every bit, and modulation is a float multiplied by the
// double result and rounded once -- a float table rounds twice for both, and
// the second rounding is a part in 16 million that never goes away.
//
// The argument is still spelled as the host spells it: dt times a tick count
// in float, divided in float, and only then widened.
__device__ __forceinline__ double table_d(const double* t, long long d,
                                          float dt, float tau) {
  return d < TABLE ? t[d] : exp((double)(-dt * (float)d / tau));
}

// Ordered exclusive scan over one block. Returns this thread's offset; `total`
// receives the block's sum. The values are zeros and ones, so these additions
// are exact and their order is free -- unlike everything downstream of them,
// which is why the scan may be fast and the deliveries may not.
//
// Warp shuffles first, then one pass over the per-warp totals. Two barriers
// for the whole scan, where the shared-memory ladder it replaced needed
// sixteen, and a tick runs several thousand scans.
__device__ __forceinline__ int block_scan(int value, int* warp_sum, int* total) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  int x = value;
  for (int offset = 1; offset < 32; offset <<= 1) {
    const int up = __shfl_up_sync(0xFFFFFFFFu, x, offset);
    if (lane >= offset) x += up;
  }
  if (lane == 31) warp_sum[warp] = x;
  __syncthreads();
  int base = 0, sum = 0;
  // Every thread walks the same handful of warp totals out of shared memory.
  // Cheaper than a second scan, and it leaves the block's sum in a register
  // instead of costing another barrier to broadcast it.
  for (int w = 0; w < WARPS; w++) {
    const int s = warp_sum[w];
    if (w < warp) base += s;
    sum += s;
  }
  *total = sum;
  __syncthreads();
  return base + x - value;
}

struct Brain {
  Cell* cell;
  Slow* slow;
  float* previous_drive;
  int* queue;
  int* queue_count;
  int* active;
  int* nactive;
  float* weight;
};

// Advance one cell to `now`, in registers. The caller owns the load and the
// store, because every caller needs the cell for something else immediately
// afterwards and a second trip to memory is the entire cost of this kernel.
__device__ __forceinline__ void evolve(Cell& c, float rest, long long now,
                                       float current, const float* av,
                                       const float* ag, const float* aa,
                                       float dt, float tau_a) {
  long long d = now - c.last;
  if (d <= 0) return;
  const int frozen = c.refractory > 0 ? c.refractory - 1 : 0;
  const int skip = (int)(d < frozen ? d : frozen);
  if (skip > 0 && c.adaptation > 0.f)
    c.adaptation = mul_rn(c.adaptation, table(aa, skip, dt, tau_a));
  c.refractory = d >= c.refractory ? 0 : (short)(c.refractory - d);
  d -= skip;
  if (d > 0) {
    const float a = table(av, d, dt, 20.f);
    const float bb = table(ag, d, dt, 5.f);
    // Spelled with round-to-nearest intrinsics, not operators. The same
    // expression written plainly compiles to rest + ((t1 + t2) + t3) here
    // while the host computes ((rest + t1) + t2) + t3, and the two differ by
    // half a unit in the last place -- which is a spike, for a cell sitting on
    // the threshold. --fmad=false does not prevent the reassociation; these
    // do, because each is one IEEE operation the compiler cannot fold.
    const float t1 = mul_rn(sub_rn(c.v, rest), a);
    const float t2 = mul_rn(current, sub_rn(1.f, a));
    const float t3 = div_rn(mul_rn(c.g, sub_rn(a, bb)), 3.f);
    c.v = add_rn(add_rn(add_rn(rest, t1), t2), t3);
    c.g = mul_rn(c.g, bb);
    if (c.adaptation > 0.f) {
      const float cc = table(aa, d, dt, tau_a);
      const float shed = mul_rn(
          div_rn(mul_rn(c.adaptation, tau_a), sub_rn(tau_a, 20.f)),
          sub_rn(cc, a));
      c.v = sub_rn(c.v, shed);
      c.adaptation = mul_rn(c.adaptation, cc);
    }
  }
  c.last = now;
}

extern "C" __global__ void memory_advance_batch(
    const int n, const int batch, const long long* __restrict__ ptr,
    const int* __restrict__ post, float* weight_all,
    const long long weight_stride,
    Cell* cell_all, Slow* slow_all, const Fixed* __restrict__ fixed,
    float* previous_drive_all,
    int* queue_all, int* queue_count_all, long long* clock_all, const int steps,
    const float dt, int* active_all, int* nactive_all,
    const int nplastic, const long long* __restrict__ plastic_edge,
    const int* __restrict__ plastic_pre,
    const float* __restrict__ baseline_weight,
    const float* __restrict__ dan_gain,
    // One entry per genome: these four are evolved, and a batch that shared
    // one physiology would not be an evolution.
    const float* __restrict__ eta_all,
    const float* __restrict__ adaptation_jump_all,
    const float* __restrict__ adaptation_tau_all,
    const float* __restrict__ kc_rest_all,
    const float tau_elig_ms, const float floor_fraction,
    const int learning_enabled,
    const float* __restrict__ av, const float* __restrict__ ag,
    // `aa` tabulates exp(-dt*i/adaptation_tau), so it is one whole table per
    // genome, TABLE entries apart. The other four do not depend on a genome.
    const float* __restrict__ aa_all, const double* __restrict__ am,
    const double* __restrict__ ae, const int delay, const int rfc,
    const int slots, int* scratch_all) {
  const int b = blockIdx.x;
  if (b >= batch) return;
  const long long off = (long long)b * n;
  const int tid = threadIdx.x;

  Brain brain;
  brain.cell = cell_all + off;
  brain.slow = slow_all + off;
  brain.previous_drive = previous_drive_all + off;
  brain.queue = queue_all + (long long)b * n * slots;
  brain.queue_count = queue_count_all + (long long)b * slots;
  brain.active = active_all + off;
  brain.nactive = nactive_all + b;
  // A stride of zero points every genome at one shared copy of the
  // weights. Sound only while nothing writes to them -- the plasticity
  // rule does -- so the host passes the real stride whenever learning is
  // enabled, and 102 MB a genome is what that costs.
  brain.weight = weight_all + (long long)b * weight_stride;

  const float eta = eta_all[b];
  const float adaptation_jump = adaptation_jump_all[b];
  const float adaptation_tau = adaptation_tau_all[b];
  const float kc_rest = kc_rest_all[b];
  const float* aa = aa_all + (long long)b * TABLE;

  long long* clock = clock_all + b;
  (void)scratch_all;

  __shared__ int s_warp[WARPS];
  __shared__ int s_kept;
  __shared__ int s_qcount;
  int total;

  // ---- sensory currents, applied after settling the old ones ----
  if (tid == 0) s_kept = *brain.nactive;
  __syncthreads();
  for (int base = 0; base < n; base += BLOCK) {
    const int i = base + tid;
    bool changed = false;
    Cell c;
    c.flags = 1;
    if (i < n) {
      c = brain.cell[i];
      changed = c.drive != brain.previous_drive[i];
      if (changed) {
        evolve(c, rest_of(fixed[i], kc_rest), *clock - 1,
               brain.previous_drive[i], av, ag, aa, dt, adaptation_tau);
        brain.previous_drive[i] = c.drive;
      }
    }
    const int want = (changed && !c.flags) ? 1 : 0;
    const int slot = block_scan(want, s_warp, &total);
    if (want) {
      c.flags = 1;
      brain.active[s_kept + slot] = i;
    }
    if (changed) brain.cell[i] = c;
    __syncthreads();
    if (tid == 0) s_kept += total;
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
      Cell c;
      unsigned char kc = 0;
      if (live) {
        i = brain.active[k];
        const Fixed f = fixed[i];
        kc = f.kc;
        const float rest = rest_of(f, kc_rest);
        c = brain.cell[i];
        evolve(c, rest, now, c.drive, av, ag, aa, dt, adaptation_tau);
        fires = (c.refractory == 0 && c.v > -45.f) ? 1 : 0;
        const float gap = -45.f - rest;
        keeps = (c.v > -45.f || c.drive > gap || c.drive + c.g > gap) ? 1 : 0;
      }
      // Ordered append to the spike queue.
      const int qslot = block_scan(fires, s_warp, &total);
      if (fires) {
        brain.queue[(long long)future * n + s_qcount + qslot] = i;
        c.counts++;
        if (kc) {
          c.adaptation += adaptation_jump;
          Slow s = brain.slow[i];
          s.eligibility *=
              table_d(ae, now - s.eligibility_last, dt, tau_elig_ms);
          s.eligibility += 1.0;
          s.eligibility_last = now;
          brain.slow[i] = s;
        }
      }
      __syncthreads();
      if (tid == 0) s_qcount += total;
      __syncthreads();
      // Ordered compaction of the active list. Reads brain.active[k] before
      // any thread writes brain.active[s_kept + kslot]; the write index never
      // exceeds the read index, so the in-place compaction is safe.
      const int kslot = block_scan(keeps, s_warp, &total);
      if (live && keeps) brain.active[s_kept + kslot] = i;
      if (live) {
        if (!keeps) c.flags = 0;
        brain.cell[i] = c;
      }
      __syncthreads();
      if (tid == 0) s_kept += total;
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
      if (fixed[i].modulatory) {
        for (long long e = first + tid; e < stop; e += BLOCK) {
          const int j = post[e];
          Slow s = brain.slow[j];
          // float *= double on the host: widen, multiply, round once.
          s.modulation = (float)((double)s.modulation *
              table_d(am, now - s.modulation_last, dt, 100.f));
          s.modulation = add_rn(s.modulation,
                                div_rn(fabsf(brain.weight[e]), .275f));
          s.modulation_last = now;
          brain.slow[j] = s;
        }
        __syncthreads();
        const int dan = fixed[i].dan;
        if (learning_enabled && dan >= 0) {
          for (int p = tid; p < nplastic; p += BLOCK) {
            const int pre = plastic_pre[p];
            const Slow s = brain.slow[pre];
            const double trace = s.eligibility *
                table_d(ae, now - s.eligibility_last, dt, tau_elig_ms);
            const float gain = dan_gain[dan * nplastic + p];
            const long long edge = plastic_edge[p];
            // The host writes `weight[edge]*std::exp(-eta*gain*trace)`, and
            // every promotion in that line matters. `-eta*gain` is two floats
            // and stays float; `trace` is a double, so the argument, the
            // exponential and the product are double, and the single rounding
            // to float happens at the assignment. Spelling it `expf` of a
            // float, which is what this was, rounds three times too early and
            // moved all 176 weights the rule touched.
            const float ea = __fmul_rn(-eta, gain);
            const float candidate = (float)((double)brain.weight[edge] *
                                            exp((double)ea * trace));
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
        Cell c;
        const bool live = base + tid < stop;
        if (live) {
          j = post[e];
          c = brain.cell[j];
          evolve(c, rest_of(fixed[j], kc_rest), now, c.drive, av, ag, aa,
                 dt, adaptation_tau);
          if (c.refractory == 0) {
            c.g = add_rn(c.g, brain.weight[e]);
            want = c.flags ? 0 : 1;
          }
        }
        const int aslot = block_scan(want, s_warp, &total);
        if (want) {
          c.flags = 1;
          brain.active[*brain.nactive + aslot] = j;
        }
        if (live) brain.cell[j] = c;
        __syncthreads();
        if (tid == 0) *brain.nactive += total;
        __syncthreads();
      }
    }
    __syncthreads();
    if (tid == 0) brain.queue_count[slot] = 0;
    __syncthreads();

    const int reset = brain.queue_count[future];
    for (int q = tid; q < reset; q += BLOCK) {
      const int i = brain.queue[(long long)future * n + q];
      Cell c = brain.cell[i];
      c.v = rest_of(fixed[i], kc_rest);
      c.g = 0.f;
      c.refractory = (short)rfc;
      brain.cell[i] = c;
    }
    __syncthreads();
    if (tid == 0) (*clock)++;
    __syncthreads();
  }

  // ---- materialise every cell at the observation boundary ----
  for (int i = tid; i < n; i += BLOCK) {
    Cell c = brain.cell[i];
    evolve(c, rest_of(fixed[i], kc_rest), *clock - 1, c.drive, av, ag, aa,
           dt, adaptation_tau);
    brain.cell[i] = c;
  }
}
