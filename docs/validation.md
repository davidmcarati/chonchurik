# Validation status

Recorded during implementation on 2026-09-09. All exchange-order tests use an in-memory SDK double; **no real orders or funded-account checks were performed**.

Final local result: **42 tests passed**, including the opt-in full-connectome test. An offline fixture run also resumed from its saved ledger/checkpoint with balances and neural time preserved. Four upstream AgentKit/Pydantic deprecation warnings remain; they did not fail the tests.

| Check | Observed result | What it does not establish |
| --- | --- | --- |
| Rebuild from checksum-verified released files | 166,700 neurons, 25,582,938 connections and 124,177,617 contacts; every compiled array matches its lock | Completeness or physiological accuracy of the reconstruction |
| Execution/unit tests | Budget, stale data, spread, inventory, permissions, preview fees, STOP, cooldown, duplicate settlement, unknown submission and read-only public-client isolation pass | Successful execution against a particular live account, region or fee tier |
| Full-network sensory/feedback test | White RGB input activates KCs; reward and aversive pulses cause spikes in identified DAN cells; eligible synapses change | Accurate fly retinal responses or an acquired trading association |
| Same-checkpoint controls | Reward changes weights differently from no external reward; frozen memory stays exactly unchanged; checkpoint restoration works | Improved decisions, retention of a useful strategy, or conditioning equivalent to a biological experiment |
| Six accelerated observations of actual Coinbase public BTC-USDC data | One neural BUY filled in paper mode; five subsequent BUY proposals vetoed by cooldown | A realistic trading return or unbiased strategy |

In that six-observation run, the first paper fill used 9.754680320 USDC plus 0.058528081920 USDC in modeled fees. The next observation scheduled a **200 ms aversive pulse**, with **55 spikes across the two PPL101 cells**. KCs produced 11–16 spikes per observation; five candidate edges differed from baseline. Those five edges had already changed before the external loss pulse: endogenous dopamine activity can also drive the rule. Do not attribute every weight change to P&L.

The repeated BUY proposals are an important limitation. This run demonstrated neither successful learning nor a profitable policy. White-field neural tests activate substantially more KCs than the actual chart, so visual representation remains a major open modeling question. The cause of the one-sidedness was later measured directly; see below.

## Diagnostics, 2026-09-14

Read-only measurements on the full retained graph, reproducible with `tools/diagnose.py`. These replace the earlier guess that "a fixed directional decoder can turn circuit bias into one-sided exposure" — measurement does not support the circuit-bias half of that sentence.

### Graph structure: what is connected and what is idle

Measured by breadth-first search over the retained graph.

| Measurement | Result |
| --- | --- |
| Brain upstream of the two decoder cells, within 3 synapses | **96.8%** (99.3% within 5) |
| MBON07/11 → DNpe017, the gate | hop 2 |
| MBON07/11 → DNp20, the direction cells | hop 3 |
| Motor neurons reachable from descending neurons | 815 / 815, within 2 hops |

The brain is already connected to the decision — connectivity is not the missing piece. Learning also reaches the gate one synapse earlier than it reaches direction, so it influences *whether* to act more directly than *which way*.

Populations present in the graph and currently unused by the runtime:

| Population | Count | Currently |
| --- | --- | --- |
| visual-related (`ol_intrinsic`, `visual_projection`, `ol_sensory`, …) | 105,267 (**63.1% of the graph**) | fed a static chart |
| descending neurons | 1,314 | 2 of them are read |
| motor neurons (`vnc_motor` + `cb_motor`) | 815 | unused |
| ascending neurons | 1,846 | unused |
| olfactory receptor neurons | 2,635 | unused |
| LB3c sugar gustatory | 23 | compiled into `graph.npz`, never driven |

`prepare.py` already locates `DNa02`, `DNp09`, `MDN` and `MN9` and writes them to `manifest.json`; nothing reads that file.

### The laterality is in the reconstruction, not in this code

| Population | Left | Right | Ratio |
| --- | --- | --- | --- |
| R1–R6 photoreceptors | 1,112 | 2,265 | **2.04** |
| mapped retina (driven) | 1,107 | 2,228 | **2.01** |
| summed retinal drive | 27,836 | 59,238 | **2.13** |
| direct input onto DNp20 | 474.1 net | 479.3 net | 1.01 |

MaleCNS v1.0 contains about twice as many right R1–R6 cells as left. Every downstream structure measured — lamina, `ol_intrinsic`, `visual_projection`, descending neurons, Kenyon cells — is symmetric to within 1%, and the two DNp20 cells receive near-identical net drive. So the asymmetry enters at the photoreceptors and is a property of the released reconstruction.

### No input manipulation removes the one-sidedness

Six fixture frames per arm, frozen weights, same reset state before each observation:

| Arm | Drive R/L | Mean R−L | BUY / SELL / HOLD |
| --- | --- | --- | --- |
| baseline | 2.124 | +4.67 Hz | 4 / **0** / 2 |
| horizontally mirrored chart | 2.140 | +5.33 Hz | 5 / **0** / 1 |
| header bar repainted | 2.000 | +6.00 Hz | 5 / **0** / 1 |
| per-side drive equalised | — | +3.33 Hz | 4 / **0** / 2 |

Not one arm produced a single SELL. Mirroring the chart makes the bias slightly *worse*, so chart content is not the cause. Repainting the dark header removes most of the per-cell luminance gap (+1.4423 → −0.1406) and pulls the drive ratio to 2.000, yet the proposals do not move. Equalising summed per-side drive through an additive tonic current also fails to balance the mix.

### Why: the readout is a knife-edge on an unstable quantity

Random neurons were silenced with a large hyperpolarising tonic current — the analogue of Kir2.1/GtACR. No node or edge is removed; `ptr`, `post` and `weight` are asserted byte-identical afterwards. Two controls passed: silencing only cells that never fired changed the decision by exactly nothing, and the graph arrays were unchanged.

| Silenced | Decisions flipped | Mean R−L | **SD of R−L** |
| --- | --- | --- | --- |
| 1% | 0 / 5 | +6.40 Hz | **2.94** |
| 5% | 1 / 5 | +4.00 Hz | **9.55** |
| 20% | 2 / 5 | +1.60 Hz | **27.17** |
| 50% | 1 / 5 | +24.00 Hz | **13.74** |

The decision threshold is 2 Hz. Silencing just 1% of the network already moves `difference_hz` by a standard deviation of ~3 Hz, and under heavier ablation the spread reaches roughly 13× the threshold. The label mostly stays BUY not because the decision is robust, but because a standing positive offset is larger than the signal the threshold is trying to resolve.

Counting label flips understates this badly. The honest statement is that the decoder resolves a 2-spike margin out of roughly 35 DNp20 spikes, on a quantity whose noise exceeds that margin. Widening the readout to more cells is therefore not sufficient on its own — it needs to **integrate over the observation window** rather than threshold a mean-rate difference.

### Can the fly tell one market state from another?

Eight canonical market shapes, one observation each from an identical reset state, scored on how far apart they are at the retina and in the mushroom body.

Measured at the reconstructed excitation/inhibition ratio of 1.0, which is
what motivated the filled chart. Read the note below the table before reusing
these numbers.

| Rendering | Receptors differing | Input correlation | Identical Kenyon codes | Kenyon activity range |
| --- | --- | --- | --- | --- |
| current (3px polyline) | 4.19% | 0.906 | **10 / 28 pairs** | 0.10% – 34.6% |
| filled area under curve | 14.87% | 0.752 | **0 / 28** | 0.05% – 35.4% |
| filled + fixed price axis | 17.25% | 0.760 | 6 / 28 | 0.10% – 35.4% |

**A rally and a crash reach the retina as 98.5% the same image** (1.47% of receptors differ, correlation 0.959). A 3-pixel line on a 320×180 chart puts the market on a few percent of the receptors; reversing its direction changes the light at roughly fifty of 3,335 cells.

The consequence is that **10 of 28 state pairs produce a byte-identical Kenyon-cell code** — a crash and a choppy market are literally indistinguishable in memory. No reinforcement rule can bind an outcome to a situation that has no distinct representation. The learning machinery measured above is working, but it is working on nothing.

Filling the area under the curve removes every collision (10 → 0) and separates the input 3.5×. Adding a fixed price axis separates the input further still (4.1×) but reintroduces six collisions, so raw input separation is the wrong thing to rank on.

**That benefit does not survive the inhibitory gain, and the filled chart is
kept anyway.** Re-run at the shipped ratio of 1.9, every rendering reaches the
mushroom body with at most 0.07% of Kenyon cells — two of them — and the
comparison stops measuring the picture. The rendering still separates the
retinal input four-fold (4.19% of receptors against 17.73%), and nothing
downstream of the retina is currently using that. It ships because it is a
better encoding at no cost, not because it is doing work.

**Sparseness is not fixed by any of this.** Every rendering leaves Kenyon activity bimodal — either near-silent (~0.1%) or about a third of the population — and never near the ~5% a real mushroom body holds. A sub-2% change in retinal input flips 80× swings in Kenyon spiking. The population has two attractors and the picture only chooses between them, so sparseness is set by network dynamics, not by the display. Fixing the input is necessary and not sufficient.

Both APL cells are present in the graph and correctly wired — 2,259 and 2,374 inhibitory outputs onto Kenyon cells (net −26,274 and −27,681), with 2,270 and 2,423 Kenyon inputs returning to them. The feedback loop that enforces sparseness in a real mushroom body exists here; it is not holding the population in range.

### Is there any parameter regime where the Kenyon code is sparse?

The separation measurement above left Kenyon activity bimodal in every chart
rendering. Sparseness is therefore set by the dynamics, not by the picture, so
each declared free parameter was swept to see whether the biological range —
a few percent of Kenyon cells per stimulus — is reachable at all. Frozen
weights, filled rendering, four market states per point, everything restored
and `ptr`/`post`/`weight` asserted byte-identical afterwards.

One lever at a time, median Kenyon activity across the four states:

| Lever | Range swept | Median Kenyon activity |
| --- | --- | --- |
| `kc_rest` | −75 to −50 mV | 0.05% – 18.5% |
| APL gain on 4,633 existing APL→KC edges | ×1 to ×16 | 0.10% – 33.0% |
| `adaptation_jump` | 0 to 40 mV | 0.10% – 33.1% |
| `lamina_bias` | 4 to 16 | 0.10% – 33.0% |

Then `kc_rest` crossed with APL gain, 30 points. **Not one holds all four
states inside a 1%–15% band.** Over all 196 observations of the sweep, 84%
leave Kenyon cells near-silent and 10% put more than a fifth of the population
in spikes; **6% land in between**. Whole-brain spiking moves with it, 385,000
spikes per observation when Kenyon cells are silent against 452,000 when they
are active, so this is a property of the network and not of one population.

The floor is literally two cells out of 4,064. The ceiling is a third of the
population. The parameters choose which of the two the network falls into;
they do not produce a graded response. **Sparseness here cannot be set by
gain**, and any improvement to the input only picks between the same two
attractors.

One finding came out of this sweep that was not what it was looking for. Every
manipulation of the *input* tested earlier produced zero SELL proposals in 24
observations. Moving these *physiology* parameters produced 23 SELLs in 196
observations, and 17 of the 30 grid points give more than one distinct
decision. That does not mean the fly changed its mind: the widest SELL margin
is 4 spikes past the threshold against 6 for BUY, on a quantity whose standard
deviation under 1% ablation is already 2.94 Hz. The readout came off its rail,
which is a different thing from deciding.

### Does the olfactory channel give the mushroom body what vision could not?

Olfactory receptor neurons reach 3,829 of the 4,064 Kenyon cells in two
synapses through the antennal lobe. Vision has no such route. The channel
described in [the model](model.md) was measured against the chart on the same
eight market states, with the chart held at the neutral scene for the
odour-only arm, so any difference is attributable to the antennal lobe.

The odour is distinct where the chart was not — 0 of 28 state pairs share a
glomerular code, against 10 of 28 sharing a Kenyon code under the old chart.
But the receptor population is itself a step: it emits no spike at all up to a
peak current of 5 and saturates the Kenyon population immediately above it.
Crossing peak current with APL gain over 25 points found one sparse regime,
APL ×2, holding Kenyon activity at 7.5–8.7% for **all eight** market states —
the first time any configuration in this experiment reached the biological
range. The states still shared 77% of their active Kenyon cells.

**Two explanations for that shared code were tested and both failed.**

| Hypothesis | Test | Result |
| --- | --- | --- |
| The same intrinsically loudest cells always win | Equalise summed excitatory input onto every Kenyon cell (spread 26.4 to 509.9, a factor of 19) | overlap 0.76 → **0.76** |
| The states share too many glomeruli to begin with | Sharpen the tuning curve, 17.2 glomeruli per state down to 4.9 | input overlap 0.44 → 0.19, Kenyon overlap 0.76 → **0.79** |

Halving the input overlap moved the Kenyon overlap by −0.03. The shared code
is made in the mushroom body, not inherited from the receptors.

### Where the market signal is actually lost

Walking out from the receptors one synapse at a time, overlap between market
states:

| Layer | Cells | Median active | Overlap | Rate correlation |
| --- | --- | --- | --- | --- |
| glomerular code (before any spike) | 53 channels | — | **0.44** | — |
| olfactory receptors | 2,635 | 43.7% | 0.61 | 0.346 |
| **antennal lobe (1 synapse)** | 1,151 | **94.0%** | **0.98** | 0.964 |
| 2 synapses | 43,438 | 16.6% | 0.87 | 0.984 |
| Kenyon cells | 4,064 | 8.9% | 0.76 | 0.933 |

**The first synapse destroys it.** 94% of the antennal lobe fires for every
market state; the overlap goes from 0.61 to 0.98 in one step and nothing
downstream recovers what is lost there.

### One ratio explains every saturation

Excitation and inhibition are both set from contact count × 0.275, with the
sign taken from the transmitter annotation. Real circuits are not balanced
that way. Scaling every one of the 9,813,608 inhibitory edges (38% of the
graph) — sign, wiring and relative magnitudes untouched:

| Inhibitory gain | Antennal lobe overlap | Kenyon overlap | Kenyon active | States in 1–15% band | Whole-brain spikes |
| --- | --- | --- | --- | --- | --- |
| ×1 (as reconstructed) | 0.98 | 0.88 | 41.3 – 46.6% | 0 / 8 | 667,430 |
| ×1.5 | 0.93 | 0.64 | 7.9 – 11.7% | 8 / 8 | 505,565 |
| ×1.7 | 0.92 | 0.57 | 3.7 – 5.3% | 8 / 8 | 466,920 |
| **×1.9** | 0.91 | **0.50** | **1.6 – 2.4%** | **8 / 8** | 442,477 |
| ×2 | 0.90 | 0.49 | 1.0 – 1.4% | 5 / 8 | 435,366 |
| ×3 | 0.84 | 0.32 | 0.05 – 0.17% | 0 / 8 | 398,305 |

×1.9 was selected by a rule declared before the sweep ran: sparse in every
market state first, lowest overlap second. It is a declared free parameter and
it ships. A real mushroom body holds an overlap nearer 0.1–0.3, so 0.50 is
still poor; what changed is that a graded regime exists at all.

**This has a cost, and the cost is informative.** At ×1.9 a white field no
longer reaches the mushroom body: 12 Kenyon spikes against 2,089 before. That
is not a regression. Measured across the whole gain range, the chart drives
either a third of the Kenyon population or exactly two cells of it and never
anything between — the visual pathway has no operating point at any ratio, and
what used to look like sensory drive was saturation. The earlier finding that
the chart produced 10 of 28 identical Kenyon codes says the same thing from
the other side.

One consequence is worth stating separately. With the network out of
saturation, an observation with no reinforcement now changes **zero** plastic
edges, where reward changes 144. The earlier measurement — 3,401 edges moved
with no external reward, 97% of the reinforced arms — was a property of the
saturated regime, not of the learning rule.

### Two calibrations that were never decisions

Both were found by measurement, not by intent.

The reward and aversive pulses are the same engineered amplitude into
identified cells, but PAM11 has 15 cells and PPL101 has 2. At the shipped
amplitude of 20 the reward compartment received **14.7× the per-cell drive**
the aversive one did — an accident of population size. 40 is the smallest
swept amplitude driving both within 2× (32.2 against 21.0 spikes per cell).

The satiety channel was written to deliver a current between zero and a peak.
A cell needs about 7 units of drive to reach threshold, so it reported gains
and nothing else: a 2% loss, a 5% loss and an untouched account all arrived as
silence. Delivered between 8 and 32 instead, it is monotone across the range —
264 spikes at −7% equity, 808 at rest, 1,275 at a gain.

### Would a wider readout help?

Three candidate readouts, each a whole anatomical class split by soma side, so
none is selected for how it behaves. Signal is the spread of the reading
across eight market states; noise is its standard deviation over 15 repeats
with 5% of the network silenced and the market held fixed. Both readings come
from the same observations, split into ten 50 ms bins.

| Readout | Cells (L/R) | Best signal-to-noise |
| --- | --- | --- |
| DNp20, as shipped | 1 / 1 | 0.43 |
| descending neurons | 656 / 648 | **0.66** |
| descending + motor | 1,065 / 1,054 | 0.64 |

An integrating statistic — whether the side difference held across the window,
rather than how large the summed difference was — is what the wide populations
score best on. **Every candidate is still below 1.** For all of them the
market moves the readout less than silencing unrelated neurons does. Widening
the readout improves it by about half and does not fix it, so **no decoder
change ships**; the bottleneck is still upstream.

A first pass at 5 repeats put `descending + motor` at 1.15 and would have
justified shipping it. The 15-repeat re-run put it at 0.64. The number that
mattered was noise in a five-sample estimate of noise.

### Participation and plasticity

Baseline over six frames: **11.2%** of neurons fire at least once; participation ratio (effective contributing population) **8,832** of 166,700, i.e. about 5%; superclass entropy 1.47 bits; 404,336 spikes per observation; 1,083 KC spikes.

Kenyon-cell activity needs about **8 observations to reach 90% of its plateau** (~4,410 spikes). A first six-observation probe therefore measured a network still warming up and saw only 5 of 7,835 edges move. Re-run at 60 observations per arm, with a third arm receiving no external reinforcement at all:

| Arm | Edges changed vs baseline | Final mean efficacy |
| --- | --- | --- |
| ordered (real reward/aversive pattern) | 3,462 | 0.9415 |
| shuffled (same labels, scrambled order) | 3,460 | 0.9336 |
| **none (no external reinforcement)** | **3,401** | **0.9868** |

| Comparison | Edges differing | Relative L1 |
| --- | --- | --- |
| ordered vs shuffled — *timing* | 3,504 | **2.6%** |
| ordered vs none — *presence* | 3,500 | **11.6%** |

Two things follow, and the second is the important one.

External reinforcement does carry signal: it roughly quadruples the depression, −5.8% mean efficacy against −1.3% with no reward at all. But the no-reward arm still moves **3,401 edges — 97% as many** as the reinforced arms. Endogenous dopamine writes almost the same *number* of synapses; reinforcement changes how far they move, not how many. Counting changed edges is therefore a misleading metric, in the same way counting decision flips is misleading for the ablation above.

**Presence of reinforcement outweighs its timing by 4.6×** (11.6% vs 2.6%). The rule responds mostly to how much dopamine arrived, not to when it arrived relative to Kenyon-cell activity. Temporal pairing — the part that would constitute credit assignment — is the minority of the effect. Any claim that this system assigns credit to actions must contend with that ratio.

**None of this demonstrates learning, profitability, or accurate fly physiology.** It locates where the current design loses information.

The former dark chart produced no KC spikes in an early three-step probe. A light-background display restored some activity without altering the neural parameters. That is a disclosed sensory-adapter change, not evidence that we found biologically correct vision.

## Re-measured on the shipped physiology

Everything above the olfactory section was measured before the inhibitory
gain, the pulse and the satiety currents changed, and through the visual
pathway alone. Those numbers describe a network that no longer exists. The
four original diagnostics were re-run at the shipped settings, with the
olfactory channel open — which is what the run loop actually does, and what
the earlier runs were silently omitting.

### The one-sided proposals are gone

| Arm | Mean R−L | Frame-to-frame spread | BUY / SELL / HOLD |
| --- | --- | --- | --- |
| baseline | +0.25 | 4.20 | 3 / **3** / 2 |
| mirrored chart | −3.50 | 6.48 | 2 / **4** / 2 |
| header repainted | −0.25 | 5.18 | 5 / **3** / 0 |
| per-side drive equalised | −6.50 | 6.91 | 1 / **7** / 0 |

**17 of 32 observations propose SELL, where the same four arms produced 0 of
24 before.** The baseline now sits 0.25 Hz from zero against a frame-to-frame
spread of 4.20 Hz. The standing bias that dominated this experiment was a
property of the saturated regime, not of the 2.04:1 receptor asymmetry in the
reconstruction — the asymmetry is still there and the decoder is no longer
pinned by it.

No manipulation of the input moves an arm further than the spread inside the
arms, so none of them is measurably doing anything. An earlier version of this
verdict read the sign of the mean and announced that mirroring "flips the
bias"; at ±0.25 Hz against a 4 Hz spread, a sign is a coin toss.

### Participation, before and after

| | As reconstructed | Shipped |
| --- | --- | --- |
| neurons firing at least once | 11.2% | 9.2% |
| participation ratio | 8,832 | 8,111 |
| **superclass entropy** | 1.47 bits | **1.78 bits** |
| spikes per observation | 404,336 | 442,464 |
| **Kenyon spikes** | 1,083 | **146** |

Slightly fewer neurons fire, and what fires is spread more evenly across the
superclasses — the brain participates more broadly while the mushroom body
itself goes sparse, which is the direction the whole exercise was aimed at.

### What is underneath the bias is noise

Ablation, re-run at the shipped settings, with the dead-cell control passing
and the graph asserted byte-identical:

| Silenced | Decisions changed | SD of R−L |
| --- | --- | --- |
| 1% | **4 / 5** | 6.62 |
| 5% | 1 / 5 | 12.09 |
| 20% | 2 / 5 | 12.24 |
| 50% | 2 / 5 | 6.37 |

Silencing 1% of the network now changes four decisions in five, where before
it changed none. The brain genuinely reaches the decision — and the decision
is not stable. By the rule written in the plan before any of this was
measured, a readout that moves chaotically at every ablation level is reading
noise, and the dynamics have to be stabilised before anything else is built on
top. That is the same conclusion the decoder measurement reached from the
other direction with a signal-to-noise of 0.66.

### Reinforcement, re-measured: the credit-assignment finding reverses

Three arms of 60 observations each from an identical reset state, with the
olfactory channel open — ordered reinforcement, the same labels scrambled, and
no external reinforcement at all.

| Arm | Edges changed vs baseline | Final mean efficacy |
| --- | --- | --- |
| ordered | 497 | 1.00122 |
| shuffled | 497 | 1.00164 |
| **none** | **0** | **1.000000** |

**Nothing is written without external reinforcement.** In the saturated regime
the no-reward arm moved 3,401 edges — 97% as many as the reinforced arms — and
that was the basis for saying endogenous dopamine wrote almost the same memory
anyway. It does not. Every plastic change measured here is attributable to the
reward and aversive pulses.

| Comparison | Relative L1 |
| --- | --- |
| ordered vs none — *presence* | 0.18% |
| ordered vs shuffled — *timing* | **0.20%** |

**Timing now outweighs presence 1.11×**, reversing the earlier 4.6× the other
way. The same labels in a scrambled order write a memory that differs from the
ordered one by *more* than the ordered one differs from no memory at all —
which can only happen if the two arms move the same edges in opposite
directions. That is temporal pairing, and it is what credit assignment
requires.

The earlier ratio was a property of the saturated regime, where Kenyon rates
were high and near-constant so there was little for a timing-sensitive rule to
key on. This does not establish that the fly assigns credit to *actions*; it
establishes that the rule is sensitive to when dopamine arrives relative to
Kenyon activity, which is the mechanism such an assignment would need.

### The moving-scene hypothesis is not supported

63% of the retained graph is visual and most of it detects motion, so the plan
proposed a scrolling scene. Tested against a still frame of the *same*
landscape, so that motion is the only difference:

| Arm | Input pixels changing | Visual neurons active | Response churn |
| --- | --- | --- | --- |
| still | 0.00% | 8.91% | 0.026 |
| flight (8 px per observation) | **19.30%** | 8.91% | 0.027 |
| shipped chart | 16.56% | 8.92% | 0.029 |

Moving a fifth of the screen changes how much of the visual system fires by
nothing at all, and how much its response moves by 0.001. The visual system is
running at 8.9% active and is indifferent to what is on the screen. By the
plan's own rule the moving display is dropped rather than built.

## Profit-selected parameter search

`tools/evolve` searches fourteen declared free parameters for profit. It is
disclosed here because `AGENTS.md` forbids *hidden* profit-based action
selection, and because with fourteen parameters and dozens of generations a
profitable fly will be found whether or not anything has been learned.

**What it does not touch.** Wiring. No edge is added, removed, re-signed or
re-routed. The parameters are resting potentials, adaptation constants, gains,
currents and one readout threshold, each already a documented model choice
before any evolution existed, each with bounds declared in `genome.py` and not
widened after seeing results. A worker asserts the graph unchanged after every
genome it evaluates; a leak would look exactly like evolution working.

**What is selected on.** Profit, and only profit, on the network's own
proposals. Nothing overrides, replaces or second-guesses a BUY, a SELL or a
HOLD; the harness fills them.

**Protocol, fixed before the first run.**

| | |
| --- | --- |
| Split | chronological 60 / 20 / 20, never shuffled |
| Test segment | opened exactly once, at the end |
| Fitness | median profit over independent chronological starts |
| Screening | 20 observations × 1 start; top third proceed |
| Full evaluation | 50 observations × 5 starts |
| Warm-up | 15 observations, run but not traded |
| Baselines | buy-and-hold, all-cash, random fly, wild-type fly |
| Kill criterion | a champion that does not beat **all four** out of sample is reported as noise |
| Reported | the whole final population out of sample, not only the winner |

**Two deviations from the plan, both forced by measurement.**

The plan asked for five random seeds per genome. The kernel is deterministic —
the same genome on the same prices gives a byte-identical spike train — so
five seeds would have produced five identical numbers and a false impression
of robustness. Independent chronological start points vary the thing that can
actually vary, and that is what runs.

The plan asked for shared base weights with per-genome sparse deltas, to keep
eight workers under about a gigabyte. Measured, a worker holding its own brain
and reconfiguring it in place costs 0.6 GB, so six workers ran in 3.6 GB and
the optimisation was not needed.

**What a champion is not.** The evolution harness uses an explicit paper
simulator with budget, inventory, order size and the same fee rate. It does
**not** apply the live guard's cooldown, spread, quote-age or STOP checks, so
a champion is not a validated trading result until it has been re-run through
`python -m stonkfly run`.

And there is a measured reason to expect little. The best readout tested has a
signal-to-noise of 0.66: the market moves it less than silencing 5% of
unrelated neurons does, and silencing 1% of the network changes four decisions
in five. Selection pressure therefore acts partly on noise. The multi-start
median, the four baselines and the kill criterion exist precisely so that this
shows up as a failed criterion rather than as a champion.

What a search can act on has improved, though. The learning rule is now
sensitive to reinforcement *timing* rather than only to dose, and writes
nothing at all without reinforcement, so the earlier qualification — that
evolution could only tune how much dopamine arrived — no longer holds.

**On a sine wave.** The offline `fixture` source replays this repository's own
deterministic sine. A sine is not a market. Profit there would show that the
evolution machinery works end to end, and nothing about trading.

### First run, 2026-09-14: the criterion fired

A deliberately small machinery check — 8 genomes, 2 generations, 6 workers,
1,200 fixture observations — run to prove the harness end to end, not to find
a trading strategy.

| | Train | Validation | Test |
| --- | --- | --- | --- |
| champion | −1.2097 | −1.3017 | **−1.1425** |
| other survivor | | −0.663 | −0.828 |
| buy-and-hold | | +0.2854 | −0.8144 |
| all-cash | | 0.0000 | **0.0000** |
| random fly | | −2.0492 | −1.7505 |
| wild-type fly | | 0.0000 | 0.0000 |

The champion beat the random fly and lost to everything else, so by the
criterion declared before the run it is reported as **noise**. The harness
works: degenerate genomes fell from 5 of 8 to 1 of 8 between the two
generations, so selection was doing something, and what it selected did not
survive the segment boundary.

Two details are worth more than the verdict.

**The train-best fly was the out-of-sample worst.** Of the two survivors, the
one that won on training lost to the other on both validation and test. With
eight genomes and two generations there was barely any search, and the
overfitting is already visible. This is what the multi-start median and the
held-out segments exist to show.

**Trading costs money.** On this sine, buy-and-hold loses 0.81 on the test
segment and all-cash makes exactly zero, because a 0.6% fee on a 10 unit order
needs a 1.2% round trip to break even. Beating all four baselines means being
strictly profitable after fees, which is a hard bar and the right one.

## Reproduce

```sh
python -m pytest -q
OPENBLAS_NUM_THREADS=1 STONKFLY_FULL_TEST=1 python -m pytest -q
python -m stonkfly verify
python -m stonkfly run --fixture --fast --steps 6 --out runs/check-fixture
python -m stonkfly run --fast --steps 6 --out runs/check-public
python tools/diagnose.py
```

The `run --fast` command without `--fixture` reads current public prices and simulates fills; `tools/diagnose.py` is offline and read-only. Prices, signals and paper outcomes will differ. All runtime evidence stays local in `runs/`; it is not uploaded with this report. A normal `run` omits `--fast` and samples at the configured wall interval.

Before claiming learned performance, implement the held-out replay, shuffled reinforcement, exposure baselines, retention and memory-reset comparisons described in [the model](model.md). This repository currently provides a functioning experimental loop, not that empirical result.
