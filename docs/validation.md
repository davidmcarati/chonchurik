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

**What is selected on.** Profit in excess of buying and holding the same
window, on the network's own proposals. Nothing overrides, replaces or
second-guesses a BUY, a SELL or a HOLD; the harness fills them. Selection and
grading are deliberately different numbers: the kill criterion below still
compares **absolute** profit against the four baselines. Grading a search on
its own objective would make the criterion unfalsifiable. The objective was
absolute profit until 2026-09-14; what changed it is recorded below.

**Protocol, fixed before the first run.**

| | |
| --- | --- |
| Split | chronological 60 / 20 / 20, never shuffled |
| Test segment | opened exactly once, at the end |
| Fitness (selection) | median of profit − buy-and-hold on the same window, over independent chronological starts |
| Fitness (grading) | median absolute profit, against the baselines |
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

### The objective was wrong, and generation 0 showed it

The first generation on real candles produced the first positive number in
the project, and it was worth nothing. (On *which* candles is no longer
knowable; see the note at the end of this section.)

| Start | Champion | Buy-and-hold | Ceiling |
| --- | --- | --- | --- |
| 0 | −2.0768 | −0.9541 | 0.965 |
| 15,642 | −1.8165 | −1.2644 | 0.684 |
| 31,284 | +1.3669 | +2.1698 | 39.684 |
| 46,925 | +4.6802 | +5.3981 | 70.559 |
| 62,567 | +0.1866 | +1.1100 | 10.156 |
| **median** | **+0.1866** | **+1.1100** | **10.156** |

Fitness **+0.1866**, and it **lost to buy-and-hold at five starts out of
five**, every time by roughly the fees it paid. Its behaviour says why: 282 to
289 BUY proposals out of 300 observations, 4 to 9 SELL, and 264 to 276 of the
proposals rejected for want of budget. It bought everything it could afford
and then held. That is not a policy, it is a baseline with a fee drag.

This is not the fly failing. It is the objective being wrong. In a window where
price rises, the profit-maximising policy is maximum exposure, so selecting on
absolute profit pushes the population towards buy-and-hold — while the kill
criterion asks the champion to *beat* buy-and-hold. The search was being driven
towards the thing it would be failed for becoming.

Two changes, and the distinction between them matters:

- **The objective moved.** Fitness is now median *excess* over buy-and-hold on
  the same window, which removes the market's own drift from what is selected
  and leaves the timing.
- **The criterion did not move.** It still grades absolute profit against all
  four baselines, out of sample, once. Moving the criterion after seeing a
  result is the failure mode this whole document exists to prevent; moving the
  objective so that it points at the criterion is a fix.

The degeneracy screen also let this fly through, because it tested for one
proposal at *every* observation and the fly sat at 96%. It now rejects any
genome proposing one side at 90% or more of observations, with the measured
289-of-300 shape kept as a regression test.

Generation 0 also cost **5,600 seconds**, not the ~2,900 estimated, putting a
twelve-generation run at 18.7 hours rather than 10.

**What the new objective's zero point means.** Excess fitness scores exactly
0.0000 for behaving like the benchmark, so "become buy-and-hold" is a tie
rather than a win, and any fly with positive excess beats it. A population
converging on 0.0000 therefore does not mean the search broke; it means
nothing found beat buying and holding, and the kill criterion reports that,
because beating a baseline is a strict inequality and a tie is not one. The
degeneracy screen catches the same shape earlier and more cheaply, but it
looks at a single screening start and can miss a genome that is one-sided only
at some starts — one such start appeared in the first verification run, at 18
BUY and 0 SELL of 20 observations, scoring an exact 0.0000 against a benchmark
it had accidentally reproduced. The two protections overlap on purpose.

### What the money says about the readout

Generation 0 of the second run reproduced the first run's initial population
exactly -- same seed, same twenty-four random genomes -- and ranked
it on the new objective. The same fly that scored **+0.1866** on absolute
profit scored **-0.8029** on excess. Its five profits are identical to the
cent; only the ruler changed.

That coincidence makes a clean decomposition possible. The benchmark pays fees
too: it fills nine orders deploying the budget. This fly filled 17 to 27.

| Window | Its profit | Buy-and-hold | Its fills | Benchmark fills | Extra fees | Shortfall | Left for timing |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | −2.0768 | −0.9541 | 27 | 9 | 1.08 | 1.12 | 0.04 |
| 15,642 | −1.8165 | −1.2644 | 17 | 9 | 0.48 | 0.55 | 0.07 |
| 31,284 | +1.3669 | +2.1698 | 21 | 9 | 0.72 | 0.80 | 0.08 |
| 46,925 | +4.6802 | +5.3981 | 21 | 9 | 0.72 | 0.72 | −0.00 |
| 62,567 | +0.1866 | +1.1100 | 23 | 9 | 0.84 | 0.92 | 0.08 |
| **mean** | | | | | **0.77** | **0.82** | **0.06** |

**Ninety-four percent of the underperformance is the fee on the extra trades.**
What is left for the trading decisions themselves is 0.06 USDC, and it is flat
across windows that range from a 2.08 loss to a 4.68 gain: 0.04, 0.07, 0.08,
−0.00, 0.08.

The fly is not trading badly. Its decisions are worth approximately nothing,
and it pays 0.77 for the right to make them.

**This is the third independent measurement of the same thing.** Two were
neural and indirect; this one is in money:

| Measurement | Reading |
| --- | --- |
| decoder signal-to-noise | **0.66** — the market moves the readout less than silencing unrelated neurons does |
| decisions changed by silencing 1% | **4 of 5** — unstable to noise the market never touches |
| timing's contribution to P&L | **0.06 USDC** against a 0.77 fee bill |

Three methods with nothing in common agree that the readout carries no market
information. The parameter search is currently tuning physiology around a
readout that reads noise, which is worth knowing before any champion is
believed.

**What would have to change.** For excess to go positive, the timing has to be
worth more than the fee bill it creates -- today about thirteen times more.
Two routes exist and neither is evolution finding a lucky genome. The first is
a readout with signal above its own noise, which is the unfixed bottleneck
every measurement above points at. The second is arithmetic: trade less often,
or pay the 0.40% maker fee instead of the 0.60% taker fee the fill-or-kill path
always incurs. A fly filling nine orders pays exactly what the benchmark pays,
and the objective already rewards that without anyone adding a rule for it.

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

### The first run's candles are gone, and its label was never right

`runs/evolution-5min` was named by hand, and the name cannot be true: no
candles file in this repository contains a `FIVE_MINUTE` series, and
`data/candles.json` -- fetched four hours *before* that run started -- holds
`ONE_MINUTE`, `FIFTEEN_MINUTE`, `ONE_HOUR`, `SIX_HOUR` and `ONE_DAY` only.

Nor is the interval recoverable from the run itself. Every evaluation records
its buy-and-hold benchmark exactly, so the five recorded values are a
fingerprint of the price series and the window; searching all five
granularities, all three segments and every evaluation length from 20 to 400
observations reproduces none of them. Whatever file that run read is not in
the repository.

What survives is what the run wrote down: a champion at **+0.1866** absolute
that proposed BUY at 96% of observations, had most of those rejected for want
of budget, and lost to buying and holding at five starts out of five. Those
numbers are still in the run's own `population.json` and they are still what
motivated changing the objective -- the argument is that maximum exposure
maximises profit in a rising window, which does not depend on the bar length.
The directory is kept as `runs/evolution-mislabelled` so nothing cites it as
reproducible, because it is not.

The run that replaced it is on `ONE_HOUR`, chosen by `tools.evolve.survey`:
`ONE_MINUTE` is untradeable on all three segments, and `FIFTEEN_MINUTE` is
tradeable on train while validation needs a 111-bar hold and test pays more for
holding than for trading -- which would produce a noise verdict for structural
reasons rather than anything about the fly. `ONE_HOUR` is tradeable on all
three, with holds of 11 to 14 bars inside a 100-observation evaluation.

## The GPU path, and what makes it the same animal

An evolution is thousands of observations and 95.2% of each one is the spiking
kernel, measured by `tools/kernel_cost.py`. `stonkfly/neural/kernel.cu` moves
that kernel to the card, one fly per CUDA block, and `tools/evolve --device
gpu` runs a whole wave of flies as one launch.

A port that is *nearly* the same is worse than no port. Spike counts would stay
plausible, profits would stay plausible, a champion would still appear, and
every number in this document would silently describe a different animal. So
the requirement was equality, not agreement, and two checks enforce it.

**The kernel.** `tools/gpu_port_check.py` advances one brain on both kernels
from the same state and compares all eighteen mutable arrays element for
element. After a full 5,000-tick observation, at batches of 8, 16, 48 and 84,
with the plasticity rule off and on: identical, every element. The rule has to
be forced to run -- nothing in an offline fixture makes a dopaminergic neuron
spike, and with none firing both weight arrays agree because neither kernel
writes to them -- so the check drives those cells and reports how many weights
actually moved.

Three things had to be true for that, none of them obvious:

- **`exp` does not agree across the two machines.** This device's
  `exp(double)` differs from the host compiler's by one unit in the last place
  on up to 6.2% of arguments. Everything the host keeps at double precision
  therefore comes from a table the host's own compiler filled; a device-built
  eligibility table disagreed on 1,048,575 entries out of 1,048,576. What the
  host rounds to float absorbs the difference: over 4,194,304 draws spanning
  every scale the plasticity rule reaches, `weight * exp(arg)` narrowed to
  float disagreed zero times.
- **The device flushes subnormal floats to zero**, in every arithmetic path,
  and `--ftz=false` does not change it. It also lies about them twice: `x ==
  0.f` is true for a subnormal and `(double)x` is `-0.0`, so both the guard and
  the widening go through the bits. A subnormal is an exact multiple of
  2^-149, so the operands are decoded bitwise, multiplied in double and
  reassembled -- verified against the host on 20,007 values.
- **The compiler reassociates.** The voltage update written plainly compiles to
  `rest + ((t1 + t2) + t3)` while the host computes `((rest + t1) + t2) + t3`,
  and the two differ by half a unit in the last place, which is a spike for a
  cell sitting on the threshold. `--fmad=false` does not prevent it;
  round-to-nearest intrinsics do.

**The whole fly.** Between the launches sit the retina, the olfactory and
gustatory channels, the memory rule, the accounts and the decoder.
`tools/herd_check.py` runs the same genomes through `evolve.evaluate`, the
worker path, and through the herd, from the same prices and the same starts,
and compares the rows: profit, final equity, each proposal count, fills,
rejections, Kenyon spikes. Eight genomes at one start and three genomes at two
starts, including flies that trade and one that makes +0.5258 at one start and
-0.1133 at the other: every field equal.

The sensory front end and the memory rule still run on the CPU, in the
functions the shipped path calls -- `prepare_drive` and `rgb_bin` are
extractions from `_neural_step` and `rgb_step` with their bodies unchanged,
and the memory rule is `rule.advance` itself, once per fly per bin. Only the
integration moved.

**Speed.** 9.7 fly-observations a second against 2.08 for the eight-worker
pool. `tools/herd_cost.py` splits it, at a batch of 84:

| | seconds | share |
| --- | --- | --- |
| kernel | 90.3 | 61.7% |
| memory rule | 39.9 | 27.3% |
| sensory front end | 10.6 | 7.3% |
| upload | 5.5 | 3.7% |
| charts and accounts | 0.9 | 0.6% |

The kernel is still the largest bucket, which is the answer to whether moving
it was the right thing to move. The rule is second and it stays as it is: a
batched version was written and tested equal, and was *slower* -- one fly's
traces are 63 kB and stay in cache, a herd's are 5 MB and do not. Its real
cost is two matrix-vector products against a float32 `gain` with float64
rates, which makes numpy promote a 1 MB matrix on each. Passing a float64
`gain` is 7.5x quicker and moves the result by one unit in the last place, so
it is a change to the model rather than to the runner, and it has not been
made.

**Two faults this work found, neither of them in the port.**

`decoder_threshold_hz` is a dead gene. `configured()` writes it into a
replaced `Settings`, and the `Decoder` read its threshold when the controller
was built and never looks at `Settings` again. One of the fourteen declared
free parameters has been doing nothing, in every evolution run so far. The
herd reproduces that rather than quietly fixing it, because matching the CPU
path is what the herd is for.

`tools/evolve/__main__.py` had not parsed since the commit that added
`plan.json`: writing that file put a literal newline inside a string literal,
so the whole evolution entry point raised `SyntaxError` before argparse ran.
Nothing caught it, because no test imports a `__main__` and every module that
is imported was fine. `tests/test_sources.py` now compiles every source file
in the repository, which is the general form of that check.

**What this does not claim.** The card makes the same fly faster. It does not
make it better: the readout is still noise-limited at a signal-to-noise ratio
of 0.66, the visual pathway still has no operating point, and nothing here has
traded at a profit. A faster search over a noisy objective finds an overfit
champion sooner, which is what the held-out segments and the four baselines
exist to catch.

## What the search was actually selecting, 2026-09-15

Three generations ran on real hourly candles with profit in excess of buying
and holding as the objective. The curve looked like progress: best fitness
-0.5171, then -0.2291, then -0.0680, converging on zero. It was not progress,
and the reason took three measurements to reach.

### Fitness was a function of how little a fly traded

Across the 28 survivors of generation 2, fitness correlated **-0.97** with the
number of sells proposed and **-0.97** with the number filled. Rank 0 sold 7
times over 500 observations; rank 27 sold 99. The ranking was a ladder of
trade counts.

The champion proposed BUY at 94 to 100 of 100 observations at every one of its
five starts, had 83 to 91 of those rejected for want of budget, and scored an
excess of **exactly 0.0** at two starts -- because with no sells at all it was
buy-and-hold, not a policy. `evaluate.degenerate` exists to catch that and
did not: it is applied to the screening row only, where the same fly sat at
0.88 one-sidedness against a bar of 0.90. Every one of the top three failed
that bar at all five full-evaluation starts, and the maximum screening share
across all 28 survivors was 0.88 -- the population had been pressed flat
against the guard from underneath.

A round trip costs the fee twice. At chance-level direction every trade loses,
so the cost term in a money objective is much louder than the skill term, and
the search optimises the loud one.

### Underneath that, it was selecting the sign of a constant

`difference_hz`, the readout the decoder thresholds, measured over 80
observations:

| genome | resting difference | spread | proposals |
| --- | --- | --- | --- |
| wild type | **-17.15 Hz** | 5.57 | 0 BUY / 80 SELL / 0 HOLD |
| rank 0 (fittest) | **+9.05 Hz** | 5.77 | 64 / 0 / 16 |
| rank 2 | +8.60 Hz | 4.76 | 76 / 1 / 3 |
| rank 27 (least fit) | +4.42 Hz | 6.34 | 50 / 12 / 18 |

The offset moved 26 Hz between the wild type and the fittest evolved genome.
The spread did not move at all. Three generations changed the constant and
never touched the variation, and the proposals follow the constant's sign
exactly.

`decoder_threshold_hz` is searched over 0.5 to 12 Hz. Against an offset of 17,
thresholds of 0.5, 2.0 and 6.0 produce byte-identical behaviour -- 12 SELL out
of 12 -- so a genome could choose between "always SELL" and "always HOLD", and
**BUY was unreachable at any threshold in the declared range**. The one genome
whose offset sat near zero, and therefore the only one making genuine
three-way decisions, ranked last of the survivors.

This also rewrites a stage-0 finding. "The decision is stable under 1%
ablation" was read as the readout being robust. It was the offset being larger
than the signal.

### And there was no signal to select for anyway

`tools/ic.py` grades a signal against the return some bars ahead, with the
window's own drift demeaned away so a fly riding a rising market scores
exactly zero, and against a null built by **circularly shifting** the signal
rather than shuffling it -- proposals come in runs, and shuffling destroys
that autocorrelation and yields a null far too narrow to fail against.

Proposals carry nothing: the largest deviation over thirty tests was z 2.49,
which is what thirty tests look like under a null. One genome's readout
reached ic 0.2408 at z 3.18 on train and fell to **0.049 at z 0.55** on
validation.

`tools/features.py` then asks the same of the descriptors before any neuron
sees them. The five close-derived ones score **0 of 30 tests under p 0.05 on
10,446 hourly bars**, and the same on every other timeframe. Nothing was
destroyed downstream because nothing was supplied upstream.

### The arithmetic said it was unwinnable regardless

Break-even information coefficient, from the per-bar volatility of each
interval against the round-trip cost:

| bars | held | Coinbase, 130 bp | Binance, 20 bp |
| --- | --- | --- | --- |
| ONE_MINUTE | 6 | 12.20 | 1.88 |
| ONE_HOUR | 6 | **1.08** | 0.17 |
| ONE_HOUR | 24 | 0.54 | 0.08 |
| SIX_HOUR | 24 | 0.21 | **0.03** |
| ONE_DAY | 24 | 0.08 | 0.01 |

An information coefficient of 1.0 is perfect foresight. Hourly trading at
Coinbase's fee needed **more than perfect foresight** to break even, so no
brain could have won and the objective was unreachable before any of the above
mattered. Real signals live between 0.02 and 0.05, which only the bottom right
of that table admits.

### What changed as a result

- **Venue.** `tools/fetch_binance.py` fetches klines at 10 bp a side.
  Execution is unchanged and still goes to Coinbase Advanced.
- **Input.** Three whole-bar odour channels, `flow`, `flow_slow` and
  `bar_position`. Measured before adoption: order flow and the bar's closing
  position carry the same sign in all four panels -- both timeframes, train and
  validation -- at about **-0.03** one bar out and gone by six. Small, and the
  only effect any descriptor in this repository has shown. Volume, trade size
  and bar range measured nothing and were left out; 53 glomeruli are a fixed
  budget. Cost measured, not assumed: 32.5 of 53 glomeruli active against 33.3
  before, and Kenyon activity 247 spikes an observation against 241, inside the
  per-observation spread of 203 to 302.
- **Decoder.** The threshold is read per observation, in both the CPU path and
  the herd, where it comes from each fly's own genome. The decision is taken
  against the fly's own resting difference -- the median of its last 60
  readouts, appended after each decision and never before, so it is made only
  of observations already past. It is a normalisation and not a policy: it
  never sees a price, a return or a balance. Passing zero reproduces every
  result recorded before it existed. At thresholds of 1, 3 and 8 Hz the wild
  type now proposes 10/10/5, 6/7/12 and 3/2/20 BUY/SELL/HOLD, against 12 SELL
  out of 12 at every threshold before.
- **Objective.** Selection is the readout's information coefficient, median
  over starts. It demeans the signal, so the constant the previous search was
  really optimising contributes exactly nothing to it, and it carries no fee
  term, so it cannot collapse into a trade count. The kill criterion is
  unchanged and still grades absolute profit against the four baselines.
- **Instrument.** The elites are re-scored on validation **every generation**
  and both numbers are printed side by side. Together is an edge; apart is
  luck being fitted. Both failures above took hours to surface and would have
  shown in the second generation.

None of this demonstrates that the fly can predict anything. It removes three
reasons it could not have, and it makes the next negative result arrive in
twenty minutes instead of eight hours.

## Ten generations on the new objective, 2026-09-15

`runs/evolution-binance`: Binance BTCUSDT hourly, 84 genomes, 100 observations
at five chronological starts, selection on the readout's information
coefficient six bars ahead, fee 10 basis points a side. Stopped after ten
generations because the answer did not need the other two.

| gen | champion | ic train | ic holdout | population median | excess | degenerate |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | `bd16660a` | +0.1052 | **-0.0032** | +0.0098 | -0.9682 | 7 |
| 1 | `83671e34` | +0.1310 | **-0.0132** | +0.0031 | -0.5490 | 7 |
| 2 | `83671e34` | +0.1310 | -0.0132 | +0.0106 | -0.5490 | 9 |
| 3 | `83671e34` | +0.1310 | -0.0132 | +0.0231 | -0.5490 | 4 |
| 4 | `952f8a8a` | +0.1382 | **+0.0527** | +0.0184 | -1.1247 | 8 |
| 5 | `8b4c34c6` | +0.1435 | **+0.0007** | +0.0203 | -2.1797 | 4 |
| 6 | `8b4c34c6` | +0.1435 | +0.0007 | +0.0048 | -2.1797 | 1 |
| 7 | `e0bf6752` | +0.1518 | **-0.0236** | +0.0025 | -0.9770 | 0 |
| 8 | `e0bf6752` | +0.1518 | -0.0236 | +0.0111 | -0.9770 | 1 |
| 9 | `e0bf6752` | +0.1518 | -0.0236 | +0.0140 | -0.9770 | 1 |

**Train rose 44% and the held-out score did not move.** Over five distinct
champions the mean train coefficient is +0.1339 and the mean held-out one is
**+0.0027**, against a measurement spread of about 0.05. The search fitted the
training segment, which is what a search does when there is nothing else to
find, and the instrument said so from the second generation rather than after
the run.

Three details worth keeping.

**Elitism is visible in the table and is how the numbers can be trusted.**
Generations 1-3, 5-6 and 7-9 repeat a champion, and both its train and its
held-out score repeat to the last digit. The kernel is deterministic, so an
unchanged genome must re-score identically; a table where that failed would
mean the elite had been lost, which is how the screening defect was found.

**The held-out number was reported wrongly at first.** It returned the two
elites' scores and the printed line took a maximum, so generations 1-3 read
+0.0728 while the champion that earned the train score read -0.0132. The
statistic built to contradict the train score was agreeing with it. It now
reports position zero, the champion's own, and covers eight survivors rather
than two, because two numbers cannot resolve an effect of 0.03 against a
spread of 0.05.

**Degenerate flies fell from 57 of 84 to between 0 and 9.** That is the
resting-difference repair: proposals are no longer pinned to one side by a
constant, and the population explores instead of collapsing.

Money moved the other way, as it must when it is not what is selected: the
champion's excess over buying and holding went -0.97, -0.55, -1.12, -2.18,
-0.98. At this granularity and horizon break-even needs a coefficient of 0.17
and the best held-out reading in the run was +0.0527, so no configuration here
pays for itself even when it predicts.

### What this does and does not show

It shows that selection on this input finds nothing that survives the segment
boundary. It does **not** show that the connectome cannot learn, and the two
are routinely confused. The inputs were already measured to carry almost
nothing -- the five close-derived descriptors score 0 of 30 tests on 10,446
bars, and the whole-bar ones reach -0.03 and are gone by six bars. A search
over an input with no information returns no information, whatever is doing
the searching.

Separating those two requires a positive control: give the fly a channel that
provably predicts and measure whether the readout picks it up. That is
`tools/control.py`, and until it has run, "the flies are not trainable" is not
a claim this repository supports.

## Reproduce

```sh
python -m pytest -q
OPENBLAS_NUM_THREADS=1 STONKFLY_FULL_TEST=1 python -m pytest -q
python -m stonkfly verify
python -m stonkfly run --fixture --fast --steps 6 --out runs/check-fixture
python -m stonkfly run --fast --steps 6 --out runs/check-public
python tools/diagnose.py
python -m tools.gpu_port_check --steps 5000 --batch 84 --learning --reinforce aversive
python -m tools.herd_check --genomes 3 --starts 2 --observations 4
```

The `run --fast` command without `--fixture` reads current public prices and simulates fills; `tools/diagnose.py` is offline and read-only. Prices, signals and paper outcomes will differ. All runtime evidence stays local in `runs/`; it is not uploaded with this report. A normal `run` omits `--fast` and samples at the configured wall interval.

Before claiming learned performance, implement the held-out replay, shuffled reinforcement, exposure baselines, retention and memory-reset comparisons described in [the model](model.md). This repository currently provides a functioning experimental loop, not that empirical result.
