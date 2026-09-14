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

| Rendering | Receptors differing | Input correlation | Identical Kenyon codes | Kenyon activity range |
| --- | --- | --- | --- | --- |
| current (3px polyline) | 4.19% | 0.906 | **10 / 28 pairs** | 0.10% – 34.6% |
| filled area under curve | 14.87% | 0.752 | **0 / 28** | 0.05% – 35.4% |
| filled + fixed price axis | 17.25% | 0.760 | 6 / 28 | 0.10% – 35.4% |

**A rally and a crash reach the retina as 98.5% the same image** (1.47% of receptors differ, correlation 0.959). A 3-pixel line on a 320×180 chart puts the market on a few percent of the receptors; reversing its direction changes the light at roughly fifty of 3,335 cells.

The consequence is that **10 of 28 state pairs produce a byte-identical Kenyon-cell code** — a crash and a choppy market are literally indistinguishable in memory. No reinforcement rule can bind an outcome to a situation that has no distinct representation. The learning machinery measured above is working, but it is working on nothing.

Filling the area under the curve removes every collision (10 → 0) and separates the input 3.5×. Adding a fixed price axis separates the input further still (4.1×) but reintroduces six collisions, so raw input separation is the wrong thing to rank on.

**Sparseness is not fixed by any of this.** Every rendering leaves Kenyon activity bimodal — either near-silent (~0.1%) or about a third of the population — and never near the ~5% a real mushroom body holds. A sub-2% change in retinal input flips 80× swings in Kenyon spiking. The population has two attractors and the picture only chooses between them, so sparseness is set by network dynamics, not by the display. Fixing the input is necessary and not sufficient.

Both APL cells are present in the graph and correctly wired — 2,259 and 2,374 inhibitory outputs onto Kenyon cells (net −26,274 and −27,681), with 2,270 and 2,423 Kenyon inputs returning to them. The feedback loop that enforces sparseness in a real mushroom body exists here; it is not holding the population in range.

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
