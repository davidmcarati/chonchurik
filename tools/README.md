# Diagnostics

Read-only measurements of the controller. Nothing here is imported by the
runtime, and nothing here lives under `stonkfly/` — `cli.py` hashes every
`.py`/`.cpp` in the package into `provenance_sha256`, so a diagnostic placed
inside the package would invalidate every existing run directory.

Requires a prepared dataset (`python -m stonkfly prepare`).

```sh
python tools/diagnose.py                      # every test
python tools/diagnose.py laterality --frames 8
python tools/diagnose.py ablate --levels 1 5 20 50 --repeats 5
```

Results land in `tools/results/diagnostics.json`. One observation costs about
2.5 s of single-threaded compute, so the default run is a few minutes and
leaves the machine usable.

## What each test answers

| Test | Question | Reads |
|---|---|---|
| `baseline` | How much of the brain participates at all? | active fraction, participation ratio, superclass entropy, spread of `difference_hz` |
| `laterality` | What produces the standing R>L drive behind the one-sided proposals? | four arms: baseline, mirrored, header repainted, per-side drive equalised |
| `separation` | Can the fly tell one market state from another? | per-receptor input separation and Kenyon-code collisions across 8 market shapes and 3 candidate chart renderings |
| `ablate` | How much of the brain actually changes the decision? | spread of `difference_hz` against the decision threshold, as neurons are silenced |
| `sparseness` | Is there any parameter regime where the Kenyon code is sparse? | `kc_rest`, APL gain, adaptation and lamina bias, one at a time and crossed |
| `olfaction` | Does the odour channel separate what the chart could not? | glomerular and Kenyon codes over peak current x APL gain |
| `kc_input` | Do the same loudest Kenyon cells always win? | code overlap with summed input onto each Kenyon cell equalised |
| `odor_tuning` | Is the shared Kenyon code inherited from the receptors? | input overlap against Kenyon overlap across four tuning widths |
| `pathway` | Which synapse does the market signal stop surviving? | overlap layer by layer out from the receptors |
| `inhibition` | Does one excitation/inhibition ratio explain every saturation? | every layer's overlap and Kenyon sparseness across inhibitory gains |
| `pulse` | How much current does each dopamine compartment actually need? | per-cell spikes in PAM11 against PPL101 |
| `decoder` | Would a wider readout have more market signal than noise? | spread across market states over standard deviation under ablation |
| `reinforce` | Does plasticity carry the reinforcement signal, or only its amount? | plastic weights under three arms: ordered, shuffled, and no external reinforcement |

`tools/scenes.py` holds the canonical market shapes and the candidate chart
renderings. Prototype encodings live there rather than in
`stonkfly/display.py` so they can be compared before the package changes.

## Measured before these tests were written

On the released MaleCNS v1.0 data, not on anything this repo computes:

```
R1-R6 photoreceptors   L 1,112   vs  R 2,265    ratio 2.04
mapped retina          L 1,107   vs  R 2,228    ratio 2.01
summed retinal drive   L 27,836  vs  R 59,238   ratio 2.13
```

Every downstream structure is symmetric to within 1% — lamina, `ol_intrinsic`,
`visual_projection`, descending neurons, Kenyon cells. The laterality is in the
reconstruction itself.

Two consequences shaped `laterality`:

- **A mirrored chart cannot fix it.** Mirroring moves the per-cell drive gap
  from `+1.4423` to `+1.5302` — slightly the wrong way. The naive mirror test
  returns null, which is why the arm is kept only as the control that rules
  content out.
- **The dark header bar is a real confound.** Left receptors put 93% of their
  samples in the top half of the viewport and 27.6% inside the header rows,
  against 17.0% for the right. Repainting rows 0–27 in background colour
  collapses the per-cell gap from `+1.4423` to `−0.1406`.

## Reading the results

Three times now, the intuitive metric has pointed the wrong way and the
magnitude metric has been right. Prefer spread and effect size over counts:

| Test | Counting says | Magnitude says |
|---|---|---|
| `ablate` | decision survives 50% ablation, so it is robust | spread reaches 14x the decision threshold, so the label is held by a standing bias |
| `reinforce` | the no-reward arm moves 97% as many edges, so reinforcement barely matters | reinforcement quadruples how far they move |
| `separation` | a fixed price axis separates the input best | it also reintroduces six Kenyon-code collisions, which is what actually blocks learning |
| `olfaction` | no two Kenyon codes are byte-identical, so the states are distinguishable | they share 88% of their active cells; zero identical pairs is what saturation looks like |
| `decoder` | a wider readout beats the shipped one at 1.15 signal-to-noise | at 15 repeats instead of 5 it is 0.64, and the first number was noise in an estimate of noise |

The verdict strings encode these rules, so read them rather than eyeballing
the raw tables.

## Method notes

**Every physiology change is restored and checked.** `sparseness`, `olfaction`,
`kc_input`, `odor_tuning`, `pathway` and `inhibition` all alter resting
potentials, adaptation, tonic current or synaptic gain inside a context
manager and assert `ptr`, `post` and `weight` byte-identical on the way out.
An arm that leaks would silently contaminate every later arm in the same run.

**Read overlap, not identical pairs.** When 38% of the Kenyon population
fires, no two codes are ever byte-identical and a collision count reads as a
perfect score while the representation carries nothing. The verdicts rank on
mean Jaccard overlap and treat a code as usable only if it is also sparse in
every market state.


**Silencing, not pruning.** `ablate` hyperpolarises cells with a large negative
`tonic` current — the in-silico analogue of Kir2.1/GtACR. No node or edge is
removed; `ptr`, `post` and `weight` are asserted byte-identical afterwards, so
this stays inside the `AGENTS.md` rule against pruning. It is also cheaper than
baseline, because silenced cells drop out of the kernel's active set.

**Dead-cell control.** Before reading any ablation number, the test silences
only cells that produced zero spikes at baseline. That must change the decision
by exactly nothing. If it does not, the silencing mechanism leaks and every
ablation figure is uninterpretable — the test says so and the verdict refuses
to draw a conclusion.

**Decoder cells are never silenced.** The two DNp20 and two DNpe017 cells are
excluded from every lesion set; otherwise the test would measure its own
intervention.

## Evolution

`tools/evolve` is a separate, disclosed experiment: a profit-selected search
over fourteen declared free physiological parameters. It never touches wiring
and never places an order.

```sh
python -m tools.evolve --source fixture --generations 10 --population 24
python -m tools.evolve --source candles --candles data/candles.json --resume
```

Offline by default and never opens a socket. `fixture` replays the
repository's own deterministic sine, which is a machinery check and not a
market. `candles` replays real one-minute closes from a file a human fetched
on purpose, so a run cannot quietly acquire data or change the segment it is
judged on halfway through.

`--workers` defaults to 8. Each worker holds one brain, measured at 0.6 GB,
and runs at below-normal priority so the machine stays usable; raise it for an
unattended night. State is written every generation, so `--resume` continues
an interrupted run rather than restarting it.

The split, the four baselines and the kill criterion are in
[docs/validation.md](../docs/validation.md#profit-selected-parameter-search).
Read `champion.json` bottom-up: the verdict first, then the population out of
sample, then the genome.
