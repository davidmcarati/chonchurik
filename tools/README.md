# Diagnostics

Read-only measurements of the controller. Nothing here is imported by the
runtime, and nothing here lives under `stonkfly/` — `cli.py` hashes every
`.py`/`.cpp` in the package into `provenance_sha256`, so a diagnostic placed
inside the package would invalidate every existing run directory.

Requires a prepared dataset (`python -m stonkfly prepare`).

```sh
python tools/diagnose.py                      # all four tests
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

The verdict strings encode these rules, so read them rather than eyeballing
the raw tables.

## Method notes

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
