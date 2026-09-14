"""Read-only diagnostics for the Stonkfly controller.

Lives outside the package on purpose: cli.py hashes every .py/.cpp under
stonkfly/ into provenance_sha256, so anything placed there would invalidate
existing run directories. Nothing here imports into the runtime.

Measured before writing this, on the released MaleCNS v1.0 data:

  R1-R6 photoreceptors   L 1,112  vs  R 2,265   ratio 2.04
  mapped retina          L 1,107  vs  R 2,228   ratio 2.01
  summed retinal drive   L 27,836 vs  R 59,238  ratio 2.13

Every downstream structure is symmetric to within 1% (lamina, ol_intrinsic,
visual_projection, descending neurons, KCs). The laterality is in the
reconstruction itself, so a mirrored chart cannot remove it -- confirmed:
mirroring moves the per-cell drive gap from +1.44 to +1.53, i.e. slightly the
wrong way. The informative manipulations are the header repaint and per-side
drive equalisation, which is what `laterality` actually runs.

Tests:
  baseline    how much of the brain participates at all
  laterality  what actually produces the standing R>L drive
  separation  whether market states are distinguishable at all
  sparseness  whether any declared parameter makes the KC code sparse
  olfaction   whether the odour channel separates what the chart could not
  kc_input    whether equalising Kenyon input drive decorrelates the codes
  odor_tuning how sharp the odour must be before the Kenyon codes separate
  pathway     which synapse the market signal stops surviving
  inhibition  whether one excitation/inhibition ratio explains the saturation
  pulse       how much current each dopamine compartment actually needs
  motion      whether a moving scene recruits the visual system a still one does not
  decoder     whether a wider readout has more market signal than noise
  ablate      how much of the brain changes the decision
  reinforce   whether plasticity carries the reinforcement signal
"""

import argparse
import contextlib
import dataclasses
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from stonkfly.config import Settings
from stonkfly.display import market_frame
from stonkfly.market import FixtureMarket
from stonkfly.neural.common import GRAPH, annotations
from stonkfly.neural.controller import FlyController
from stonkfly.neural.olfaction import features as olfactory_features
from stonkfly.neural.sensory import retinal_samples

from scenes import (BASELINE, RENDERERS, SCENES, flight_canvas,
                    render_flight, scene_history)

PRODUCT = "BTC-USDC"
BACKGROUND = (235, 240, 249)
HEADER_ROWS = 28


def frames(count):
    """Deterministic offline observations, produced exactly as the run loop does.

    Returns pictures and the price history each one was drawn from. They are
    returned together because they must not be mismatched: the history opens
    the olfactory channel, and since the inhibitory gain the chart alone
    reaches the mushroom body with two Kenyon cells. A test that passes the
    picture and forgets the history measures a fly with no working senses.
    """
    market = FixtureMarket((PRODUCT,))
    pictures, histories = [], []
    for _ in range(count):
        quotes = market.snapshot()
        q = quotes[PRODUCT]
        history = list(market.history[PRODUCT])
        pictures.append(market_frame(PRODUCT, history, q.bid, q.ask))
        histories.append(history)
        market.record(quotes)
    return pictures, histories


def observe(controller, frame, reinforcement="none", reset=True, history=None):
    """One isolated observation. Reset makes trials independent of each other.

    Without a history no odour is delivered, so every test written before the
    olfactory channel keeps measuring exactly what it measured then.
    """
    if reset:
        controller.brain.reset()
    return controller.observe(frame, reinforcement, history)


def set_learning(controller, learning):
    controller.s = dataclasses.replace(controller.s, learning=learning)
    controller.brain.weights_frozen = not learning


def retina_sides(brain):
    root = annotations(brain.ids).rootSide.fillna("").astype(str).to_numpy()
    side = root[brain.retina]
    return side == "L", side == "R"


def drive_of(brain, frame):
    """Reproduces the retinal drive the kernel will see for this frame."""
    lum = retinal_samples(frame, brain.uv)
    return 30 * lum / (0.02 + lum)


def participation(counts, superclass):
    """How much of the network is involved in producing this observation."""
    counts = np.asarray(counts)
    active = int(np.count_nonzero(counts))
    total = int(counts.sum())
    entropy = 0.0
    effective = 0.0
    if total:
        share = np.array(
            [counts[superclass == k].sum() for k in np.unique(superclass)], dtype=float
        )
        share = share[share > 0] / total
        entropy = float(-(share * np.log2(share)).sum())
        # Participation ratio: effective number of contributing neurons.
        effective = float(total**2 / np.square(counts.astype(float)).sum())
    return {
        "active_neurons": active,
        "active_fraction": active / len(counts),
        "total_spikes": total,
        "superclass_entropy_bits": entropy,
        "effective_population": effective,
    }


# --- baseline --------------------------------------------------------------


def baseline(controller, count, superclass):
    """Reference numbers every later stage is compared against."""
    print(f"[baseline] {count} frames", flush=True)
    stats, diffs = [], []
    pictures, histories = frames(count)
    for i, (frame, history) in enumerate(zip(pictures, histories)):
        n = observe(controller, frame, history=history)
        p = participation(controller.brain.counts, superclass)
        p.update(side=n["side"], difference_hz=n["difference_hz"],
                 KC_spikes=n["KC_spikes"])
        stats.append(p)
        diffs.append(n["difference_hz"])
        print(f"  {i + 1:3}/{count}  active {p['active_fraction']:7.3%}  "
              f"PR {p['effective_population']:9,.0f}  "
              f"H {p['superclass_entropy_bits']:.3f}  "
              f"{n['side']:4} R-L {n['difference_hz']:+7.2f}", flush=True)
    return {
        "frames": count,
        "active_fraction_mean": float(np.mean([s["active_fraction"] for s in stats])),
        "effective_population_mean": float(
            np.mean([s["effective_population"] for s in stats])
        ),
        "superclass_entropy_mean": float(
            np.mean([s["superclass_entropy_bits"] for s in stats])
        ),
        "total_spikes_mean": float(np.mean([s["total_spikes"] for s in stats])),
        "KC_spikes_mean": float(np.mean([s["KC_spikes"] for s in stats])),
        "difference_hz_std": float(np.std(diffs)),
        "difference_hz_distinct": len(set(diffs)),
        "per_frame": stats,
        "verdict": ("decoder output is constant across frames: input never reaches "
                    "the readout" if np.std(diffs) < 1e-9 else
                    "decoder output varies with input"),
    }


# --- laterality ------------------------------------------------------------


def laterality(controller, count, superclass):
    """Which of chart content, header geometry or receptor count drives R>L.

    A1 (mirrored) is kept as the control that proves content is NOT the cause.
    A4 equalises summed per-side drive through `tonic`, the same additive
    current path the reinforcement pulse already uses -- no graph is touched.
    """
    brain = controller.brain
    left, right = retina_sides(brain)
    nL, nR = int(left.sum()), int(right.sum())
    print(f"[laterality] mapped retina L {nL:,} R {nR:,} "
          f"(structural ratio {nR / nL:.3f})", flush=True)

    base, histories = frames(count)
    arms = {
        "A0_baseline": [f for f in base],
        "A1_mirrored": [f[:, ::-1].copy() for f in base],
        "A3_header_repainted": [repaint(f) for f in base],
    }

    result = {"mapped_retina": {"left": nL, "right": nR, "ratio": nR / nL}}
    for name, series in arms.items():
        result[name] = run_arm(
            controller, name, series, histories, left, right, superclass
        )

    # A4: additive tonic on the left retina so summed drive matches the right.
    d = drive_of(brain, base[0])
    delta = float((d[right].sum() - d[left].sum()) / nL)
    saved = brain.tonic.copy()
    brain.tonic[brain.retina[left]] += delta
    try:
        result["A4_drive_equalised"] = run_arm(
            controller, "A4_drive_equalised", base, histories, left, right,
            superclass,
        )
        result["A4_drive_equalised"]["tonic_added_to_left"] = delta
    finally:
        brain.tonic[:] = saved

    result["verdict"] = verdict_laterality(result)
    return result


def repaint(frame):
    out = frame.copy()
    out[:HEADER_ROWS, :, :] = BACKGROUND
    return out


def run_arm(controller, name, series, histories, left, right, superclass):
    brain = controller.brain
    sides, diffs, ratios = [], [], []
    for i, (frame, history) in enumerate(zip(series, histories)):
        d = drive_of(brain, frame)
        ratios.append(float(d[right].sum() / d[left].sum()))
        # The odour is identical across arms; only the picture is manipulated,
        # so a difference between arms is still attributable to the picture.
        n = observe(controller, frame, history=history)
        sides.append(n["side"])
        diffs.append(n["difference_hz"])
        print(f"  {name:20} {i + 1:3}/{len(series)}  drive R/L {ratios[-1]:5.3f}  "
              f"{n['side']:4}  R-L {n['difference_hz']:+7.2f} Hz", flush=True)
    return {
        "sides": {s: sides.count(s) for s in ("BUY", "SELL", "HOLD")},
        "mean_difference_hz": float(np.mean(diffs)),
        "mean_drive_ratio": float(np.mean(ratios)),
        "difference_hz": diffs,
    }


def verdict_laterality(r):
    """Judge an arm against the spread inside the arms, not against zero.

    An earlier version of this read the sign of the mean and concluded that
    mirroring the chart "flips the bias". With the network out of saturation
    the means sit within a fraction of a hertz of zero while the frame-to-frame
    spread is several hertz, so a sign is a coin toss and reading one as a
    finding is exactly the mistake this file keeps catching elsewhere.
    """
    arms = ["A0_baseline", "A1_mirrored", "A3_header_repainted",
            "A4_drive_equalised"]
    means = {k: r[k]["mean_difference_hz"] for k in arms}
    spread = {k: float(np.std(r[k]["difference_hz"], ddof=1)) for k in arms}
    total = sum(sum(r[k]["sides"].values()) for k in arms)
    sells = sum(r[k]["sides"]["SELL"] for k in arms)
    noise = float(np.mean(list(spread.values())))
    table = "; ".join(
        f"{k.split('_', 1)[1]} {means[k]:+.2f}+-{spread[k]:.2f} "
        f"({r[k]['sides']['BUY']}/{r[k]['sides']['SELL']}/{r[k]['sides']['HOLD']})"
        for k in arms
    )
    head = (f"mean R-L per arm, with frame-to-frame spread and "
            f"BUY/SELL/HOLD: {table}. {sells} of {total} observations propose "
            f"SELL")
    moved = [
        k for k in arms[1:]
        if abs(means[k] - means["A0_baseline"]) > spread[k] + spread["A0_baseline"]
    ]
    if abs(means["A0_baseline"]) < noise and sells:
        tail = (f". The standing one-sided bias is gone: the baseline sits "
                f"{abs(means['A0_baseline']):.2f} Hz from zero against a "
                f"frame-to-frame spread of {spread['A0_baseline']:.2f} Hz, so "
                f"the readout is no longer pinned to one side")
    else:
        tail = (f". The baseline still sits {means['A0_baseline']:+.2f} Hz off "
                f"zero against a spread of {spread['A0_baseline']:.2f} Hz")
    if not moved:
        return (head + tail + ". No manipulation of the input moves an arm "
                "further than the spread within it, so none of them is "
                "measurably doing anything")
    return (head + tail + f". Only {', '.join(m.split('_', 1)[1] for m in moved)} "
            f"moves further than the spread within the arms, and that is the "
            f"only manipulation worth pursuing")


# --- ablation --------------------------------------------------------------


def ablate(controller, levels, repeats, seed, superclass):
    """Silence a random share of the brain and see whether the decision moves.

    Silencing uses a large hyperpolarising `tonic` current -- the in-silico
    analogue of Kir2.1/GtACR. No node or edge is removed: ptr, post and weight
    are asserted byte-identical afterwards, so this stays inside the AGENTS.md
    rule against pruning.
    """
    brain = controller.brain
    n = brain.n
    protected = np.zeros(n, dtype=bool)
    for key in ("left", "right", "gate"):
        protected[getattr(controller.decoder, key)] = True
    candidates = np.flatnonzero(~protected)
    rng = np.random.default_rng(seed)
    guard = [brain.weight.copy(), brain.ptr.copy(), brain.post.copy()]

    pictures, histories = frames(1)
    frame, history = pictures[0], histories[0]
    control = observe(controller, frame, history=history)
    quiet = np.flatnonzero(controller.brain.counts == 0)
    print(f"[ablate] intact {control['side']} "
          f"(R-L {control['difference_hz']:+.2f} Hz), "
          f"{len(quiet):,} silent cells available for the null control",
          flush=True)

    def trial(victims):
        saved = brain.tonic.copy()
        brain.tonic[victims] = -1000.0
        try:
            return observe(controller, frame, history=history)
        finally:
            brain.tonic[:] = saved

    # Null control: lesioning cells that never fired must change exactly nothing.
    dead = trial(rng.choice(quiet, size=min(len(quiet), len(candidates) // 10),
                            replace=False)) if len(quiet) else None
    null_ok = dead is None or dead["difference_hz"] == control["difference_hz"]
    print(f"  dead-cell control: {'PASS' if null_ok else 'FAIL - method invalid'}",
          flush=True)

    rows = []
    for level in levels:
        changed, diffs = 0, []
        for r in range(repeats):
            victims = rng.choice(
                candidates, size=int(len(candidates) * level / 100), replace=False
            )
            out = trial(victims)
            changed += out["side"] != control["side"]
            diffs.append(out["difference_hz"])
            print(f"  {level:5.1f}%  trial {r + 1:2}/{repeats}  {out['side']:4}  "
                  f"R-L {out['difference_hz']:+7.2f} Hz", flush=True)
        rows.append({
            "ablated_percent": level,
            "decisions_changed": changed,
            "trials": repeats,
            "changed_fraction": changed / repeats,
            "mean_difference_hz": float(np.mean(diffs)),
            "std_difference_hz": float(np.std(diffs)),
        })

    for name, before, after in zip(("weight", "ptr", "post"), guard,
                                   (brain.weight, brain.ptr, brain.post)):
        if not np.array_equal(before, after):
            raise RuntimeError(f"ablation mutated the graph array {name}")

    return {
        "intact": {"side": control["side"],
                   "difference_hz": control["difference_hz"]},
        "dead_cell_control_passed": bool(null_ok),
        "graph_unchanged": True,
        "levels": rows,
        "verdict": verdict_ablate(
            rows, null_ok, controller.s.decoder_threshold_hz
        ),
    }


def verdict_ablate(rows, null_ok, threshold):
    """Read the spread, not the flip count.

    Counting label flips understates the problem: the label is held in place by
    a standing bias while the quantity under it moves freely. The decisive
    number is the spread of difference_hz against the decision threshold.
    """
    if not null_ok:
        return ("dead-cell control FAILED: silencing leaks, the ablation numbers "
                "are not interpretable -- fix the method before reading them")
    noise = max(r["std_difference_hz"] for r in rows)
    smallest = min(rows, key=lambda r: r["ablated_percent"])
    ratio = noise / threshold if threshold else float("inf")
    if smallest["std_difference_hz"] >= threshold:
        return (f"readout is knife-edge: perturbing {smallest['ablated_percent']}% "
                f"of neurons already moves difference_hz by "
                f"sd {smallest['std_difference_hz']:.2f} Hz against a "
                f"{threshold:g} Hz threshold, and spread reaches {ratio:.0f}x the "
                "threshold under ablation. The label is held by a standing bias, "
                "not by the signal -> the decoder needs to INTEGRATE over the "
                "window, not just widen to more cells")
    if ratio > 3:
        return (f"decision quantity is unstable (spread up to {ratio:.0f}x the "
                "threshold) -> prefer an integrative population readout")
    return ("decision is stable under ablation: the readout is not the "
            "bottleneck; look at the sensory side instead")


# --- separation ------------------------------------------------------------


def separation(controller, superclass):
    """Can the fly tell one market state from another, at input and in memory?

    Associative learning needs distinguishable codes. If a rally and a crash
    arrive as the same pattern, no reinforcement rule can bind an outcome to a
    situation -- the learning machinery would be working on nothing.

    Each candidate chart rendering is scored on input separation (per-receptor
    luminance) and memory separation (which Kenyon cells fired).
    """
    brain = controller.brain
    a = annotations(brain.ids)
    kinds = a.type.fillna("").astype(str)
    kc = np.flatnonzero(kinds.str.startswith("KC").to_numpy())
    apl = np.flatnonzero(kinds.str.fullmatch("APL").fillna(False).to_numpy())
    names = list(SCENES)
    print(f"[separation] {len(names)} market states x {len(RENDERERS)} renderings; "
          f"{len(kc):,} KCs, {len(apl)} APL cells", flush=True)

    out = {"scenes": names, "kenyon_cells": len(kc), "apl_cells": len(apl)}
    for render_name, render in RENDERERS.items():
        lum, code, rows = {}, {}, {}
        for scene in names:
            frame = render(scene_history(scene))
            lum[scene] = retinal_samples(frame, brain.uv)
            n = observe(controller, frame)
            counts = controller.brain.counts
            code[scene] = counts[kc] > 0
            rows[scene] = {
                "kc_active_fraction": float((counts[kc] > 0).mean()),
                "kc_spikes": int(counts[kc].sum()),
                "apl_spikes": int(counts[apl].sum()),
                "side": n["side"],
                "difference_hz": n["difference_hz"],
            }
            print(f"  {render_name:19} {scene:11} KC "
                  f"{rows[scene]['kc_active_fraction']:6.2%}  "
                  f"spikes {rows[scene]['kc_spikes']:6,}  "
                  f"APL {rows[scene]['apl_spikes']:4}  "
                  f"{n['side']:4} R-L {n['difference_hz']:+6.2f}", flush=True)

        pairs = [(x, y) for i, x in enumerate(names) for y in names[i + 1:]]
        corr = [float(np.corrcoef(lum[x], lum[y])[0, 1]) for x, y in pairs]
        differ = [float((np.abs(lum[x] - lum[y]) > 0.1).mean()) for x, y in pairs]
        jac = []
        for x, y in pairs:
            union = float((code[x] | code[y]).sum())
            jac.append(float((code[x] & code[y]).sum()) / union if union else 0.0)
        active = [rows[s]["kc_active_fraction"] for s in names]
        sides = [rows[s]["side"] for s in names]
        out[render_name] = {
            "per_scene": rows,
            "input_mean_correlation": float(np.mean(corr)),
            "input_mean_receptors_differing": float(np.mean(differ)),
            "kc_mean_jaccard": float(np.mean(jac)),
            "kc_identical_pairs": int(sum(j > 0.999 for j in jac)),
            "kc_active_min": float(min(active)),
            "kc_active_max": float(max(active)),
            "distinct_sides": len(set(sides)),
        }
    out["verdict"] = verdict_separation(out)
    return out


def verdict_separation(out):
    """Rank on code collisions, not on raw input separation.

    More input separation is worthless if two market states still land on the
    same Kenyon code -- a collision is exactly what makes an association
    impossible. So identical pairs decide first, mean overlap second, and raw
    input separation only breaks ties.
    """
    cur = out[BASELINE]
    best = min(
        RENDERERS,
        key=lambda k: (out[k]["kc_identical_pairs"], out[k]["kc_mean_jaccard"],
                       -out[k]["input_mean_receptors_differing"]),
    )
    pairs = len(out["scenes"]) * (len(out["scenes"]) - 1) // 2
    # With the shipped excitation/inhibition ratio the chart reaches the
    # mushroom body with a couple of cells whatever it draws. Comparing
    # renderings there measures a pathway that is not delivering, not the
    # pictures, and the comparison has to say so rather than declare a winner.
    floor = max(out[k]["kc_active_max"] for k in RENDERERS)
    if floor < 0.005:
        return (f"the chart reaches the mushroom body with at most "
                f"{floor:.2%} of Kenyon cells in EVERY rendering, so this "
                f"comparison is not measuring the picture. The visual pathway "
                f"does not deliver a code at the shipped excitation/inhibition "
                f"ratio; renderings were last told apart at ratio 1.0, where "
                f"the filled area removed all ten collisions the polyline "
                f"caused. Input separation still differs -- "
                f"{cur['input_mean_receptors_differing']:.2%} of receptors for "
                f"'{BASELINE}' against "
                f"{out['filled']['input_mean_receptors_differing']:.2%} filled "
                f"-- but nothing downstream of the retina is using it")
    line = (f"current chart: a market state changes only "
            f"{cur['input_mean_receptors_differing']:.2%} of receptors "
            f"(correlation {cur['input_mean_correlation']:.3f}), and "
            f"{cur['kc_identical_pairs']}/{pairs} state pairs land on an "
            f"IDENTICAL Kenyon code -- those pairs can never be told apart by "
            "any reinforcement rule")
    bimodal = all(
        out[k]["kc_active_min"] < 0.01 and out[k]["kc_active_max"] > 0.2
        for k in RENDERERS
    )
    tail = ""
    if bimodal:
        tail = (". Note: every rendering leaves Kenyon activity bimodal "
                "(near-silent or ~30% active, never near the ~5% a real "
                "mushroom body holds), so sparseness is set by network "
                "dynamics, not by the picture -- fixing the input is necessary "
                "but not sufficient")
    if best == BASELINE:
        return line + ". No candidate rendering removes more collisions" + tail
    return (line + f". Best candidate '{best}' removes them: "
            f"{out[best]['kc_identical_pairs']}/{pairs} identical pairs, "
            f"overlap {out[best]['kc_mean_jaccard']:.2f}, input separation "
            f"{out[best]['input_mean_receptors_differing']:.2%}" + tail)


# --- reinforcement control -------------------------------------------------


def reinforce(controller, count, seed):
    """Does plasticity depend on reinforcement TIMING, or only on its amount?

    Three arms over identical frames from an identical reset state:

      ordered   the real alternating reward/aversive pattern
      shuffled  the same multiset of labels in a scrambled order
      none      no external reinforcement at all

    `shuffled` isolates temporal pairing. `none` is the arm that matters most:
    PPL101 fires endogenously (18 spikes were logged with stimulus "none" in an
    earlier run), so without it there is no way to tell an externally driven
    memory from one the network would have written anyway.
    """
    brain = controller.brain
    plastic = brain.circuit["edges"]
    set_learning(controller, True)
    series, histories = frames(count)
    rng = np.random.default_rng(seed)
    labels = ["reward" if i % 3 == 0 else "aversive" if i % 3 == 1 else "none"
              for i in range(count)]
    shuffled = list(labels)
    rng.shuffle(shuffled)
    quiet = ["none"] * count

    out, traces = {}, {}
    for name, order in [("ordered", labels), ("shuffled", shuffled),
                        ("none", quiet)]:
        brain.reset()
        trace = []
        for i, (frame, kind, history) in enumerate(zip(series, order, histories)):
            n = observe(controller, frame, kind, reset=False, history=history)
            trace.append({
                "changed_edges": n["memory"]["changed_edges"],
                "KC_spikes": n["KC_spikes"],
                "reward_spikes": n["reward_spikes"],
                "aversive_spikes": n["aversive_spikes"],
                "mean_efficacy": n["memory"]["mean_efficacy"],
            })
            if i % 10 == 0 or i == count - 1:
                print(f"  {name:8} {i + 1:4}/{count}  {kind:8}  "
                      f"changed {n['memory']['changed_edges']:5}  "
                      f"KC {n['KC_spikes']:6,}  "
                      f"DAN r/a {n['reward_spikes']:4}/{n['aversive_spikes']:4}  "
                      f"eff {n['memory']['mean_efficacy']:.6f}", flush=True)
        out[name] = brain.weight[plastic].copy()
        traces[name] = trace

    def gap(x, y):
        scale = max(float(np.abs(x).sum()), 1e-12)
        return {
            "relative_l1": float(np.abs(x - y).sum() / scale),
            "max_abs": float(np.abs(x - y).max()),
            "edges_differing": int(np.count_nonzero(x != y)),
        }

    pairs = {
        "ordered_vs_shuffled": gap(out["ordered"], out["shuffled"]),
        "ordered_vs_none": gap(out["ordered"], out["none"]),
        "shuffled_vs_none": gap(out["shuffled"], out["none"]),
    }
    changed = {
        k: int(np.count_nonzero(v != brain.baseline_plastic))
        for k, v in out.items()
    }
    return {
        "observations": count,
        "plastic_edges": int(len(plastic)),
        "changed_vs_baseline": changed,
        "pairwise": pairs,
        "traces": traces,
        "verdict": verdict_reinforce(pairs, changed, len(plastic)),
    }


def verdict_reinforce(pairs, changed, plastic_edges):
    """Report the size of the effect, not the number of edges touched.

    Edge counts misled here once already: in the saturated regime the no-reward
    arm touched 97% as many edges as the reinforced arms, which read as
    "reinforcement barely matters" while what it changed was how far they
    moved. The relative L1 is the measurement and the counts are context.
    """
    timing = pairs["ordered_vs_shuffled"]["relative_l1"]
    presence = pairs["ordered_vs_none"]["relative_l1"]
    if presence < 1e-6:
        return ("external reinforcement changes NOTHING: the same memory is "
                "written with no reward or punishment at all. The dopamine "
                "pulses are decorative -> STOP, the GA has nothing to select")
    if timing < 1e-6:
        return ("plasticity is order-blind: only the amount of reinforcement "
                "matters, never its timing -> no credit assignment is possible, "
                "fix trace/delay before any GA")
    head = (f"reinforcement carries signal: presence moves the memory "
            f"{presence:.2%} (relative L1), timing moves it {timing:.2%}, "
            f"against an endogenous baseline of {changed['none']:,} edges "
            f"written with no external reinforcement at all")
    if not changed["none"]:
        head += (" -- nothing is written without it, so every plastic change "
                 "measured here is attributable to the reward and aversive "
                 "pulses")
    # Which of the two is larger decides what a profit-selected search can
    # actually act on, so it is stated as a comparison and not asserted.
    if timing > presence:
        return (head + f". Timing outweighs presence {timing / presence:.2f}x: "
                f"the same labels in a scrambled order write a memory that "
                f"differs from the ordered one by MORE than the ordered one "
                f"differs from no memory at all, so the rule is responding to "
                f"WHEN dopamine arrived relative to Kenyon activity. That is "
                f"the temporal pairing credit assignment requires")
    return (head + f". Presence outweighs timing {presence / timing:.2f}x -- "
            "the rule responds mostly to HOW MUCH dopamine arrived, not WHEN. "
            "A profit-selected search can act on dose, not on credit "
            "assignment, and that has to be disclosed")


# --- sparseness ------------------------------------------------------------

# A real mushroom body answers any one odour with a few percent of its Kenyon
# cells. This network was measured at either ~0.1% or ~30% and never between,
# in all three chart renderings -- so the sparseness is set by the dynamics,
# not by the picture. This test asks whether any declared free parameter
# reaches the biological range at all.
TARGET_ACTIVE = 0.05
BAND = (0.01, 0.15)
DEFAULT_LAMINA_BIAS = 12.0
SWEEP_SCENES = ["rally", "crash", "flat", "chop"]
SWEEP_RENDERER = "filled"

# Every lever below is already a declared free parameter. kc_rest,
# adaptation_jump and tonic are constructor arguments of MemoryBrain;
# lamina_bias is documented as a display proxy; the APL gain scales existing
# GABAergic APL->KC edges, whose magnitude is itself a declared proxy (contact
# count x 0.275). No edge or node is added, removed or rewired.
LEVERS = {
    "kc_rest": [-75.0, -70.0, -65.0, -60.0, -55.0, -50.0],
    "apl_gain": [1.0, 2.0, 4.0, 8.0, 16.0],
    "adaptation_jump": [0.0, 8.0, 20.0, 40.0],
    "lamina_bias": [4.0, 8.0, 12.0, 16.0],
}
DEFAULTS = {
    "kc_rest": -60.0,
    "apl_gain": 1.0,
    "adaptation_jump": 8.0,
    "lamina_bias": DEFAULT_LAMINA_BIAS,
}
# The two levers crossed in phase 2. Both act on which cells fire rather than
# on how often, which is what an active fraction measures.
GRID = ["kc_rest", "apl_gain"]


def apl_kc_edges(brain, apl, kc):
    """Existing APL->KC connections. Scaling them changes gain, not wiring."""
    mask = np.zeros(brain.n, dtype=bool)
    mask[kc] = True
    out = []
    for i in apl:
        e = np.arange(brain.ptr[i], brain.ptr[i + 1])
        out.append(e[mask[brain.post[e]]])
    return np.concatenate(out).astype(np.int64) if out else np.empty(0, np.int64)


@contextlib.contextmanager
def physiology(brain, kc, apl_edges, combination):
    """Apply one parameter combination, then put everything back exactly."""
    saved = {
        "rest": brain.rest[kc].copy(),
        "initial_v": brain.initial["v"][kc].copy(),
        "weight": brain.weight[apl_edges].copy(),
        "tonic": brain.tonic[brain.lamina].copy(),
        "jump": brain.adaptation_jump,
    }
    try:
        brain.rest[kc] = combination["kc_rest"]
        brain.initial["v"][kc] = combination["kc_rest"]
        brain.weight[apl_edges] = saved["weight"] * combination["apl_gain"]
        brain.adaptation_jump = float(combination["adaptation_jump"])
        # drive[lamina] = lamina_bias, then drive += tonic, so an offset on the
        # lamina cells is exactly an altered bias without touching the call.
        brain.tonic[brain.lamina] = combination["lamina_bias"] - DEFAULT_LAMINA_BIAS
        yield
    finally:
        brain.rest[kc] = saved["rest"]
        brain.initial["v"][kc] = saved["initial_v"]
        brain.weight[apl_edges] = saved["weight"]
        brain.tonic[brain.lamina] = saved["tonic"]
        brain.adaptation_jump = saved["jump"]


def kc_response(controller, kc, apl, scenes, render, codes=False):
    """Kenyon activity for one parameter combination across market states."""
    rows, code = {}, {}
    for scene in scenes:
        n = observe(controller, render(scene_history(scene)))
        counts = controller.brain.counts
        fired = counts[kc] > 0
        if codes:
            code[scene] = fired
        rows[scene] = {
            "kc_active_fraction": float(fired.mean()),
            "kc_spikes": int(counts[kc].sum()),
            "apl_spikes": int(counts[apl].sum()),
            "total_spikes": int(counts.sum()),
            "side": n["side"],
            "difference_hz": float(n["difference_hz"]),
        }
    return (rows, code) if codes else rows


def summarise(rows):
    active = [r["kc_active_fraction"] for r in rows.values()]
    return {
        "per_scene": rows,
        "kc_active_min": float(min(active)),
        "kc_active_max": float(max(active)),
        "kc_active_median": float(np.median(active)),
        "in_band": bool(all(BAND[0] <= x <= BAND[1] for x in active)),
        "distance_from_target": float(abs(np.median(active) - TARGET_ACTIVE)),
        "distinct_sides": len({r["side"] for r in rows.values()}),
    }


def bistability(rows):
    """Where the population actually sits, over every observation of the sweep.

    The question is not the average activity but whether the intermediate
    range is ever occupied. A population that is bimodal has no sparse regime
    to tune into, however many parameter points are tried.
    """
    active = np.array([r["kc_active_fraction"] for r in rows])
    total = np.array([r["total_spikes"] for r in rows], dtype=float)
    low, high = active < BAND[0], active > 0.2
    return {
        "observations": int(len(active)),
        "below_band_fraction": float(low.mean()),
        "above_20_percent_fraction": float(high.mean()),
        "inside_band_fraction": float((~low & ~high).mean()),
        "whole_brain_spikes_when_kc_silent": float(total[low].mean()) if low.any() else 0.0,
        "whole_brain_spikes_when_kc_active": float(total[high].mean()) if high.any() else 0.0,
        "kc_vs_whole_brain_correlation": float(np.corrcoef(active, total)[0, 1]),
    }


def decisions(rows, threshold):
    """How far past the decision threshold the proposals actually sit.

    Counting sides is what misled the earlier ablation reading. One DNp20 cell
    per side over a 500 ms window makes a single spike worth 2 Hz, which is
    exactly the threshold -- so a margin is only meaningful expressed in
    spikes, and a side that is one spike past the line is a coin flip whatever
    its label says.
    """
    out = {"threshold_hz": float(threshold), "counts": {}, "margins": {}}
    for side in ["BUY", "SELL", "HOLD"]:
        margins = [abs(r["difference_hz"]) for r in rows if r["side"] == side]
        out["counts"][side] = len(margins)
        if margins:
            out["margins"][side] = {
                "median_hz": float(np.median(margins)),
                "max_hz": float(max(margins)),
                "max_spikes_past_threshold": float(max(margins) / threshold),
            }
    return out


def sparseness(controller, superclass):
    """Is there any parameter regime where Kenyon activity is sparse at all?

    Three phases. One lever at a time, to see which of them even moves the
    active fraction and in which direction; then the two levers that act on
    *which* cells fire, crossed; then the surviving combinations re-measured
    on all eight market states, because a regime that is sparse by collapsing
    every state onto the same few cells is worse than no regime at all.
    """
    brain = controller.brain
    kinds = annotations(brain.ids).type.fillna("").astype(str)
    kc = np.flatnonzero(kinds.str.startswith("KC").to_numpy())
    apl = np.flatnonzero(kinds.str.fullmatch("APL").fillna(False).to_numpy())
    edges = apl_kc_edges(brain, apl, kc)
    render = RENDERERS[SWEEP_RENDERER]
    guard = (brain.ptr.copy(), brain.post.copy(), brain.weight.copy())
    print(f"[sparseness] target {TARGET_ACTIVE:.0%} of {len(kc):,} KCs; "
          f"{len(apl)} APL cells over {len(edges):,} APL->KC edges; "
          f"'{SWEEP_RENDERER}' rendering", flush=True)

    out = {
        "target_active_fraction": TARGET_ACTIVE,
        "acceptable_band": list(BAND),
        "renderer": SWEEP_RENDERER,
        "sweep_scenes": SWEEP_SCENES,
        "kenyon_cells": int(len(kc)),
        "apl_cells": int(len(apl)),
        "apl_kc_edges": int(len(edges)),
        "levers": LEVERS,
        "defaults": DEFAULTS,
    }

    print("  phase 1: one lever at a time", flush=True)
    single, seen = {}, []
    for lever, values in LEVERS.items():
        single[lever] = {}
        for value in values:
            combination = {**DEFAULTS, lever: value}
            with physiology(brain, kc, edges, combination):
                rows = kc_response(controller, kc, apl, SWEEP_SCENES, render)
            seen.extend(rows.values())
            s = summarise(rows)
            single[lever][repr(value)] = s
            print(f"    {lever:16} {value:>7}  KC "
                  f"{s['kc_active_min']:6.2%}-{s['kc_active_max']:6.2%}  "
                  f"median {s['kc_active_median']:6.2%}"
                  f"{'  IN BAND' if s['in_band'] else ''}", flush=True)
    out["single_lever"] = single

    print(f"  phase 2: {' x '.join(GRID)}", flush=True)
    grid = {}
    for x in LEVERS[GRID[0]]:
        for y in LEVERS[GRID[1]]:
            combination = {**DEFAULTS, GRID[0]: x, GRID[1]: y}
            with physiology(brain, kc, edges, combination):
                rows = kc_response(controller, kc, apl, SWEEP_SCENES, render)
            seen.extend(rows.values())
            s = summarise(rows)
            grid[f"{GRID[0]}={x},{GRID[1]}={y}"] = {**s, "combination": combination}
            print(f"    {GRID[0]}={x:<7} {GRID[1]}={y:<5}  KC "
                  f"{s['kc_active_min']:6.2%}-{s['kc_active_max']:6.2%}  "
                  f"median {s['kc_active_median']:6.2%}"
                  f"{'  IN BAND' if s['in_band'] else ''}", flush=True)
    out["grid"] = grid
    out["bistability"] = bistability(seen)
    out["decisions"] = decisions(seen, controller.decoder.threshold)

    ranked = sorted(
        (v for v in grid.values() if v["in_band"]),
        key=lambda v: v["distance_from_target"],
    )[:3]
    out["confirmed"] = {}
    if ranked:
        names = list(SCENES)
        pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1:]]
        print(f"  phase 3: {len(ranked)} candidate(s) on all "
              f"{len(names)} market states", flush=True)
        for candidate in ranked:
            combination = candidate["combination"]
            with physiology(brain, kc, edges, combination):
                rows, code = kc_response(
                    controller, kc, apl, names, render, codes=True
                )
            jac = []
            for a, b in pairs:
                union = float((code[a] | code[b]).sum())
                jac.append(float((code[a] & code[b]).sum()) / union if union else 1.0)
            s = summarise(rows)
            key = f"{GRID[0]}={combination[GRID[0]]},{GRID[1]}={combination[GRID[1]]}"
            out["confirmed"][key] = {
                **s,
                "combination": combination,
                "kc_mean_jaccard": float(np.mean(jac)),
                "kc_identical_pairs": int(sum(j > 0.999 for j in jac)),
                "scene_pairs": len(pairs),
            }
            print(f"    {key}  KC {s['kc_active_min']:6.2%}-"
                  f"{s['kc_active_max']:6.2%}  identical codes "
                  f"{out['confirmed'][key]['kc_identical_pairs']}/{len(pairs)}  "
                  f"overlap {np.mean(jac):.2f}", flush=True)
    else:
        print("  phase 3 skipped: nothing reached the band", flush=True)

    for name, before, after in zip(
        ["ptr", "post", "weight"], guard, (brain.ptr, brain.post, brain.weight)
    ):
        if not np.array_equal(before, after):
            raise AssertionError(f"sparseness modified brain.{name}")
    out["graph_restored"] = True
    out["verdict"] = verdict_sparseness(out)
    return out


def verdict_sparseness(out):
    """Report the reachable range, not the number of combinations tried.

    A regime only counts if it is sparse AND still tells the market states
    apart. Sparseness reached by silencing the population is not sparseness,
    it is silence, and it would read as a success on the active fraction alone.
    """
    spans = []
    for lever, values in out["single_lever"].items():
        medians = [v["kc_active_median"] for v in values.values()]
        spans.append((max(medians) - min(medians), lever, min(medians), max(medians)))
    spans.sort(reverse=True)
    strongest = ", ".join(
        f"{lever} {lo:.2%}-{hi:.2%}" for _, lever, lo, hi in spans[:2]
    )
    b = out["bistability"]
    d = out["decisions"]
    sells = d["counts"].get("SELL", 0)
    sides = sum(1 for v in out["grid"].values() if v["distinct_sides"] > 1)
    # Reported alongside the sparseness answer because it is the same
    # measurement: moving these parameters unlocks the one-sided proposals the
    # input manipulations could not move, but only to within a spike or two of
    # the threshold, which is not the same as deciding anything.
    unlocked = (
        f". Separately: {sides} of {len(out['grid'])} grid points produce more "
        f"than one distinct decision and {sells} of {b['observations']} "
        f"observations propose SELL, where every input manipulation tested "
        f"before produced none -- but the widest SELL margin is "
        f"{d['margins'].get('SELL', {}).get('max_spikes_past_threshold', 0):.0f} "
        f"spikes past the threshold against "
        f"{d['margins'].get('BUY', {}).get('max_spikes_past_threshold', 0):.0f} "
        f"for BUY, so this is the readout coming off its rail, not the fly "
        f"changing its mind"
    )
    reached = [v for v in out["grid"].values() if v["in_band"]]
    if not reached:
        medians = [v["kc_active_median"] for v in out["grid"].values()]
        return (f"no declared free parameter reaches the {BAND[0]:.0%}-"
                f"{BAND[1]:.0%} band: across the whole {GRID[0]} x {GRID[1]} "
                f"grid the median Kenyon activity goes {min(medians):.2%} to "
                f"{max(medians):.2%} without stopping in between, and over all "
                f"{b['observations']} observations only "
                f"{b['inside_band_fraction']:.0%} land inside the band against "
                f"{b['below_band_fraction']:.0%} near-silent and "
                f"{b['above_20_percent_fraction']:.0%} above 20%. The "
                f"population is bistable, so sparseness cannot be set by gain "
                f"alone (strongest levers: {strongest})" + unlocked)
    usable = sorted(
        (v["kc_identical_pairs"], v["kc_mean_jaccard"], k)
        for k, v in out["confirmed"].items()
    )
    if not usable:
        return (f"{len(reached)} of {len(out['grid'])} grid points hold Kenyon "
                f"activity in band but none was confirmed on the full scene "
                f"set (strongest levers: {strongest})")
    collisions, overlap, key = usable[0]
    best = out["confirmed"][key]
    verdict = (f"{len(reached)} of {len(out['grid'])} grid points hold Kenyon "
               f"activity inside the {BAND[0]:.0%}-{BAND[1]:.0%} band; best is "
               f"{key} at {best['kc_active_median']:.2%} median "
               f"({best['kc_active_min']:.2%}-{best['kc_active_max']:.2%} "
               f"across {len(SCENES)} market states)")
    if collisions:
        return (verdict + f", but it still collapses {collisions}/"
                f"{best['scene_pairs']} state pairs onto an identical Kenyon "
                f"code, so it buys sparseness at the cost of separation "
                f"(strongest levers: {strongest})")
    return (verdict + f", with {collisions}/{best['scene_pairs']} identical "
            f"codes and {overlap:.2f} mean overlap -- sparse and still "
            f"separable. Strongest levers: {strongest}")


# --- olfaction -------------------------------------------------------------

# The neutral chart every odour-only observation sees. Holding the picture
# fixed is what makes the arm attributable: any difference between two market
# states then came through the antennal lobe and nowhere else.
NEUTRAL = "flat"


# Peak current per receptor neuron, in the same units as saturated retinal
# drive (30). Declared sweep ranges, not fitted values.
ODOR_CURRENTS = [1.0, 2.0, 5.0, 10.0, 20.0, 30.0]
ODOR_GRID_CURRENTS = [8.0, 10.0, 15.0, 20.0, 30.0]
ODOR_GRID_APL = [1.0, 2.0, 4.0, 8.0, 16.0]
# Above this, two market states share most of their active Kenyon cells and
# the code carries little about which state it is, however few pairs are
# byte-identical. Counting identical pairs alone rewards mere saturation:
# when 38% of the population fires, no two codes are ever identical and the
# metric reads as a success while the representation is useless.
USABLE_OVERLAP = 0.5


def usable(summary, states):
    """A code is usable only if it is sparse AND states stay far apart."""
    return (
        summary["inside_band"] == states
        and summary["kc_mean_jaccard"] <= USABLE_OVERLAP
    )


def code_summary(rows, code, pairs):
    """Sparseness and separability of one set of observations."""
    jac = []
    for a, b in pairs:
        union = float((code[a] | code[b]).sum())
        jac.append(float((code[a] & code[b]).sum()) / union if union else 1.0)
    active = [r["kc_active_fraction"] for r in rows.values()]
    return {
        "kc_identical_pairs": int(sum(j > 0.999 for j in jac)),
        "kc_mean_jaccard": float(np.mean(jac)),
        "kc_active_min": float(min(active)),
        "kc_active_max": float(max(active)),
        "kc_active_median": float(np.median(active)),
        "inside_band": int(sum(BAND[0] <= x <= BAND[1] for x in active)),
        "distinct_sides": len({r["side"] for r in rows.values()}),
        "scene_pairs": len(pairs),
    }


def olfaction(controller, superclass):
    """Does the odour channel give the mushroom body what vision could not?

    Vision reaches Kenyon cells only over a long indirect path and was
    measured delivering whole-field luminance: ten of twenty-eight market-state
    pairs shared a byte-identical Kenyon code, and no parameter regime held the
    population out of its two attractors. Olfactory receptors reach 3,829 of
    4,064 Kenyon cells in two synapses through the antennal lobe, which is the
    input the structure is built around.

    Three arms over the same eight market states: chart only (what shipped),
    odour only against a fixed neutral chart, and both together.
    """
    brain = controller.brain
    if controller.olfaction is None:
        raise RuntimeError("Controller built without the olfactory channel")
    kinds = annotations(brain.ids).type.fillna("").astype(str)
    kc = np.flatnonzero(kinds.str.startswith("KC").to_numpy())
    orn = controller.olfaction.indices
    render = RENDERERS[SWEEP_RENDERER]
    names = list(SCENES)
    pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1:]]
    neutral = render(scene_history(NEUTRAL))
    print(f"[olfaction] {len(orn):,} receptor neurons in "
          f"{len(controller.olfaction.names)} glomeruli; {len(names)} market "
          f"states x 3 arms", flush=True)

    out = {
        "scenes": names,
        "receptor_neurons": int(len(orn)),
        "glomeruli": len(controller.olfaction.names),
        "neutral_scene": NEUTRAL,
        "assignment": controller.olfaction.report["assignment"],
    }
    # Peak odour current is a declared free parameter, and the first
    # measurement put it far too high: 30 units drove Kenyon spikes from 16 to
    # 6,684, straight into the saturated attractor the visual pathway already
    # demonstrated. Sweep it rather than guess it.
    print("  phase 1: peak odour current, odour only", flush=True)
    default_current = controller.olfaction.current
    sweep = {}
    for current in ODOR_CURRENTS:
        controller.olfaction.current = float(current)
        rows, code = {}, {}
        for scene in names:
            n = observe(controller, neutral, history=scene_history(scene))
            counts = brain.counts
            code[scene] = counts[kc] > 0
            rows[scene] = {
                "kc_active_fraction": float((counts[kc] > 0).mean()),
                "kc_spikes": int(counts[kc].sum()),
                "orn_spikes": int(counts[orn].sum()),
                "side": n["side"],
                "difference_hz": float(n["difference_hz"]),
            }
        sweep[repr(float(current))] = {
            **code_summary(rows, code, pairs),
            "per_scene": rows,
        }
        summary = sweep[repr(float(current))]
        print(f"    current {current:5.1f}  ORN "
              f"{np.mean([r['orn_spikes'] for r in rows.values()]):8,.0f}  KC "
              f"{summary['kc_active_min']:7.3%}-{summary['kc_active_max']:7.3%}"
              f"  in band {summary['inside_band']}/{len(names)}  identical "
              f"{summary['kc_identical_pairs']}/{len(pairs)}  sides "
              f"{summary['distinct_sides']}", flush=True)
    out["current_sweep"] = sweep

    # The current sweep alone cannot win: receptor neurons are silent below
    # about 10 units and the Kenyon population saturates above it, which is
    # the same bistability the visual pathway showed. APL is the mushroom
    # body's own gain control and was measured to move Kenyon activity over
    # three orders of magnitude, so the two are crossed rather than tuned one
    # at a time.
    kinds_all = annotations(brain.ids).type.fillna("").astype(str)
    apl_cells = np.flatnonzero(kinds_all.str.fullmatch("APL").fillna(False).to_numpy())
    apl_edges = apl_kc_edges(brain, apl_cells, kc)
    guard = (brain.ptr.copy(), brain.post.copy(), brain.weight.copy())
    print(f"  phase 1b: odour current x APL gain over {len(apl_edges):,} "
          f"APL->KC edges", flush=True)
    grid = {}
    for current in ODOR_GRID_CURRENTS:
        for gain in ODOR_GRID_APL:
            controller.olfaction.current = float(current)
            combination = {**DEFAULTS, "apl_gain": gain}
            rows, code = {}, {}
            with physiology(brain, kc, apl_edges, combination):
                for scene in names:
                    n = observe(controller, neutral, history=scene_history(scene))
                    counts = brain.counts
                    code[scene] = counts[kc] > 0
                    rows[scene] = {
                        "kc_active_fraction": float((counts[kc] > 0).mean()),
                        "kc_spikes": int(counts[kc].sum()),
                        "orn_spikes": int(counts[orn].sum()),
                        "side": n["side"],
                        "difference_hz": float(n["difference_hz"]),
                    }
            summary = code_summary(rows, code, pairs)
            grid[f"current={current},apl_gain={gain}"] = {
                **summary,
                "per_scene": rows,
                "odor_current": float(current),
                "apl_gain": float(gain),
                "usable": usable(summary, len(names)),
            }
            print(f"    current {current:5.1f}  apl x{gain:<5.1f}  KC "
                  f"{summary['kc_active_min']:7.3%}-"
                  f"{summary['kc_active_max']:7.3%}  in band "
                  f"{summary['inside_band']}/{len(names)}  overlap "
                  f"{summary['kc_mean_jaccard']:.2f}  identical "
                  f"{summary['kc_identical_pairs']}/{len(pairs)}"
                  f"{'  USABLE' if usable(summary, len(names)) else ''}",
                  flush=True)
    for label, before, after in zip(
        ["ptr", "post", "weight"], guard, (brain.ptr, brain.post, brain.weight)
    ):
        if not np.array_equal(before, after):
            raise AssertionError(f"olfaction modified brain.{label}")
    out["current_apl_grid"] = grid
    out["graph_restored"] = True
    # Most states sparse first, then lowest overlap, then closest to target.
    # Identical-pair count deliberately does not appear: it rewards the
    # saturated regime, which is exactly the failure being measured.
    chosen = max(
        grid.values(),
        key=lambda v: (
            v["inside_band"],
            -v["kc_mean_jaccard"],
            -abs(v["kc_active_median"] - TARGET_ACTIVE),
        ),
    )
    controller.olfaction.current = chosen["odor_current"]
    out["selected_current"] = chosen["odor_current"]
    out["selected_apl_gain"] = chosen["apl_gain"]
    out["default_current"] = float(default_current)
    out["any_usable"] = any(v["usable"] for v in grid.values())
    print(f"  phase 2: three arms at current {chosen['odor_current']:g}, "
          f"APL gain x{chosen['apl_gain']:g}", flush=True)
    selected = {**DEFAULTS, "apl_gain": chosen["apl_gain"]}

    arms = {
        "chart_only": (lambda s: render(scene_history(s)), lambda s: None),
        "odor_only": (lambda s: neutral, scene_history),
        "both": (lambda s: render(scene_history(s)), scene_history),
    }
    for arm, (frame_of, history_of) in arms.items():
        rows, code, glom = {}, {}, {}
        for scene in names:
            history = history_of(scene)
            with physiology(brain, kc, apl_edges, selected):
                n = observe(controller, frame_of(scene), history=history)
                counts = brain.counts.copy()
            code[scene] = counts[kc] > 0
            if history is not None:
                glom[scene] = controller.olfaction.activation(
                    olfactory_features(history)
                ) > 0
            rows[scene] = {
                "kc_active_fraction": float((counts[kc] > 0).mean()),
                "kc_spikes": int(counts[kc].sum()),
                "orn_spikes": int(counts[orn].sum()),
                "side": n["side"],
                "difference_hz": float(n["difference_hz"]),
            }
            print(f"  {arm:11} {scene:11} ORN {rows[scene]['orn_spikes']:6,}  "
                  f"KC {rows[scene]['kc_active_fraction']:7.3%} "
                  f"({rows[scene]['kc_spikes']:5,} spikes)  "
                  f"{n['side']:4} R-L {n['difference_hz']:+6.2f}", flush=True)
        out[arm] = {**code_summary(rows, code, pairs), "per_scene": rows}
        if glom:
            gj = [
                float((glom[a] & glom[b]).sum())
                / max(1.0, float((glom[a] | glom[b]).sum()))
                for a, b in pairs
            ]
            out[arm]["glomerular_identical_pairs"] = int(sum(j > 0.999 for j in gj))
            out[arm]["glomerular_mean_jaccard"] = float(np.mean(gj))
    out["verdict"] = verdict_olfaction(out)
    return out


def verdict_olfaction(out):
    """Compare the arms on what blocks learning, not on how loud they are.

    An arm is better only if it leaves fewer market-state pairs sharing one
    Kenyon code. Extra spikes are not an improvement, and an odour that merely
    drives the population into its saturated attractor is the failure the
    visual pathway already demonstrated.
    """
    chart, odor, both = out["chart_only"], out["odor_only"], out["both"]
    pairs, states = chart["scene_pairs"], len(out["scenes"])
    steps = out["current_sweep"]
    silent = [c for c, v in steps.items()
              if max(r["orn_spikes"] for r in v["per_scene"].values()) == 0]
    step = (f"the receptor population is itself a step: it emits no spike at "
            f"all up to current {max(float(c) for c in silent):g} and "
            f"saturates the Kenyon population immediately above it"
            if silent else "the receptor population responds gradually")
    line = (f"the odour reaches the mushroom body -- "
            f"{odor['glomerular_identical_pairs']}/{pairs} identical "
            f"glomerular codes and {odor['kc_identical_pairs']}/{pairs} "
            f"identical Kenyon codes against {chart['kc_identical_pairs']}"
            f"/{pairs} for the chart -- but {step}")
    body = (f". At the best of {len(out['current_apl_grid'])} "
            f"current x APL-gain points (current "
            f"{out['selected_current']:g}, APL x{out['selected_apl_gain']:g}) "
            f"Kenyon activity is {odor['kc_active_min']:.2%}-"
            f"{odor['kc_active_max']:.2%} with {odor['inside_band']}/{states} "
            f"states in the {BAND[0]:.0%}-{BAND[1]:.0%} band and mean overlap "
            f"{odor['kc_mean_jaccard']:.2f}")
    if out["any_usable"]:
        return (line + body + f". That is a sparse code whose states stay "
                f"apart, so the antennal-lobe route does what the visual one "
                f"could not. Both channels together: "
                f"{both['kc_identical_pairs']}/{pairs} identical codes, "
                f"{both['inside_band']}/{states} in band, overlap "
                f"{both['kc_mean_jaccard']:.2f}, "
                f"{both['distinct_sides']} distinct decisions")
    reached = [v for v in out["current_apl_grid"].values()
               if v["inside_band"] == states]
    sparse = (f", the first regime measured in this network that is sparse "
              f"for every market state ({len(reached)} of "
              f"{len(out['current_apl_grid'])} grid points reach it)"
              if reached else "")
    return (line + body + sparse + f". No point is also separable (overlap at "
            f"or below {USABLE_OVERLAP:.1f}): the states share about "
            f"{odor['kc_mean_jaccard']:.0%} of their active Kenyon cells, so "
            f"what limits the code now is WHICH cells win, not how many. "
            f"Sparseness is reachable; decorrelation is the open problem")


# --- Kenyon input normalisation --------------------------------------------

# APL gain was measured to put Kenyon activity in the biological band, and the
# codes stayed 86% overlapped there: sparse, but nearly the same cells
# whatever the market is doing. The suspicion this tests is that the winners
# are decided by how much total input each Kenyon cell happens to have in the
# reconstruction, not by which glomeruli are active -- in which case inhibition
# selects the same intrinsically loudest cells every time.
NORMALISE_APL = [1.0, 2.0, 4.0]


def kenyon_inputs(brain, kc):
    """Edges arriving at Kenyon cells, and the positive drive each one gets."""
    mask = np.zeros(brain.n, dtype=bool)
    mask[kc] = True
    edges = np.flatnonzero(mask[brain.post]).astype(np.int64)
    targets = brain.post[edges].astype(np.int64)
    total = np.bincount(
        targets, weights=np.maximum(brain.weight[edges], 0.0), minlength=brain.n
    )
    return edges, targets, total


@contextlib.contextmanager
def normalised_inputs(brain, edges, targets, total, enabled=True):
    """Give every Kenyon cell the same summed excitatory input.

    Relative contributions inside one cell are untouched; only the per-cell
    scale changes, so no edge is added, removed or re-signed. Real Kenyon
    cells do normalise their input homeostatically, but the factor applied
    here is a declared model choice, not a measured one.
    """
    saved = brain.weight[edges].copy()
    try:
        if enabled:
            live = total[total > 0]
            reference = float(np.median(live)) if len(live) else 0.0
            scale = np.ones(brain.n, dtype=np.float32)
            hot = total > 0
            scale[hot] = (reference / total[hot]).astype(np.float32)
            brain.weight[edges] = saved * scale[targets]
        yield
    finally:
        brain.weight[edges] = saved


def kc_input(controller, superclass):
    """Does equalising Kenyon input drive decorrelate the codes?

    Odour only, chart held at the neutral scene, so any change is attributable
    to the mushroom body rather than to the picture. Read the overlap, not the
    count of byte-identical codes: at 38% active no two codes are ever
    identical and that metric reads as a success while the representation
    carries nothing.
    """
    brain = controller.brain
    if controller.olfaction is None:
        raise RuntimeError("Controller built without the olfactory channel")
    kinds = annotations(brain.ids).type.fillna("").astype(str)
    kc = np.flatnonzero(kinds.str.startswith("KC").to_numpy())
    apl = np.flatnonzero(kinds.str.fullmatch("APL").fillna(False).to_numpy())
    apl_edges = apl_kc_edges(brain, apl, kc)
    edges, targets, total = kenyon_inputs(brain, kc)
    render = RENDERERS[SWEEP_RENDERER]
    neutral = render(scene_history(NEUTRAL))
    names = list(SCENES)
    pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1:]]
    guard = (brain.ptr.copy(), brain.post.copy(), brain.weight.copy())
    live = total[total > 0]
    print(f"[kc_input] {len(edges):,} edges onto {len(kc):,} Kenyon cells; "
          f"summed excitatory input spans {live.min():.1f} to {live.max():.1f} "
          f"(median {np.median(live):.1f}, ratio "
          f"{live.max() / max(live.min(), 1e-6):,.0f}x)", flush=True)

    out = {
        "kenyon_cells": int(len(kc)),
        "input_edges": int(len(edges)),
        "input_sum_min": float(live.min()),
        "input_sum_max": float(live.max()),
        "input_sum_median": float(np.median(live)),
        "silent_kenyon_cells": int((total[kc] <= 0).sum()),
        "odor_current": float(controller.olfaction.current),
        "arms": {},
    }
    for enabled in [False, True]:
        for gain in NORMALISE_APL:
            rows, code = {}, {}
            with normalised_inputs(brain, edges, targets, total, enabled):
                with physiology(brain, kc, apl_edges, {**DEFAULTS, "apl_gain": gain}):
                    for scene in names:
                        n = observe(controller, neutral,
                                    history=scene_history(scene))
                        counts = brain.counts
                        code[scene] = counts[kc] > 0
                        rows[scene] = {
                            "kc_active_fraction": float((counts[kc] > 0).mean()),
                            "kc_spikes": int(counts[kc].sum()),
                            "side": n["side"],
                            "difference_hz": float(n["difference_hz"]),
                        }
            summary = code_summary(rows, code, pairs)
            key = f"{'normalised' if enabled else 'as_reconstructed'},apl_gain={gain}"
            out["arms"][key] = {
                **summary,
                "per_scene": rows,
                "normalised": enabled,
                "apl_gain": gain,
                "usable": usable(summary, len(names)),
            }
            print(f"    {'normalised' if enabled else 'reconstructed':14} "
                  f"apl x{gain:<5.1f} KC {summary['kc_active_min']:7.3%}-"
                  f"{summary['kc_active_max']:7.3%}  in band "
                  f"{summary['inside_band']}/{len(names)}  overlap "
                  f"{summary['kc_mean_jaccard']:.2f}"
                  f"{'  USABLE' if usable(summary, len(names)) else ''}",
                  flush=True)
    for label, before, after in zip(
        ["ptr", "post", "weight"], guard, (brain.ptr, brain.post, brain.weight)
    ):
        if not np.array_equal(before, after):
            raise AssertionError(f"kc_input modified brain.{label}")
    out["graph_restored"] = True
    out["verdict"] = verdict_kc_input(out)
    return out


def verdict_kc_input(out):
    """Compare like with like: same APL gain, normalisation on against off."""
    gains = sorted({v["apl_gain"] for v in out["arms"].values()})
    moves = []
    for gain in gains:
        a = out["arms"][f"as_reconstructed,apl_gain={gain}"]
        b = out["arms"][f"normalised,apl_gain={gain}"]
        moves.append((gain, a["kc_mean_jaccard"], b["kc_mean_jaccard"],
                      a["inside_band"], b["inside_band"]))
    spread = (f"summed excitatory input onto a Kenyon cell spans "
              f"{out['input_sum_min']:.1f} to {out['input_sum_max']:.1f} in the "
              f"reconstruction, a factor of "
              f"{out['input_sum_max'] / max(out['input_sum_min'], 1e-6):,.0f}")
    table = "; ".join(
        f"APL x{g:g}: overlap {before:.2f} -> {after:.2f}, in band {ib}->{jb}"
        for g, before, after, ib, jb in moves
    )
    best = min(
        (v for v in out["arms"].values() if v["usable"]),
        key=lambda v: v["kc_mean_jaccard"],
        default=None,
    )
    if best is not None:
        return (f"{spread}. Equalising it gives a code that is sparse and "
                f"separable at APL x{best['apl_gain']:g}: "
                f"{best['kc_active_min']:.2%}-{best['kc_active_max']:.2%} "
                f"active, overlap {best['kc_mean_jaccard']:.2f} ({table})")
    gained = sum(1 for _, before, after, _, _ in moves if after < before - 0.02)
    return (f"{spread}. Equalising it lowers the overlap at {gained} of "
            f"{len(moves)} gains but never far enough to be usable "
            f"(sparse in all {len(out['arms'][list(out['arms'])[0]]['per_scene'])} "
            f"states with overlap at or below {USABLE_OVERLAP:.1f}): {table}. "
            f"The Kenyon population is not selecting on which glomeruli are "
            f"active, so the loudest-cell hypothesis does not by itself "
            f"explain the shared code")


# --- decoder candidates ----------------------------------------------------

# The shipped readout takes one DNp20 cell per side over a 500 ms window, so a
# single spike is worth 2 Hz and the threshold is 2 Hz. Ablation measured the
# standard deviation of that quantity at 2.94 Hz when 1% of the network is
# silenced and 27 Hz at 20%. The question here is not whether a wider readout
# fires more, but whether any candidate has a spread across market states that
# is large compared with its own noise.
DECODER_BINS = 10
DECODER_ABLATION = [1.0, 5.0]
DECODER_REPEATS = 15


def decoder_populations(brain):
    """Left/right readout candidates, taken from annotations only.

    No population is selected for how it behaves. Each is a whole anatomical
    class split by soma side, so nothing here is a search for buy neurons.
    """
    a = annotations(brain.ids)
    kinds = a.type.fillna("").astype(str)
    sides = a.somaSide.fillna("").astype(str).to_numpy()
    superclass = a.superclass.fillna("").astype(str)
    # Exact class names, not substrings: "descending_neuron_tbc" and
    # "sensory_descending" are different populations and are left out.
    dn = superclass.eq("descending_neuron").to_numpy()
    motor = superclass.isin(["vnc_motor", "cb_motor"]).to_numpy()
    dnp20 = kinds.eq("DNp20").to_numpy()
    out = {}
    for name, mask in [
        ("dnp20_shipped", dnp20),
        ("descending", dn),
        ("descending_and_motor", dn | motor),
    ]:
        left = np.flatnonzero(mask & (sides == "L")).astype(np.int32)
        right = np.flatnonzero(mask & (sides == "R")).astype(np.int32)
        if len(left) and len(right):
            out[name] = (left, right)
    return out


def binned_counts(controller, frame, history=None, bins=DECODER_BINS):
    """One observation, returned as a sequence of per-bin spike counts.

    The shipped decoder sums the whole window before comparing sides, which
    throws away whether the difference held for the window or arrived in one
    burst. Keeping the bins is what makes an integrating readout measurable.
    """
    b = controller.brain
    b.reset()
    odor = None
    if controller.olfaction is not None and history is not None:
        odor, _ = controller.olfaction.stimulation(history)
    total_steps = round(controller.s.neural_ms / b.dt)
    per_bin = total_steps // bins
    if per_bin < 1:
        raise ValueError("Observation is too short to split into bins")
    chunks = []
    for _ in range(bins):
        remaining = per_bin
        acc = np.zeros(b.n, dtype=np.int64)
        while remaining:
            n = min(remaining, round(controller.s.neural_bin_ms / b.dt))
            c, _ = b.rgb_step(
                frame, n * b.dt, learning=False,
                stimulation=None if odor is None else [odor],
            )
            acc += c
            remaining -= n
        chunks.append(acc)
    return chunks, per_bin * b.dt / 1000.0


def decoder_statistics(chunks, seconds, left, right):
    """Three readings of the same spikes, from the same population.

    mean_hz is what ships: sum the window, compare side means. consistency
    and t_score integrate instead -- they ask whether the difference held
    across the window, which is the property a single spike cannot fake.
    """
    d = np.array(
        [
            float(np.mean(c[right]) / seconds - np.mean(c[left]) / seconds)
            for c in chunks
        ]
    )
    mean = float(d.mean())
    sd = float(d.std(ddof=1)) if len(d) > 1 else 0.0
    sign = np.sign(mean)
    return {
        "mean_hz": mean,
        "consistency": float(np.mean(np.sign(d) == sign)) if sign else 0.0,
        "t_score": float(mean / (sd / math.sqrt(len(d)))) if sd > 0 else 0.0,
        "spiking_bins": int(np.count_nonzero(d)),
    }


def decoder(controller, superclass):
    """Which readout has a market signal larger than its own noise?

    Two measurements over the same candidates. Spread: how far apart the
    readings are across eight market states, which is the signal. Noise: how
    far the reading moves when a random percentage of the network is silenced
    and the market is held fixed. A readout is only worth widening to if the
    first is large compared with the second.
    """
    brain = controller.brain
    populations = decoder_populations(brain)
    protected = np.unique(
        np.r_[
            controller.decoder.left, controller.decoder.right, controller.decoder.gate,
            np.concatenate([np.r_[l, r] for l, r in populations.values()]),
        ]
    ).astype(np.int32)
    render = RENDERERS[SWEEP_RENDERER]
    names = list(SCENES)
    guard = (brain.ptr.copy(), brain.post.copy(), brain.weight.copy())
    print(f"[decoder] {len(populations)} candidates over {DECODER_BINS} bins; "
          + ", ".join(f"{k} {len(l)}L/{len(r)}R"
                      for k, (l, r) in populations.items()), flush=True)

    out = {
        "bins": DECODER_BINS,
        "threshold_hz": float(controller.decoder.threshold),
        "populations": {
            k: {"left": int(len(l)), "right": int(len(r))}
            for k, (l, r) in populations.items()
        },
        "across_states": {},
        "under_ablation": {},
    }

    print("  spread across market states", flush=True)
    readings = {k: [] for k in populations}
    for scene in names:
        chunks, seconds = binned_counts(
            controller, render(scene_history(scene)), scene_history(scene)
        )
        for name, (left, right) in populations.items():
            readings[name].append(decoder_statistics(chunks, seconds, left, right))
    for name in populations:
        rows = readings[name]
        out["across_states"][name] = {
            "per_scene": dict(zip(names, rows)),
            **{
                f"{k}_spread": float(np.std([r[k] for r in rows], ddof=1))
                for k in ["mean_hz", "consistency", "t_score"]
            },
        }
        print(f"    {name:22} mean_hz spread "
              f"{out['across_states'][name]['mean_hz_spread']:8.3f}  t_score "
              f"spread {out['across_states'][name]['t_score_spread']:8.3f}",
              flush=True)

    print("  noise under ablation, market held fixed", flush=True)
    frame = render(scene_history("rally"))
    history = scene_history("rally")
    rng = np.random.default_rng(20260914)
    eligible = np.setdiff1d(np.arange(brain.n, dtype=np.int32), protected)
    noise = {k: {} for k in populations}
    for level in DECODER_ABLATION:
        repeats = {k: [] for k in populations}
        for _ in range(DECODER_REPEATS):
            victims = rng.choice(
                eligible, size=int(len(eligible) * level / 100), replace=False
            )
            saved = brain.tonic[victims].copy()
            brain.tonic[victims] = -1000.0
            try:
                chunks, seconds = binned_counts(controller, frame, history)
            finally:
                brain.tonic[victims] = saved
            for name, (left, right) in populations.items():
                repeats[name].append(decoder_statistics(chunks, seconds, left, right))
        for name in populations:
            noise[name][f"{level:g}%"] = {
                f"{k}_sd": float(np.std([r[k] for r in repeats[name]], ddof=1))
                for k in ["mean_hz", "consistency", "t_score"]
            }
    out["under_ablation"] = noise
    for name in populations:
        line = "  ".join(
            f"{lvl} mean_hz sd {v['mean_hz_sd']:7.3f} t_score sd {v['t_score_sd']:7.3f}"
            for lvl, v in noise[name].items()
        )
        print(f"    {name:22} {line}", flush=True)

    for label, before, after in zip(
        ["ptr", "post", "weight"], guard, (brain.ptr, brain.post, brain.weight)
    ):
        if not np.array_equal(before, after):
            raise AssertionError(f"decoder modified brain.{label}")
    out["graph_restored"] = True
    out["signal_to_noise"] = {
        name: {
            statistic: float(
                out["across_states"][name][f"{statistic}_spread"]
                / max(noise[name][f"{DECODER_ABLATION[-1]:g}%"][f"{statistic}_sd"], 1e-9)
            )
            for statistic in ["mean_hz", "t_score"]
        }
        for name in populations
    }
    out["verdict"] = verdict_decoder(out)
    return out


def verdict_decoder(out):
    """Rank on signal against noise, never on how much a candidate fires.

    A wider population produces larger numbers and a smoother curve while
    telling you nothing more; the only thing that matters is whether the
    variation caused by the market is bigger than the variation caused by
    silencing neurons that have nothing to do with the market.
    """
    ranked = sorted(
        out["signal_to_noise"].items(),
        key=lambda kv: -max(kv[1].values()),
    )
    shipped = out["signal_to_noise"]["dnp20_shipped"]
    best_name, best = ranked[0]
    statistic = max(best, key=best.get)
    table = "; ".join(
        f"{name} {max(v.values()):.2f}" for name, v in ranked
    )
    head = (f"signal-to-noise, spread across eight market states over the "
            f"standard deviation under {DECODER_ABLATION[-1]:g}% ablation: "
            f"{table}")
    if max(best.values()) < 1.0:
        return (head + f". Every candidate is below 1, so for all of them the "
                f"market moves the readout less than silencing unrelated "
                f"neurons does. Widening the readout does not fix that, and a "
                f"decoder is not the missing piece yet")
    if best_name == "dnp20_shipped":
        return (head + f". The shipped two-cell readout is already the best of "
                f"these, so widening the population is not the improvement")
    return (head + f". '{best_name}' on '{statistic}' is the only candidate "
            f"whose market signal exceeds its ablation noise "
            f"({max(best.values()):.2f} against "
            f"{max(shipped.values()):.2f} for the shipped readout)")


# --- odour tuning width ----------------------------------------------------

# The odour channel reached a sparse Kenyon code and left the market states
# sharing about three quarters of their active cells. Before blaming the
# mushroom body, check the input: a wide tuning curve puts seventeen of the
# fifty-three glomeruli above threshold for every state, and states that share
# a third of their glomeruli cannot be expected to own separate Kenyon codes.
# Sharpening costs resolution inside a band, so it is a trade, not a free win.
TUNINGS = [
    (0.9, 0.1),
    (0.6, 0.3),
    (0.5, 0.5),
    (0.3, 0.3),
]
TUNING_APL = 2.0


def odor_tuning(controller, superclass):
    """How sharp does the odour have to be before the Kenyon codes separate?"""
    brain = controller.brain
    if controller.olfaction is None:
        raise RuntimeError("Controller built without the olfactory channel")
    kinds = annotations(brain.ids).type.fillna("").astype(str)
    kc = np.flatnonzero(kinds.str.startswith("KC").to_numpy())
    apl = np.flatnonzero(kinds.str.fullmatch("APL").fillna(False).to_numpy())
    apl_edges = apl_kc_edges(brain, apl, kc)
    render = RENDERERS[SWEEP_RENDERER]
    neutral = render(scene_history(NEUTRAL))
    names = list(SCENES)
    pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1:]]
    saved = (controller.olfaction.sigma, controller.olfaction.floor)
    guard = (brain.ptr.copy(), brain.post.copy(), brain.weight.copy())
    print(f"[odor_tuning] {len(TUNINGS)} tuning widths at APL x{TUNING_APL:g}, "
          f"odour only", flush=True)

    out = {"apl_gain": TUNING_APL, "shipped": list(saved), "arms": {}}
    try:
        for sigma, floor in TUNINGS:
            controller.olfaction.sigma = float(sigma)
            controller.olfaction.floor = float(floor)
            rows, code, glom = {}, {}, {}
            with physiology(brain, kc, apl_edges,
                            {**DEFAULTS, "apl_gain": TUNING_APL}):
                for scene in names:
                    history = scene_history(scene)
                    n = observe(controller, neutral, history=history)
                    counts = brain.counts
                    code[scene] = counts[kc] > 0
                    glom[scene] = controller.olfaction.activation(
                        {**olfactory_features(history), "executed_trade": 0.5}
                    ) > 0
                    rows[scene] = {
                        "kc_active_fraction": float((counts[kc] > 0).mean()),
                        "kc_spikes": int(counts[kc].sum()),
                        "glomeruli_active": int(glom[scene].sum()),
                        "side": n["side"],
                        "difference_hz": float(n["difference_hz"]),
                    }
            summary = code_summary(rows, code, pairs)
            gj = [
                float((glom[a] & glom[b]).sum())
                / max(1.0, float((glom[a] | glom[b]).sum()))
                for a, b in pairs
            ]
            key = f"sigma={sigma},floor={floor}"
            out["arms"][key] = {
                **summary,
                "per_scene": rows,
                "sigma": float(sigma),
                "floor": float(floor),
                "glomerular_mean_jaccard": float(np.mean(gj)),
                "glomeruli_per_state": float(
                    np.mean([r["glomeruli_active"] for r in rows.values()])
                ),
                "usable": usable(summary, len(names)),
            }
            a = out["arms"][key]
            print(f"    sigma {sigma:4.1f} floor {floor:4.1f}  glomeruli "
                  f"{a['glomeruli_per_state']:5.1f}/53 overlap "
                  f"{a['glomerular_mean_jaccard']:.2f}  ->  KC "
                  f"{summary['kc_active_min']:7.3%}-"
                  f"{summary['kc_active_max']:7.3%}  in band "
                  f"{summary['inside_band']}/{len(names)}  overlap "
                  f"{summary['kc_mean_jaccard']:.2f}"
                  f"{'  USABLE' if a['usable'] else ''}", flush=True)
    finally:
        controller.olfaction.sigma, controller.olfaction.floor = saved
    for label, before, after in zip(
        ["ptr", "post", "weight"], guard, (brain.ptr, brain.post, brain.weight)
    ):
        if not np.array_equal(before, after):
            raise AssertionError(f"odor_tuning modified brain.{label}")
    out["graph_restored"] = True
    out["verdict"] = verdict_odor_tuning(out)
    return out


def verdict_odor_tuning(out):
    """Report how far the input overlap carries into the Kenyon overlap.

    If halving the glomerular overlap barely moves the Kenyon overlap, the
    shared code is made in the mushroom body and sharpening the input is the
    wrong lever however good the receptor numbers look.
    """
    arms = list(out["arms"].values())
    lo = min(arms, key=lambda v: v["glomerular_mean_jaccard"])
    hi = max(arms, key=lambda v: v["glomerular_mean_jaccard"])
    table = "; ".join(
        f"sigma {v['sigma']:g}/floor {v['floor']:g}: "
        f"{v['glomeruli_per_state']:.1f} glomeruli, input overlap "
        f"{v['glomerular_mean_jaccard']:.2f}, Kenyon overlap "
        f"{v['kc_mean_jaccard']:.2f}, in band {v['inside_band']}"
        for v in arms
    )
    transfer = (hi["kc_mean_jaccard"] - lo["kc_mean_jaccard"]) / max(
        hi["glomerular_mean_jaccard"] - lo["glomerular_mean_jaccard"], 1e-9
    )
    usable_arms = [v for v in arms if v["usable"]]
    head = (f"sharpening the tuning curve takes the input overlap from "
            f"{hi['glomerular_mean_jaccard']:.2f} to "
            f"{lo['glomerular_mean_jaccard']:.2f} and the Kenyon overlap from "
            f"{hi['kc_mean_jaccard']:.2f} to {lo['kc_mean_jaccard']:.2f}, a "
            f"transfer of {transfer:.2f} ({table})")
    if usable_arms:
        best = min(usable_arms, key=lambda v: v["kc_mean_jaccard"])
        return (head + f". sigma {best['sigma']:g}/floor {best['floor']:g} is "
                f"sparse in all states and separable at overlap "
                f"{best['kc_mean_jaccard']:.2f}")
    return (head + f". No width is both sparse in every state and below "
            f"{USABLE_OVERLAP:.1f} overlap. A transfer well under 1 means the "
            f"shared Kenyon code is made in the mushroom body, not inherited "
            f"from the receptors, so sharpening the input further is the "
            f"wrong lever")


# --- where the market signal is lost ---------------------------------------

# Two hypotheses for the shared Kenyon code have now failed: equalising the
# summed input onto each Kenyon cell changed the overlap by 0.00, and halving
# the overlap of the odour itself changed it by -0.03. So the information is
# lost somewhere between the receptors and the mushroom body. This walks the
# path one synapse at a time and reports where the states stop being
# different, instead of guessing which layer to blame next.
PATHWAY_APL = 2.0


def pathway_layers(brain, orn, kc):
    """Successive synaptic layers out from the olfactory receptors."""
    seen = np.zeros(brain.n, dtype=bool)
    seen[orn] = True
    layers = [("receptors", np.asarray(orn, dtype=np.int32))]
    front = np.asarray(orn, dtype=np.int32)
    for hop in (1, 2):
        if not len(front):
            break
        nxt = np.unique(
            np.concatenate(
                [brain.post[brain.ptr[i]:brain.ptr[i + 1]] for i in front]
            )
        )
        nxt = nxt[~seen[nxt]].astype(np.int32)
        seen[nxt] = True
        front = nxt
        layers.append((f"hop{hop}", nxt))
    layers.append(("kenyon", np.asarray(kc, dtype=np.int32)))
    return layers


def separation_of(counts, indices, pairs, names):
    """How different two market states look to one population.

    Jaccard reads which cells fired, correlation reads how much. A population
    can keep the second while losing the first, and the mushroom body only
    ever sees the first.
    """
    jac, corr = [], []
    for a, b in pairs:
        x, y = counts[a][indices].astype(float), counts[b][indices].astype(float)
        union = float(((x > 0) | (y > 0)).sum())
        jac.append(float(((x > 0) & (y > 0)).sum()) / union if union else 1.0)
        if x.std() > 0 and y.std() > 0:
            corr.append(float(np.corrcoef(x, y)[0, 1]))
    active = [float((counts[s][indices] > 0).mean()) for s in names]
    return {
        "cells": int(len(indices)),
        "active_fraction_median": float(np.median(active)),
        "mean_jaccard": float(np.mean(jac)),
        "mean_correlation": float(np.mean(corr)) if corr else 1.0,
    }


def pathway(controller, superclass):
    """Layer by layer from the receptors, where do the market states converge?"""
    brain = controller.brain
    if controller.olfaction is None:
        raise RuntimeError("Controller built without the olfactory channel")
    kinds = annotations(brain.ids).type.fillna("").astype(str)
    kc = np.flatnonzero(kinds.str.startswith("KC").to_numpy())
    apl = np.flatnonzero(kinds.str.fullmatch("APL").fillna(False).to_numpy())
    apl_edges = apl_kc_edges(brain, apl, kc)
    orn = controller.olfaction.indices
    layers = pathway_layers(brain, orn, kc)
    render = RENDERERS[SWEEP_RENDERER]
    neutral = render(scene_history(NEUTRAL))
    names = list(SCENES)
    pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1:]]
    guard = (brain.ptr.copy(), brain.post.copy(), brain.weight.copy())
    print(f"[pathway] {' -> '.join(f'{n} ({len(ix):,})' for n, ix in layers)} "
          f"at APL x{PATHWAY_APL:g}", flush=True)

    counts = {}
    with physiology(brain, kc, apl_edges, {**DEFAULTS, "apl_gain": PATHWAY_APL}):
        for scene in names:
            observe(controller, neutral, history=scene_history(scene))
            counts[scene] = brain.counts.copy()
    out = {"apl_gain": PATHWAY_APL, "scenes": names, "layers": {}}
    # The glomerular code is the input, before any spike: it is the ceiling
    # every later layer is measured against.
    glom = {
        s: controller.olfaction.activation(
            {**olfactory_features(scene_history(s)), "executed_trade": 0.5}
        ) > 0
        for s in names
    }
    gj = [
        float((glom[a] & glom[b]).sum()) / max(1.0, float((glom[a] | glom[b]).sum()))
        for a, b in pairs
    ]
    out["glomerular_mean_jaccard"] = float(np.mean(gj))
    print(f"    {'glomerular code':16} {len(controller.olfaction.names):7,} "
          f"channels                overlap {np.mean(gj):.2f}", flush=True)
    for name, indices in layers:
        row = separation_of(counts, indices, pairs, names)
        out["layers"][name] = row
        print(f"    {name:16} {row['cells']:7,} cells  active "
              f"{row['active_fraction_median']:7.3%}  overlap "
              f"{row['mean_jaccard']:.2f}  rate correlation "
              f"{row['mean_correlation']:.3f}", flush=True)
    for label, before, after in zip(
        ["ptr", "post", "weight"], guard, (brain.ptr, brain.post, brain.weight)
    ):
        if not np.array_equal(before, after):
            raise AssertionError(f"pathway modified brain.{label}")
    out["graph_restored"] = True
    out["verdict"] = verdict_pathway(out)
    return out


def verdict_pathway(out):
    """Name the synapse where the states stop being different."""
    order = list(out["layers"])
    steps = [(out["glomerular_mean_jaccard"], "glomerular code")] + [
        (out["layers"][k]["mean_jaccard"], k) for k in order
    ]
    jumps = [
        (steps[i + 1][0] - steps[i][0], steps[i][1], steps[i + 1][1])
        for i in range(len(steps) - 1)
    ]
    worst = max(jumps, key=lambda j: j[0])
    trail = " -> ".join(f"{name} {value:.2f}" for value, name in steps)
    if worst[0] <= 0.05:
        return (f"overlap along the path: {trail}. No single synapse converges "
                f"the states; the code is diluted gradually rather than lost "
                f"at one place")
    return (f"overlap along the path: {trail}. The convergence happens at "
            f"{worst[1]} -> {worst[2]}, which adds {worst[0]:+.2f} on its own. "
            f"That synapse is where the market signal stops being a difference "
            f"between states, and it is the only place worth changing")


# --- global inhibitory gain -------------------------------------------------

# Every failure measured so far has the same shape: a population that is
# either silent or saturated, with no graded middle. The antennal lobe is the
# clearest case -- 94% of the layer one synapse from the receptors fires for
# every market state, taking the overlap from 0.61 to 0.98 in a single step.
#
# The model assigns excitation and inhibition from the same quantity: contact
# count times 0.275, with the sign taken from the transmitter annotation. Real
# circuits do not balance that way; inhibition is relatively much stronger,
# and it is what keeps a population in range. So rather than patch each layer
# with its own gain, test the one parameter that could explain all of them.
INHIBITORY_GAINS = [1.0, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.5, 3.0]


@contextlib.contextmanager
def inhibitory_gain(brain, negative, saved, gain):
    """Scale every inhibitory weight in the graph, then restore it exactly.

    Sign, wiring and relative magnitudes within the inhibitory set are all
    untouched; only the excitation/inhibition ratio moves. That ratio is a
    declared model choice -- the 0.275 factor is applied to both -- and not a
    measured property of the reconstruction.
    """
    try:
        brain.weight[negative] = saved * gain
        yield
    finally:
        brain.weight[negative] = saved


def inhibition(controller, superclass):
    """Does one global excitation/inhibition ratio explain every saturation?"""
    brain = controller.brain
    if controller.olfaction is None:
        raise RuntimeError("Controller built without the olfactory channel")
    kinds = annotations(brain.ids).type.fillna("").astype(str)
    kc = np.flatnonzero(kinds.str.startswith("KC").to_numpy())
    orn = controller.olfaction.indices
    layers = pathway_layers(brain, orn, kc)
    render = RENDERERS[SWEEP_RENDERER]
    neutral = render(scene_history(NEUTRAL))
    names = list(SCENES)
    pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1:]]
    negative = np.flatnonzero(brain.weight < 0).astype(np.int64)
    saved = brain.weight[negative].copy()
    guard = (brain.ptr.copy(), brain.post.copy(), brain.weight.copy())
    print(f"[inhibition] {len(negative):,} inhibitory of {len(brain.weight):,} "
          f"edges ({len(negative) / len(brain.weight):.1%})", flush=True)

    out = {
        "states": len(names),
        "inhibitory_edges": int(len(negative)),
        "inhibitory_fraction": float(len(negative) / len(brain.weight)),
        "odor_current": float(controller.olfaction.current),
        "arms": {},
    }
    for gain in INHIBITORY_GAINS:
        counts = {}
        with inhibitory_gain(brain, negative, saved, gain):
            for scene in names:
                observe(controller, neutral, history=scene_history(scene))
                counts[scene] = brain.counts.copy()
        rows = {
            name: separation_of(counts, indices, pairs, names)
            for name, indices in layers
        }
        active = [float((counts[s][kc] > 0).mean()) for s in names]
        out["arms"][repr(gain)] = {
            "layers": rows,
            "kc_active_min": float(min(active)),
            "kc_active_max": float(max(active)),
            "kc_active_median": float(np.median(active)),
            "inside_band": int(sum(BAND[0] <= x <= BAND[1] for x in active)),
            "whole_brain_spikes": float(
                np.mean([int(counts[s].sum()) for s in names])
            ),
        }
        print(f"    gain x{gain:<4g} " + "  ".join(
            f"{name} {rows[name]['mean_jaccard']:.2f}" for name, _ in layers
        ) + f"   KC {min(active):7.3%}-{max(active):7.3%}  in band "
            f"{out['arms'][repr(gain)]['inside_band']}/{len(names)}  spikes "
            f"{out['arms'][repr(gain)]['whole_brain_spikes']:,.0f}", flush=True)

    for label, before, after in zip(
        ["ptr", "post", "weight"], guard, (brain.ptr, brain.post, brain.weight)
    ):
        if not np.array_equal(before, after):
            raise AssertionError(f"inhibition modified brain.{label}")
    out["graph_restored"] = True
    out["verdict"] = verdict_inhibition(out)
    return out


def verdict_inhibition(out):
    """Judge on the first synapse and on the Kenyon code, in that order.

    The antennal lobe is where the states were measured converging, so if a
    gain does not open that layer, nothing downstream of it can recover.
    """
    arms = out["arms"]
    first = {g: v["layers"]["hop1"]["mean_jaccard"] for g, v in arms.items()}
    ken = {g: v["layers"]["kenyon"]["mean_jaccard"] for g, v in arms.items()}
    base = repr(INHIBITORY_GAINS[0])
    table = "; ".join(
        f"x{float(g):g}: antennal lobe {first[g]:.2f}, Kenyon {ken[g]:.2f}, "
        f"in band {arms[g]['inside_band']}"
        for g in arms
    )
    best = min(arms, key=lambda g: (ken[g], first[g]))
    moved = first[base] - first[best]
    head = (f"scaling all {out['inhibitory_edges']:,} inhibitory edges "
            f"({out['inhibitory_fraction']:.0%} of the graph): {table}")
    if moved <= 0.02:
        return (head + f". No gain opens the antennal lobe -- it stays at "
                f"{first[base]:.2f} overlap whatever the inhibition, so the "
                f"saturation there is not an excitation/inhibition balance "
                f"problem")
    usable_gain = [
        g for g in arms
        if ken[g] <= USABLE_OVERLAP and arms[g]["inside_band"] == out["states"]
    ]
    if usable_gain:
        g = min(usable_gain, key=lambda g: ken[g])
        return (head + f". x{float(g):g} gives a Kenyon code that is sparse in "
                f"every state and separable at overlap {ken[g]:.2f}, against "
                f"{ken[base]:.2f} as reconstructed")
    return (head + f". x{float(best):g} is the best of these -- antennal-lobe "
            f"overlap {first[base]:.2f} -> {first[best]:.2f}, Kenyon "
            f"{ken[base]:.2f} -> {ken[best]:.2f} -- but no gain reaches a code "
            f"that is both sparse in every state and below "
            f"{USABLE_OVERLAP:.1f} overlap")


# --- reinforcement pulse calibration ---------------------------------------

# The reward and aversive pulses are engineered currents into identified cells,
# and the same amplitude goes to both compartments. They are not the same size:
# PAM11 has 15 cells and PPL101 has 2. At the amplitude the experiment shipped
# with, the reward compartment received roughly fifteen times the drive PER
# CELL that the aversive one did, which is an accident of population size
# rather than a modeling decision anyone made.
PULSE_CURRENTS = [20.0, 30.0, 40.0, 60.0, 80.0, 120.0]
# Declared before the sweep: the smallest current at which the two identified
# dopamine populations are driven within this factor of each other per cell.
PULSE_BALANCE = 2.0


def pulse(controller, superclass):
    """How much current does each dopamine compartment actually need?"""
    brain = controller.brain
    render = RENDERERS[SWEEP_RENDERER]
    frame = render(scene_history("rally"))
    history = scene_history("rally")
    reward, aversive = brain.circuit["reward"], brain.circuit["aversive"]
    saved = controller.s.pulse_current
    print(f"[pulse] {len(reward)} reward cells, {len(aversive)} aversive cells; "
          f"shipped amplitude {saved:g}", flush=True)
    out = {
        "reward_cells": int(len(reward)),
        "aversive_cells": int(len(aversive)),
        "shipped_current": float(saved),
        "balance_factor": PULSE_BALANCE,
        "arms": {},
    }
    try:
        for current in PULSE_CURRENTS:
            controller.s = dataclasses.replace(controller.s, pulse_current=current)
            row = {}
            for kind, cells in [("reward", reward), ("aversive", aversive)]:
                n = observe(controller, frame, kind, history=history)
                row[kind] = {
                    "spikes": int(n[f"{kind}_spikes"]),
                    "per_cell": float(n[f"{kind}_spikes"] / len(cells)),
                    "changed_edges": int(n["memory"]["changed_edges"]),
                    "kc_spikes": int(n["KC_spikes"]),
                }
            ratio = row["reward"]["per_cell"] / max(row["aversive"]["per_cell"], 1e-9)
            row["per_cell_ratio"] = float(ratio)
            row["balanced"] = bool(
                row["aversive"]["per_cell"] > 0
                and 1 / PULSE_BALANCE <= ratio <= PULSE_BALANCE
            )
            out["arms"][repr(current)] = row
            print(f"    current {current:6.0f}  reward "
                  f"{row['reward']['spikes']:5,} ({row['reward']['per_cell']:6.1f}"
                  f"/cell)  aversive {row['aversive']['spikes']:4,} "
                  f"({row['aversive']['per_cell']:6.1f}/cell)  ratio "
                  f"{ratio:6.1f}{'  BALANCED' if row['balanced'] else ''}",
                  flush=True)
    finally:
        controller.s = dataclasses.replace(controller.s, pulse_current=saved)
    out["verdict"] = verdict_pulse(out)
    return out


def verdict_pulse(out):
    """Report per-cell drive, never total spikes.

    Totals make the fifteen-cell reward compartment look stronger than the
    two-cell aversive one for a reason that has nothing to do with dopamine.
    """
    balanced = [c for c, v in out["arms"].items() if v["balanced"]]
    shipped = out["arms"].get(repr(out["shipped_current"]))
    table = "; ".join(
        f"{float(c):g}: {v['per_cell_ratio']:.1f}x" for c, v in out["arms"].items()
    )
    head = (f"the same amplitude reaches {out['reward_cells']} reward cells and "
            f"{out['aversive_cells']} aversive ones, so per-cell drive is "
            f"unequal: reward/aversive ratio by current -- {table}")
    if shipped:
        head += (f". At the shipped {out['shipped_current']:g} the aversive "
                 f"compartment gets {shipped['aversive']['per_cell']:.1f} spikes "
                 f"per cell against {shipped['reward']['per_cell']:.1f} for "
                 f"reward")
    if not balanced:
        return (head + f". No swept amplitude brings them within "
                f"{PULSE_BALANCE:g}x of each other, so the two compartments "
                f"cannot be driven comparably by one number")
    best = min(balanced, key=float)
    v = out["arms"][best]
    return (head + f". {float(best):g} is the smallest amplitude that drives "
            f"both within {PULSE_BALANCE:g}x "
            f"({v['reward']['per_cell']:.1f} against "
            f"{v['aversive']['per_cell']:.1f} spikes per cell)")


# --- motion ----------------------------------------------------------------

# 63% of the retained graph is visual, and most of that is built to detect
# motion. The experiment feeds it a chart that barely changes between
# observations. The plan's hypothesis was that a moving scene would wake it.
# Tested here against a still frame of the SAME picture, so the only
# difference between the two arms is that one of them moves.
MOTION_STEPS = 10
MOTION_PIXELS = 8


def motion(controller, superclass):
    """Does a moving scene recruit the visual system that a still one does not?

    Three arms, ten successive observations each, run without resetting
    between them because motion is a property of a sequence. `still` and
    `flight` crop the same landscape, one always at the same offset and one
    advancing; `shipped` is the package chart with the price history moving
    on, which is what the run loop actually produces.
    """
    brain = controller.brain
    kinds = annotations(brain.ids)
    visual = np.flatnonzero(
        kinds.superclass.fillna("").astype(str).isin(
            ["ol_intrinsic", "visual_projection", "ol_sensory", "visual_centrifugal"]
        ).to_numpy()
    )
    history = scene_history("chop")
    canvas = flight_canvas(history)
    print(f"[motion] {MOTION_STEPS} successive observations x 3 arms; "
          f"{len(visual):,} visual neurons "
          f"({len(visual) / brain.n:.1%} of the graph); flight advances "
          f"{MOTION_PIXELS} px per observation", flush=True)

    arms = {
        "still": lambda i: render_flight(canvas, 0),
        "flight": lambda i: render_flight(canvas, i * MOTION_PIXELS),
        "shipped": lambda i: RENDERERS[SWEEP_RENDERER](history[: 60 + i]),
    }
    out = {
        "steps": MOTION_STEPS,
        "pixels_per_observation": MOTION_PIXELS,
        "visual_neurons": int(len(visual)),
        "arms": {},
    }
    for arm, frame_of in arms.items():
        brain.reset()
        counts, inputs = [], []
        for i in range(MOTION_STEPS):
            frame = frame_of(i)
            inputs.append(frame)
            controller.observe(frame, "none", history)
            counts.append(brain.counts.copy())
        rows = [participation(c, superclass) for c in counts]
        # How much the response moves from one observation to the next. A
        # motion detector answering a translation should change; answering a
        # still frame it should not.
        churn = [
            1.0 - float(((a[visual] > 0) & (b[visual] > 0)).sum())
            / max(1.0, float(((a[visual] > 0) | (b[visual] > 0)).sum()))
            for a, b in zip(counts, counts[1:])
        ]
        pixels = [
            float((a != b).any(axis=2).mean()) for a, b in zip(inputs, inputs[1:])
        ]
        out["arms"][arm] = {
            "active_fraction": float(np.mean([r["active_fraction"] for r in rows])),
            "superclass_entropy_bits": float(
                np.mean([r["superclass_entropy_bits"] for r in rows])
            ),
            "effective_population": float(
                np.mean([r["effective_population"] for r in rows])
            ),
            "total_spikes": float(np.mean([r["total_spikes"] for r in rows])),
            "visual_active_fraction": float(
                np.mean([(c[visual] > 0).mean() for c in counts])
            ),
            "visual_response_churn": float(np.mean(churn)),
            "input_pixels_changing": float(np.mean(pixels)),
        }
        a = out["arms"][arm]
        print(f"    {arm:8} input moves {a['input_pixels_changing']:6.2%} of "
              f"pixels  ->  active {a['active_fraction']:6.2%}  visual "
              f"{a['visual_active_fraction']:6.2%}  churn "
              f"{a['visual_response_churn']:.3f}  entropy "
              f"{a['superclass_entropy_bits']:.2f} bits  spikes "
              f"{a['total_spikes']:,.0f}", flush=True)
    out["verdict"] = verdict_motion(out)
    return out


def verdict_motion(out):
    """Compare motion against a still frame of the same picture.

    Comparing against the shipped chart would confound motion with a different
    image. The only honest pair is still against flight, and the reading that
    matters is whether the visual response moves when the scene does.
    """
    still, flight = out["arms"]["still"], out["arms"]["flight"]
    shipped = out["arms"]["shipped"]
    moved = flight["visual_response_churn"] - still["visual_response_churn"]
    recruited = flight["visual_active_fraction"] - still["visual_active_fraction"]
    head = (f"the same landscape, still against advancing "
            f"{out['pixels_per_observation']} px per observation: input moves "
            f"{still['input_pixels_changing']:.2%} of pixels against "
            f"{flight['input_pixels_changing']:.2%}; visual neurons active "
            f"{still['visual_active_fraction']:.2%} against "
            f"{flight['visual_active_fraction']:.2%}; response churn "
            f"{still['visual_response_churn']:.3f} against "
            f"{flight['visual_response_churn']:.3f}. The shipped chart sits at "
            f"{shipped['visual_active_fraction']:.2%} active, churn "
            f"{shipped['visual_response_churn']:.3f}")
    if recruited > 0.02 or moved > 0.05:
        return (head + ". Motion recruits the visual system that a still frame "
                f"does not, so the hypothesis holds and a moving scene is "
                f"worth building")
    return (head + ". Motion changes neither how much of the visual system "
            f"fires nor how much its response moves. The hypothesis that this "
            f"network needs a moving scene is NOT supported, and by the plan's "
            f"own rule the moving display is dropped rather than built")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    # argparse validates a list default against `choices` as one value, so the
    # default is resolved after parsing instead.
    p.add_argument("tests", nargs="*",
                   choices=["baseline", "laterality", "separation",
                            "sparseness", "olfaction", "kc_input",
                            "odor_tuning", "pathway", "inhibition", "pulse",
                            "motion", "decoder", "ablate", "reinforce"])
    p.add_argument("--frames", type=int, default=8)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--levels", type=float, nargs="+", default=[1, 5, 20, 50])
    p.add_argument("--seed", type=int, default=20260914)
    p.add_argument("--out", type=Path, default=Path("tools/results"))
    a = p.parse_args()
    tests = a.tests or ["baseline", "laterality", "separation", "sparseness",
                        "olfaction", "kc_input", "odor_tuning", "pathway",
                        "inhibition", "pulse", "motion", "decoder", "ablate",
                        "reinforce"]

    started = time.time()
    superclass = np.load(GRAPH, allow_pickle=False)["superclass"]
    print("building controller on the full retained graph...", flush=True)
    controller = FlyController(Settings(learning=False))
    print(f"ready in {time.time() - started:.1f} s: {controller.brain.n:,} neurons, "
          f"{len(controller.brain.post):,} edges", flush=True)

    report = {
        "neurons": int(controller.brain.n),
        "edges": int(len(controller.brain.post)),
        "decoder_cells": controller.decoder.identities,
        "seed": a.seed,
        # Which physiology these numbers were taken at. Without it a recorded
        # table cannot be told apart from the same table measured before a
        # calibration changed underneath it.
        "physiology": {
            "inhibitory_gain": controller.brain.inhibitory_gain,
            "kc_rest_mV": float(controller.brain.rest[controller.brain.circuit["kc"]][0]),
            "adaptation_jump_mV": controller.brain.adaptation_jump,
            "adaptation_tau_ms": controller.brain.adaptation_tau,
            "pulse_current": controller.s.pulse_current,
            "decoder_threshold_hz": controller.s.decoder_threshold_hz,
            "odor_peak_current": controller.olfaction.current,
            "odor_sigma": controller.olfaction.sigma,
            "odor_floor": controller.olfaction.floor,
            "satiety_floor_current": controller.gustation.floor,
            "satiety_span_current": controller.gustation.span,
        },
    }
    for name in tests:
        set_learning(controller, name in ("reinforce", "pulse"))
        if name == "baseline":
            report[name] = baseline(controller, a.frames, superclass)
        elif name == "laterality":
            report[name] = laterality(controller, a.frames, superclass)
        elif name == "separation":
            report[name] = separation(controller, superclass)
        elif name == "sparseness":
            report[name] = sparseness(controller, superclass)
        elif name == "olfaction":
            report[name] = olfaction(controller, superclass)
        elif name == "kc_input":
            report[name] = kc_input(controller, superclass)
        elif name == "odor_tuning":
            report[name] = odor_tuning(controller, superclass)
        elif name == "pathway":
            report[name] = pathway(controller, superclass)
        elif name == "inhibition":
            report[name] = inhibition(controller, superclass)
        elif name == "pulse":
            report[name] = pulse(controller, superclass)
        elif name == "motion":
            report[name] = motion(controller, superclass)
        elif name == "decoder":
            report[name] = decoder(controller, superclass)
        elif name == "ablate":
            report[name] = ablate(controller, a.levels, a.repeats, a.seed, superclass)
        elif name == "reinforce":
            report[name] = reinforce(controller, a.frames, a.seed)
        print(f"  -> {report[name]['verdict']}\n", flush=True)

    report["wall_seconds"] = round(time.time() - started, 1)
    a.out.mkdir(parents=True, exist_ok=True)
    path = a.out / "diagnostics.json"
    # Merge, so running one test at a time does not discard the others.
    if path.exists():
        report = {**json.loads(path.read_text(encoding="utf-8")), **report}
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"written {path}  ({report['wall_seconds']} s)\n", flush=True)
    for name in tests:
        print(f"  {name:11} {report[name]['verdict']}")


if __name__ == "__main__":
    main()
