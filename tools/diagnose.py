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
  ablate      how much of the brain changes the decision
  reinforce   whether plasticity carries the reinforcement signal
"""

import argparse
import dataclasses
import json
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
from stonkfly.neural.sensory import retinal_samples

from scenes import RENDERERS, SCENES, scene_history

PRODUCT = "BTC-USDC"
BACKGROUND = (235, 240, 249)
HEADER_ROWS = 28


def frames(count):
    """Deterministic offline chart frames, produced exactly as the run loop does."""
    market = FixtureMarket((PRODUCT,))
    out = []
    for _ in range(count):
        quotes = market.snapshot()
        q = quotes[PRODUCT]
        out.append(market_frame(PRODUCT, market.history[PRODUCT], q.bid, q.ask))
        market.record(quotes)
    return out


def observe(controller, frame, reinforcement="none", reset=True):
    """One isolated observation. Reset makes trials independent of each other."""
    if reset:
        controller.brain.reset()
    return controller.observe(frame, reinforcement)


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
    for i, frame in enumerate(frames(count)):
        n = observe(controller, frame)
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

    base = frames(count)
    arms = {
        "A0_baseline": [f for f in base],
        "A1_mirrored": [f[:, ::-1].copy() for f in base],
        "A3_header_repainted": [repaint(f) for f in base],
    }

    result = {"mapped_retina": {"left": nL, "right": nR, "ratio": nR / nL}}
    for name, series in arms.items():
        result[name] = run_arm(controller, name, series, left, right, superclass)

    # A4: additive tonic on the left retina so summed drive matches the right.
    d = drive_of(brain, base[0])
    delta = float((d[right].sum() - d[left].sum()) / nL)
    saved = brain.tonic.copy()
    brain.tonic[brain.retina[left]] += delta
    try:
        result["A4_drive_equalised"] = run_arm(
            controller, "A4_drive_equalised", base, left, right, superclass
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


def run_arm(controller, name, series, left, right, superclass):
    brain = controller.brain
    sides, diffs, ratios = [], [], []
    for i, frame in enumerate(series):
        d = drive_of(brain, frame)
        ratios.append(float(d[right].sum() / d[left].sum()))
        n = observe(controller, frame)
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
    a0 = r["A0_baseline"]["mean_difference_hz"]
    a1 = r["A1_mirrored"]["mean_difference_hz"]
    a4 = r["A4_drive_equalised"]["mean_difference_hz"]
    mix = r["A4_drive_equalised"]["sides"]
    balanced = 0 < mix["BUY"] < sum(mix.values())
    if np.sign(a1) != np.sign(a0) and abs(a1) > 0.25 * abs(a0):
        return ("mirroring flips the bias: chart content dominates after all -> "
                "symmetrise the drawing, not the receptor drive")
    if abs(a4) <= 1.0 and balanced:
        return ("equalising per-side retinal drive removes the bias -> adopt "
                "per-side normalisation as a declared sensory-transfer parameter "
                "(stage 1a) and document the 2.04:1 reconstruction asymmetry")
    if abs(a4) > 0.5 * abs(a0):
        return ("bias survives drive equalisation: it is intrinsic to network "
                "dynamics, not input -> stop fixing the input, go to stage 2")
    return "partial: drive equalisation helps but does not balance the mix"


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

    frame = frames(1)[0]
    control = observe(controller, frame)
    quiet = np.flatnonzero(controller.brain.counts == 0)
    print(f"[ablate] intact {control['side']} "
          f"(R-L {control['difference_hz']:+.2f} Hz), "
          f"{len(quiet):,} silent cells available for the null control",
          flush=True)

    def trial(victims):
        saved = brain.tonic.copy()
        brain.tonic[victims] = -1000.0
        try:
            return observe(controller, frame)
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
    cur = out["current"]
    best = min(
        RENDERERS,
        key=lambda k: (out[k]["kc_identical_pairs"], out[k]["kc_mean_jaccard"],
                       -out[k]["input_mean_receptors_differing"]),
    )
    pairs = len(out["scenes"]) * (len(out["scenes"]) - 1) // 2
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
    if best == "current":
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
    series = frames(count)
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
        for i, (frame, kind) in enumerate(zip(series, order)):
            n = observe(controller, frame, kind, reset=False)
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

    Edge counts mislead here the same way label flips mislead in `ablate`: the
    no-reward arm touches almost as many edges as the reinforced arms. What
    reinforcement changes is how far they move, so the relative L1 is the
    measurement and the counts are context.
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
    ratio = presence / timing
    return (f"reinforcement carries signal: presence moves the memory "
            f"{presence:.1%} (relative L1) against a {changed['none']:,}-edge "
            f"endogenous baseline, timing moves it {timing:.1%}. Presence "
            f"dominates timing {ratio:.1f}x -- the rule responds mostly to HOW "
            "MUCH dopamine arrived, not WHEN. GA is unblocked, but what it can "
            "select on is dose, not credit assignment")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    # argparse validates a list default against `choices` as one value, so the
    # default is resolved after parsing instead.
    p.add_argument("tests", nargs="*",
                   choices=["baseline", "laterality", "separation", "ablate",
                            "reinforce"])
    p.add_argument("--frames", type=int, default=8)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--levels", type=float, nargs="+", default=[1, 5, 20, 50])
    p.add_argument("--seed", type=int, default=20260914)
    p.add_argument("--out", type=Path, default=Path("tools/results"))
    a = p.parse_args()
    tests = a.tests or ["baseline", "laterality", "separation", "ablate",
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
    }
    for name in tests:
        set_learning(controller, name == "reinforce")
        if name == "baseline":
            report[name] = baseline(controller, a.frames, superclass)
        elif name == "laterality":
            report[name] = laterality(controller, a.frames, superclass)
        elif name == "separation":
            report[name] = separation(controller, superclass)
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
