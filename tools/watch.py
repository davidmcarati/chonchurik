"""Watch an evolution run from another window. Read-only, no dependencies.

    python tools\\watch.py
    python tools\\watch.py --out runs/evolution-5min --log evolution.log

A generation takes tens of minutes -- measured, half an hour for eighty-four
genomes on the card and an hour and a half for twenty-four on the pool -- and
nothing is written to disk until one ends. A plain `tail` on the log is
therefore indistinguishable from a crashed run for that whole time, which is
what this is for: the worker panel reads CPU from the scheduler, so the view
can say the run is alive while it has nothing new to report.

It never writes to the run directory, never imports the model and holds no file
open, so it cannot disturb the run it watches or lose a race with the atomic
replace that updates the state.
"""

import argparse
import ctypes
import json
import re
import subprocess
import sys
import time
from pathlib import Path

BAR = 20
RESET, BOLD, DIM = "\033[0m", "\033[1m", "\033[2m"
RED, GREEN, YELLOW, CYAN, GREY = (
    "\033[31m", "\033[32m", "\033[33m", "\033[36m", "\033[90m"
)
PRIORITY = {0x4000: "BelowNormal", 0x20: "Normal", 0x8000: "AboveNormal",
            0x80: "High", 0x40: "Idle", 0x100: "Realtime"}

# A fresh cmd.exe is code page 437 or 1252, neither of which can encode a box
# character. Asked for UTF-8 first; if the console refuses, every glyph falls
# back to something ASCII rather than the view dying on a UnicodeEncodeError
# in front of the person who just wanted to see how their run was going.
UNICODE = {"rule": "─", "full": "█", "empty": "░",
           "axis": "│", "dot": "·", "fly": "➤", "zero": "┼",
           "left": "├", "right": "┤", "one": "·", "few": ":", "many": "#",
           "spin": "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"}
ASCII = {"rule": "-", "full": "#", "empty": ".", "axis": "|", "dot": "-",
         "fly": ">", "zero": "+", "left": "[", "right": "]", "one": ".",
         "few": ":", "many": "#", "spin": "|/-\\"}
G = dict(UNICODE)


def prepare_console():
    """Colour and a code page that can draw a bar. Falls back rather than fails."""
    global G
    if sys.platform == "win32":
        k = ctypes.WinDLL("kernel32", use_last_error=True)
        k.GetStdHandle.restype = ctypes.c_void_p
        handle = k.GetStdHandle(-11)
        mode = ctypes.c_uint32()
        if k.GetConsoleMode(ctypes.c_void_p(handle), ctypes.byref(mode)):
            k.SetConsoleMode(ctypes.c_void_p(handle), mode.value | 0x0004)
        k.SetConsoleOutputCP(65001)
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="strict")
    except (AttributeError, OSError, LookupError):
        pass
    try:
        "".join(UNICODE.values()).encode(sys.stdout.encoding or "ascii")
    except (UnicodeEncodeError, LookupError):
        G = dict(ASCII)


# --- what the operating system knows ---------------------------------------


class Filetime(ctypes.Structure):
    _fields_ = [("low", ctypes.c_uint32), ("high", ctypes.c_uint32)]

    @property
    def ticks(self):
        return (self.high << 32) | self.low


class Counters(ctypes.Structure):
    _fields_ = [("cb", ctypes.c_uint32), ("faults", ctypes.c_uint32)] + [
        (n, ctypes.c_size_t) for n in [
            "peak_working_set", "working_set", "quota_peak_paged", "quota_paged",
            "quota_peak_nonpaged", "quota_nonpaged", "pagefile", "peak_pagefile",
        ]
    ]


def python_pids():
    if sys.platform == "win32":
        command = ["tasklist", "/FI", "IMAGENAME eq python.exe", "/FO", "CSV",
                   "/NH"]
    else:
        command = ["pgrep", "-f", "tools.evolve"]
    try:
        out = subprocess.run(
            command, capture_output=True, text=True, timeout=8,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    if sys.platform != "win32":
        return [int(x) for x in out.split() if x.isdigit()]
    pids = []
    for line in out.splitlines():
        parts = [p.strip('"') for p in line.split('","')]
        if len(parts) > 1 and parts[1].isdigit():
            pids.append(int(parts[1]))
    return pids


def workers():
    """CPU seconds, resident bytes, priority and start time, straight from Windows.

    Reported rather than assumed: the whole reason the bounded pool existed was
    to run below normal priority, and it silently failed to for months. A view
    that printed the intended priority instead of the actual one would have
    reported that bug as working.
    """
    if sys.platform != "win32":
        return []
    k = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    k.OpenProcess.restype = ctypes.c_void_p
    k.GetPriorityClass.argtypes = [ctypes.c_void_p]
    k.GetPriorityClass.restype = ctypes.c_uint32
    k.CloseHandle.argtypes = [ctypes.c_void_p]
    now = time.time()
    found = []
    for pid in python_pids():
        handle = k.OpenProcess(0x1000 | 0x0010, False, pid)
        if not handle:
            continue
        try:
            made, ended, kernel, user = (Filetime() for _ in range(4))
            if not k.GetProcessTimes(
                ctypes.c_void_p(handle), ctypes.byref(made), ctypes.byref(ended),
                ctypes.byref(kernel), ctypes.byref(user)
            ):
                continue
            counters = Counters()
            counters.cb = ctypes.sizeof(Counters)
            psapi.GetProcessMemoryInfo(
                ctypes.c_void_p(handle), ctypes.byref(counters),
                ctypes.sizeof(Counters),
            )
            # FILETIME counts 100 ns ticks from 1601; 11644473600 s to the epoch.
            found.append({
                "pid": pid,
                "cpu": (kernel.ticks + user.ticks) / 1e7,
                "rss": counters.working_set,
                "priority": k.GetPriorityClass(ctypes.c_void_p(handle)),
                "age": now - (made.ticks / 1e7 - 11644473600),
            })
        finally:
            k.CloseHandle(ctypes.c_void_p(handle))
    return found


# --- the run's own record --------------------------------------------------


def read_json(path):
    """Tolerant of the atomic replace that rewrites this file underneath us."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def find_log(out, given):
    """The run's stdout, if it can be found without being told.

    A double-clicked executable gets no arguments, so the usual places are
    tried: the run directory itself, then the newest log beside it. Returns
    None rather than guessing wildly, and an unknown target is drawn as
    unknown rather than as finished.
    """
    if given:
        return given
    for candidate in [out / "evolution.log", out / "run.log",
                      out.parent / f"{out.name}.log"]:
        if candidate.exists():
            return candidate
    try:
        beside = sorted(out.parent.glob("*.log"), key=lambda f: -f.stat().st_mtime)
    except OSError:
        return None
    return beside[0] if beside else None


def plan_from_log(path):
    """Population, generations and the source, from the two header lines.

    Taken from the log because the running job predates recording them in its
    own state, and a view that only worked against a future format would be
    useless exactly when it is wanted.
    """
    if not path:
        return {}
    try:
        head = path.read_text(encoding="utf-8", errors="replace")[:2000]
    except OSError:
        return {}
    plan = {}
    source = re.search(r"^source (.+?): (\d+) observations", head, re.M)
    if source:
        plan["label"] = source.group(1)
    shape = re.search(
        r"(\d+) genomes x (\d+) generations on (\d+) workers.*?;\s*"
        r"(\d+) observations x (\d+) starts", head, re.S,
    )
    if shape:
        plan.update(
            population=int(shape.group(1)), generations=int(shape.group(2)),
            workers=int(shape.group(3)), observations=int(shape.group(4)),
            starts=int(shape.group(5)),
        )
    return plan


# --- drawing ---------------------------------------------------------------


def clock(seconds):
    seconds = int(max(0, seconds))
    if seconds < 3600:
        return f"{seconds // 60}m{seconds % 60:02d}s"
    return f"{seconds // 3600}h{seconds % 3600 // 60:02d}m"


def centred_bar(value, scale, width=BAR):
    """Grows left or right from a fixed centre, because excess can be negative.

    A left-anchored bar draws -1.4 and +1.4 the same length and lets the eye
    read a loss as progress.
    """
    cells = 0 if scale <= 0 else min(width, round(abs(value) / scale * width))
    if value > 0:
        return " " * width + G["axis"] + GREEN + G["full"] * cells + RESET
    if value < 0:
        return (" " * (width - cells) + RED + G["full"] * cells + RESET + G["axis"])
    return " " * width + YELLOW + G["axis"] + RESET


def wrap(text, width):
    line, out = "", []
    for word in text.split():
        if len(line) + len(word) + 1 > width:
            out.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    if line:
        out.append(line)
    return out


# --- the race --------------------------------------------------------------

TRACK = 52


def lane(value, low, high, width=TRACK):
    """Which column of the track a profit sits at. Clamped, never wrapped."""
    if high <= low:
        return width // 2
    return max(0, min(width - 1, round((value - low) / (high - low) * (width - 1))))


def swarm(values, low, high, width=TRACK):
    """Every fly at once, as a density of characters along the track.

    Eighty-four lanes will not fit on a console, and the shape of the field is
    the interesting part anyway: whether the population is spread out or piled
    on one number.
    """
    counts = [0] * width
    for v in values:
        counts[lane(v, low, high, width)] += 1
    out = []
    for n in counts:
        out.append(" " if not n else G["one"] if n == 1
                   else G["few"] if n < 4 else G["many"])
    return "".join(out)


def runner_line(label, value, low, high, colour, width=TRACK):
    """One named fly on its own lane, with the number beside it."""
    at = lane(value, low, high, width)
    track = [" "] * width
    zero = lane(0.0, low, high, width) if low < 0 < high else None
    if zero is not None:
        track[zero] = G["axis"]
    track[at] = "@"
    drawn = "".join(track).replace("@", f"{colour}{G['fly']}{RESET}")
    return f"  {DIM}{label:<7}{RESET}{drawn} {colour}{value:+8.4f}{RESET}"


def heartbeat(progress, report, frame):
    """How long since the run last said anything, and whether that is fine."""
    spin = G["spin"][frame % len(G["spin"])]
    age = time.time() - progress.get("updated", 0)
    if report:
        return f"{DIM}finished{RESET}"
    if age < 5:
        return f"{GREEN}{spin} racing{RESET}{DIM} {age:.1f}s ago{RESET}"
    if age < 90:
        return f"{YELLOW}{spin} last beat {clock(age)} ago{RESET}"
    return f"{RED}no beat for {clock(age)} — stalled?{RESET}"


def race(progress, report, frame):
    """The live panel: what the run is doing right now, and who is winning."""
    lines = []
    add = lines.append
    stage = progress.get("stage") or "running"
    if G is ASCII or G.get("fly") == ">":
        # The stage text comes from the run, not from this file, so a console
        # that refused UTF-8 would still be handed whatever the loop wrote.
        stage = stage.replace("·", "-").encode("ascii", "replace").decode()
    flies = progress.get("flies", 0)
    starts = progress.get("starts") or []
    where = (f"{len(starts)} starts" if len(starts) > 1
             else f"start {starts[0]}" if starts else "")
    wave, waves = progress.get("wave", 0), progress.get("waves", 0)
    add(f"{BOLD}THE RACE{RESET}  {stage}"
        + (f"{DIM} {G['dot']} wave {wave} of {waves}{RESET}" if waves > 1 else "")
        + f"{DIM} {G['dot']} {flies} flies {G['dot']} {where}{RESET}")

    done = progress.get("observation", 0)
    total = max(1, progress.get("observations", 1))
    filled = min(BAR, round(BAR * done / total))
    # No percentage: the bar already is one, and this line has to fit an
    # eighty-column console beside the rate and the heartbeat.
    add(f" {DIM}obs{RESET} {done:3}/{total:<4}{CYAN}{G['full'] * filled}{RESET}"
        f"{GREY}{G['empty'] * (BAR - filled)}{RESET}  "
        f"{DIM}{progress.get('rate', 0):.1f} fly-obs/s{RESET}  "
        f"{heartbeat(progress, report, frame)}")

    profit = progress.get("profit") or []
    if not profit:
        return lines
    warmup = progress.get("warmup", 0)
    if done <= warmup:
        add(f"  {DIM}warm-up {G['dot']} mushroom bodies charging, accounts "
            f"open at observation {warmup + 1}{RESET}")
        return lines

    low, high = min(profit), max(profit)
    if high <= low:
        add(f"  {DIM}every fly is at {low:+.4f} — nobody has filled an order "
            f"yet{RESET}")
        return lines
    pad = " " * 9
    add(f"{DIM}{low:>+8.4f} {G['left']}{G['rule'] * (TRACK - 2)}"
        f"{G['right']} {high:+.4f}{RESET}")
    add(f"  {DIM}{'swarm':<7}{RESET}{CYAN}{swarm(profit, low, high)}{RESET}"
        f" {DIM}{len(profit)} flies{RESET}")
    order = sorted(range(len(profit)), key=lambda i: -profit[i])
    picks = [("leader", order[0], GREEN),
             ("median", order[len(order) // 2], YELLOW),
             ("tail", order[-1], RED)]
    # A fly that has never filled an order sits at exactly zero, and early in
    # a generation most of them do. Saying how many stops "leader +0.0000"
    # from reading as a fly that is winning.
    idle = sum(1 for x in profit if x == 0)
    seen = set()
    for label, i, colour in picks:
        if i in seen:
            continue
        seen.add(i)
        note = (f" {GREY}fly {i}{RESET}" if profit[i] != 0
                else f" {GREY}{idle} flat{RESET}")
        add(runner_line(f"{label}", profit[i], low, high, colour) + note)
    return lines


def render(out, state, report, procs, previous, plan,
           progress=None, frame=0):
    lines = []
    add = lines.append
    history = (state or {}).get("history", [])
    busy = [p for p in procs if p["cpu"] > 5]

    where = str(out)
    if len(where) > 50:
        where = "..." + where[-47:]
    add(f"{BOLD}Johnny Silverfly{RESET}{DIM} — evolution{RESET}"
        f"{GREY}{where:>50}{RESET}")
    add(GREY + G["rule"] * 79 + RESET)
    if plan.get("label"):
        add(f"{DIM}{plan['label']} {G['dot']} "
            f"{plan.get('population', '?')} genomes x "
            f"{plan.get('generations', '?')} generations {G['dot']} "
            f"{plan.get('observations', '?')} observations x "
            f"{plan.get('starts', '?')} starts{RESET}")
    add(f"{DIM}ranked on {plan.get('objective', 'excess over buy-and-hold')}, "
        f"graded on profit against 4 baselines{RESET}")
    add("")

    # First, because it is the only part that changes while a generation is
    # still running, and "is it stuck" is the question the view exists for.
    if progress:
        lines.extend(race(progress, report, frame))
        add("")

    planned = plan.get("generations")
    if history:
        done = len(history)
        pace = sorted(h["seconds"] for h in history)[done // 2]
        if planned:
            left = max(0, planned - done) * pace
            filled = min(2 * BAR, round(2 * BAR * done / planned))
            add(f"{BOLD}generation{RESET} {done}/{planned}  {CYAN}"
                f"{G['full'] * filled}{RESET}{GREY}"
                f"{G['empty'] * (2 * BAR - filled)}{RESET} "
                f"{100 * done // planned:3d}%")
            add(f"{DIM}per generation{RESET} {clock(pace)}   "
                f"{DIM}remaining{RESET} {clock(left)}   {DIM}ends about{RESET} "
                f"{time.strftime('%H:%M', time.localtime(time.time() + left))}")
        else:
            # A full bar drawn because the target is unknown would report an
            # unfinished run as a finished one.
            add(f"{BOLD}generation{RESET} {done} done   {DIM}target unknown "
                f"— pass --generations or --log{RESET}")
            add(f"{DIM}per generation{RESET} {clock(pace)}")
        add("")
        def held_of(h):
            """The champion's own out-of-sample score, or None.

            Position zero and never a maximum over the group: the number is
            there to contradict the train score beside it, and a best-of
            cannot. Older runs stored only the list.
            """
            if h.get("holdout_champion") is not None:
                return h["holdout_champion"]
            values = h.get("holdout_fitness")
            return values[0] if values else None

        scale = max([abs(h["best_fitness"]) for h in history]
                    + [abs(held_of(h) or 0.0) for h in history] + [1e-9])
        add(f"{BOLD}best {plan.get('objective', 'excess over buy-and-hold')}"
            f"{RESET}{DIM}, per generation {G['dot']} train against the "
            f"champion's own held-out score{RESET}")
        for h in history:
            v = h["best_fitness"]
            colour = GREEN if v > 0 else (RED if v < 0 else YELLOW)
            out = held_of(h)
            if out is None:
                note = f"  {DIM}holdout   --   {RESET}"
            else:
                # Apart is the thing to notice, so it is what gets coloured.
                near = abs(v - out) <= 0.5 * max(abs(v), 1e-9)
                mark = GREEN if near else RED
                note = (f"  {DIM}holdout{RESET} {mark}{out:+7.4f}{RESET}"
                        + ("" if near else f" {RED}apart{RESET}"))
            tail = f"  {YELLOW}tie with the benchmark{RESET}" if v == 0 else ""
            add(f" {DIM}gen{RESET}{h['generation']:3}  {colour}{v:+8.4f}{RESET} "
                f"{centred_bar(v, scale)}{note}{tail}")
        add("")
        last = history[-1]
        mid = last.get("best_median_start")
        add(f"{BOLD}last generation{RESET}{DIM}  #{last['generation']} took "
            f"{clock(last['seconds'])}{RESET}")
        if mid:
            add(f"  at its median start   profit {mid['profit']:+8.4f}   "
                f"buy+hold {mid['buy_and_hold']:+8.4f}   "
                f"{DIM}(these two subtract exactly){RESET}")
        else:
            add(f"  {DIM}this run predates per-start reporting; only the "
                f"medians above are available{RESET}")
        add(f"  population median     {last['median_fitness']:+8.4f} "
            f"{DIM}excess{RESET}")
        add(f"  killed at screening   {last['degenerate']:8} {DIM}of "
            f"{last['evaluated']} — one-sided, silent, or never filled{RESET}")
    else:
        add(f"{YELLOW}generation 0 has not finished{RESET}")
        add(f"{DIM}Per-generation state is written only when a generation "
            f"ends, so there is{RESET}")
        add(f"{DIM}nothing to plot here yet.{RESET}")
        add(f"{DIM}{'The race above is the live one.' if progress else
                    'The worker line below is what tells you it is alive.'}"
            f"{RESET}")
    add("")

    if busy:
        cpu = sum(p["cpu"] for p in busy)
        moved = cpu - previous["cpu"]
        window = max(1e-9, time.time() - previous["at"])
        load = 100 * moved / window if previous["cpu"] else float("nan")
        names = sorted({PRIORITY.get(p["priority"], hex(p["priority"]))
                        for p in busy})
        elapsed = clock(max(p["age"] for p in busy))
        if load != load:
            verdict = f"{DIM}measuring…{RESET}"
        elif load > 50 * len(busy):
            verdict = f"{GREEN}working{RESET}"
        elif load > 20:
            verdict = f"{YELLOW}partly idle{RESET}"
        else:
            verdict = f"{RED}burning no cpu — stalled?{RESET}"
        if report:
            verdict = f"{DIM}but this run has already finished{RESET}"
        add(f"{BOLD}workers{RESET} {len(busy)} busy   "
            f"{sum(p['rss'] for p in busy) / 1e9:.1f} GB   "
            f"{'/'.join(names)}   {load:.0f}% of one core   {verdict}")
        add(f"{DIM}running for {elapsed}"
            f"{' — python processes are counted machine-wide, so these may '
               'belong to another run' if report else ''}{RESET}")
        if "Normal" in names and "BelowNormal" not in names:
            add(f"  {RED}not below normal — this run is competing with your "
                f"desktop{RESET}")
        previous.update(cpu=cpu, at=time.time())
    else:
        add(f"{BOLD}workers{RESET} {RED}none busy — the run is not going{RESET}")

    if report:
        add("")
        add(f"{BOLD}VERDICT{RESET}")
        for chunk in wrap(report.get("verdict", ""), 75):
            add("  " + chunk)

    add("")
    add(f"{GREY}ctrl-c to quit {G['dot']} reads only, writes nothing {G['dot']} "
        f"{time.strftime('%H:%M:%S')}{RESET}")
    return lines


def newest(root=Path("runs")):
    """The run that most recently said anything.

    The default used to name one run directory, which was right on the day it
    was written and wrong every day after: the view sat on a finished run
    reporting it as stalled while another was going beside it. A heartbeat is
    what "this one is live" means everywhere else here, so it is what picks.
    """
    beats = sorted(root.glob("*/progress.json"),
                   key=lambda f: -f.stat().st_mtime)
    if beats:
        return beats[0].parent
    states = sorted(root.glob("*/population.json"),
                    key=lambda f: -f.stat().st_mtime)
    return states[0].parent if states else root / "evolution"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=None,
                   help="run directory to watch. Defaults to whichever under "
                        "runs/ last wrote a heartbeat, so a finished run does "
                        "not keep the view pointed at itself while a new one "
                        "is going")
    p.add_argument("--log", type=Path,
                   help="the run's stdout, for its population and generation "
                        "count; looked for beside the run directory if omitted")
    p.add_argument("--generations", type=int,
                   help="the target, when there is no log to read it from")
    p.add_argument("--interval", type=float, default=1.0,
                   help="seconds between frames; the race moves "
                        "every observation, so this is a second "
                        "rather than the five a generation bar "
                        "would need")
    p.add_argument("--once", action="store_true",
                   help="print one frame and exit, for a pipe or a check")
    p.add_argument("--ascii", action="store_true",
                   help="plain characters, for a console that will not take "
                        "UTF-8 even when asked")
    a = p.parse_args()

    prepare_console()
    if a.ascii:
        globals()["G"] = dict(ASCII)
    # A run writes its own plan.json; the log is the fallback for runs that
    # predate it, and --generations the fallback for having neither. This
    # ordering is what lets the view be started with no arguments at all.
    if a.out is None:
        a.out = newest()
    plan = read_json(a.out / "plan.json") or plan_from_log(find_log(a.out, a.log))
    if a.generations:
        plan["generations"] = a.generations
    # Load is a difference between two samples, so one is taken before the
    # first frame is drawn. Without it the view opens on "nan% of one core",
    # which is exactly the moment someone is looking to find out whether their
    # run is alive.
    seed = workers()
    previous = {"cpu": sum(w["cpu"] for w in seed if w["cpu"] > 5), "at": time.time()}
    if previous["cpu"]:
        time.sleep(min(1.0, a.interval))
    tick = 0
    try:
        while True:
            drawn = render(
                a.out, read_json(a.out / "population.json"),
                read_json(a.out / "champion.json"), workers(), previous,
                plan, read_json(a.out / "progress.json"), tick,
            )
            if not a.once:
                sys.stdout.write("\033[H\033[J")
            sys.stdout.write("\n".join(drawn) + "\n")
            sys.stdout.flush()
            if a.once:
                return
            tick += 1
            time.sleep(a.interval)
    except KeyboardInterrupt:
        sys.stdout.write("\n")


if __name__ == "__main__":
    main()
