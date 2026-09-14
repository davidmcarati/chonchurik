"""Watch an evolution run from another window. Read-only, no dependencies.

    python tools\\watch.py
    python tools\\watch.py --out runs/evolution-5min --log evolution.log

A generation at five-minute candles takes about ninety minutes, and nothing is
written to disk until one ends. A plain `tail` on the log is therefore
indistinguishable from a crashed run for an hour and a half, which is what this
is for: the worker panel reads CPU from the scheduler, so the view can say the
run is alive while it has nothing new to report.

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
           "axis": "│", "dot": "·"}
ASCII = {"rule": "-", "full": "#", "empty": ".", "axis": "|", "dot": "-"}
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


def render(out, state, report, procs, previous, plan):
    lines = []
    add = lines.append
    history = (state or {}).get("history", [])
    busy = [p for p in procs if p["cpu"] > 5]

    where = str(out)
    if len(where) > 56:
        where = "..." + where[-53:]
    add(f"{BOLD}Johnny Silverfly{RESET}{DIM} — evolution{RESET}"
        f"{GREY}{where:>56}{RESET}")
    add(GREY + G["rule"] * 79 + RESET)
    if plan.get("label"):
        add(f"{DIM}{plan['label']} {G['dot']} "
            f"{plan.get('population', '?')} genomes x "
            f"{plan.get('generations', '?')} generations {G['dot']} "
            f"{plan.get('observations', '?')} observations x "
            f"{plan.get('starts', '?')} starts{RESET}")
    add(f"{DIM}ranked on profit over buy-and-hold, graded on absolute profit "
        f"against 4 baselines{RESET}")
    add("")

    planned = plan.get("generations") or len(history) or 1
    if history:
        done = len(history)
        pace = sorted(h["seconds"] for h in history)[done // 2]
        left = max(0, planned - done) * pace
        filled = min(2 * BAR, round(2 * BAR * done / planned))
        add(f"{BOLD}generation{RESET} {done}/{planned}  {CYAN}"
            f"{G['full'] * filled}{RESET}{GREY}{G['empty'] * (2 * BAR - filled)}{RESET} "
            f"{100 * done // planned:3d}%")
        add(f"{DIM}per generation{RESET} {clock(pace)}   "
            f"{DIM}remaining{RESET} {clock(left)}   {DIM}ends about{RESET} "
            f"{time.strftime('%H:%M', time.localtime(time.time() + left))}")
        add("")
        scale = max([abs(h["best_fitness"]) for h in history] + [1e-9])
        add(f"{BOLD}best excess over buy-and-hold{RESET}{DIM}, per generation"
            f"{RESET}")
        for h in history:
            v = h["best_fitness"]
            colour = GREEN if v > 0 else (RED if v < 0 else YELLOW)
            tail = f"  {YELLOW}tie with the benchmark{RESET}" if v == 0 else ""
            add(f" {DIM}gen{RESET}{h['generation']:3}  {colour}{v:+8.4f}{RESET} "
                f"{centred_bar(v, scale)}{tail}")
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
        add(f"{DIM}State is written only when a generation ends, so there is "
            f"nothing to plot yet.{RESET}")
        add(f"{DIM}The worker line below is what tells you it is alive.{RESET}")
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


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("runs/evolution-5min"))
    p.add_argument("--log", type=Path,
                   help="the run's stdout, for its population and generation "
                        "count; without it the progress bar has no target")
    p.add_argument("--interval", type=float, default=5)
    p.add_argument("--once", action="store_true",
                   help="print one frame and exit, for a pipe or a check")
    p.add_argument("--ascii", action="store_true",
                   help="plain characters, for a console that will not take "
                        "UTF-8 even when asked")
    a = p.parse_args()

    prepare_console()
    if a.ascii:
        globals()["G"] = dict(ASCII)
    plan = plan_from_log(a.log)
    # Load is a difference between two samples, so one is taken before the
    # first frame is drawn. Without it the view opens on "nan% of one core",
    # which is exactly the moment someone is looking to find out whether their
    # run is alive.
    seed = workers()
    previous = {"cpu": sum(w["cpu"] for w in seed if w["cpu"] > 5), "at": time.time()}
    if previous["cpu"]:
        time.sleep(min(1.0, a.interval))
    try:
        while True:
            frame = render(
                a.out, read_json(a.out / "population.json"),
                read_json(a.out / "champion.json"), workers(), previous,
                plan or plan_from_log(a.log),
            )
            if not a.once:
                sys.stdout.write("\033[H\033[J")
            sys.stdout.write("\n".join(frame) + "\n")
            sys.stdout.flush()
            if a.once:
                return
            time.sleep(a.interval)
    except KeyboardInterrupt:
        sys.stdout.write("\n")


if __name__ == "__main__":
    main()
