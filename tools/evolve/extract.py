"""Pull a loadable genome out of an evolution's saved state.

    python -m tools.evolve.extract --out runs/evolution-5min
    python -m tools.evolve.extract --out runs/evolution-5min --rank 2

`judge` writes `champion-genome.json` when a run finishes, and a run that is
still going, or one that was interrupted, or one that finished before that
file existed, has nothing to load. It does have every genome it has ever
evaluated: `population.json` carries the whole surviving population after each
generation, with fitness. So the search is not lost, only unwired, and this is
the wire.

It writes nothing unless asked with --write, and never into the run it reads.
"""

import argparse
import json
from pathlib import Path

from stonkfly.genome import SPACE

from . import genome as G


def ranked(state):
    """Survivors of the last completed generation, best excess first."""
    survivors = state.get("survivors") or []
    scored = [s for s in survivors if "fitness" in s]
    if not scored:
        raise SystemExit(
            "no generation has finished yet: population.json has no scored "
            "survivors, so there is no best fly to extract"
        )
    return sorted(scored, key=lambda s: -s["fitness"])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("runs/evolution-5min"))
    p.add_argument("--rank", type=int, default=1,
                   help="1 is the best on training excess. The best on train "
                        "is not always the best out of sample -- that is what "
                        "the held-out segments are for -- so the runners-up "
                        "are reachable too")
    p.add_argument("--write", type=Path,
                   help="where to put it; defaults to printing, so a run "
                        "directory is never written to by accident")
    a = p.parse_args()

    state = json.loads((a.out / "population.json").read_text(encoding="utf-8"))
    order = ranked(state)
    if not 1 <= a.rank <= len(order):
        raise SystemExit(f"--rank must be 1 to {len(order)}; "
                         f"{len(order)} survivors were scored")
    pick = order[a.rank - 1]
    genome = {k: pick["genome"][k] for k in sorted(SPACE)}
    text = json.dumps(genome, indent=2) + "\n"

    print(f"generation {state['generation'] - 1}, rank {a.rank} of "
          f"{len(order)}, id {G.identity(pick['genome'])}")
    print(f"  excess over buy-and-hold {pick['fitness']:+.4f}   "
          f"absolute profit {pick.get('profit', float('nan')):+.4f}")
    if a.write:
        a.write.parent.mkdir(parents=True, exist_ok=True)
        a.write.write_text(text, encoding="utf-8")
        print(f"\nwritten {a.write}")
        print(f"  python -m stonkfly run --genome {a.write} --fixture --fast "
              f"--steps 6 --out runs/champion-check")
    else:
        print()
        print(text, end="")
        print("pass --write to save it")


if __name__ == "__main__":
    main()
