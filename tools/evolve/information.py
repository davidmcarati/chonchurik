"""How much a signal knows about what price does next. The search's objective.

This was a diagnostic (`tools/ic.py`) before it was an objective. It is here
because selection now uses it, and a statistic that decides what breeds cannot
live in a tool that imports the thing it grades.

Why the search is ranked on this and not on money
-------------------------------------------------
Money was the objective for the first three generations on real candles, and
it selected something nobody asked for. A round trip costs the fee twice, so
at chance-level direction every trade loses, and fitness collapsed into a
function of how little a fly traded: correlation -0.97 with the number of
sells across the surviving population, and a champion that was buy-and-hold
with the change.

Underneath that was something worse. The readout carries a constant that
differs by genome -- measured at -17.15 Hz for the wild type against +9.05,
+8.60 and +4.42 for three evolved ones, while the spread stayed between 4.76
and 6.34 for all four. The search moved that constant 26 Hz and never touched
the variation. It was selecting the sign of an offset, and the offset decided
every proposal.

An information coefficient cannot be fooled that way. It demeans the signal,
so a constant contributes exactly nothing to it, whatever its size or sign.
What is left is the part that moves -- which is the only part that could ever
have known anything.

The statistic
-------------
With `s` the signal at an observation and `r` the return over the next
`horizon` bars, both demeaned inside the run:

    edge = mean(s * r)
    ic   = edge / (std(r) * rms(s))

For a continuous readout, demeaning the signal makes this an ordinary Pearson
correlation. For ternary proposals the signal is deliberately left uncentred,
so a fly that proposes BUY at every observation scores exactly zero rather
than scoring its own one-sidedness: drift is not skill.

The null is the signal *circularly shifted* against the same prices. Flies
propose in runs and readouts drift in runs; a plain shuffle would destroy that
autocorrelation and give a null far too narrow to fail against. A shift keeps
the signal exactly as autocorrelated as it is and destroys only the alignment,
which is the thing being tested.
"""

import math

SIGN = {"BUY": 1.0, "SELL": -1.0, "HOLD": 0.0}


def forward(prices, index, horizon):
    """Return over the next `horizon` bars, or None past the end of the data."""
    if index + horizon >= len(prices):
        return None
    return prices[index + horizon] / prices[index] - 1.0


def aligned(prices, values, base, horizon, centre=False):
    """(signal, demeaned forward return) for one run at one horizon.

    `base` is the index of the price the first value was produced at; the rest
    follow consecutively. `values` are proposals or readouts -- a string is
    read through SIGN, a number is taken as it is.

    `centre` demeans the signal too, which turns `ic` below into a Pearson
    correlation. Right for a continuous readout, whose resting level is a
    property of the circuit; deliberately wrong for proposals, where leaving
    the mean in is what makes a one-sided fly score zero.
    """
    signal, ret = [], []
    for i, value in enumerate(values):
        r = forward(prices, base + i, horizon)
        if r is None:
            break
        signal.append(SIGN[value] if isinstance(value, str) else float(value))
        ret.append(r)
    if not ret:
        return [], []
    mean = sum(ret) / len(ret)
    if centre:
        offset = sum(signal) / len(signal)
        signal = [x - offset for x in signal]
    return signal, [r - mean for r in ret]


def edge(runs):
    """mean(s * r) pooled over runs, each already demeaned inside itself."""
    total = count = 0.0
    for signal, ret in runs:
        for s, r in zip(signal, ret):
            total += s * r
            count += 1
    return total / count if count else 0.0


def normaliser(runs):
    """std(r) * rms(s), pooled. Turns the edge into a correlation."""
    rs, ss = [], []
    for signal, ret in runs:
        rs.extend(ret)
        ss.extend(signal)
    if not rs:
        return 0.0
    var = sum(r * r for r in rs) / len(rs)
    power = sum(s * s for s in ss) / len(ss)
    return math.sqrt(var) * math.sqrt(power)


def coefficient(runs):
    """The information coefficient itself. Zero when there is nothing to divide."""
    scale = normaliser(runs)
    return edge(runs) / scale if scale else 0.0


def shifted(runs, rng):
    """Every run's signal rolled by its own random amount. The null."""
    out = []
    for signal, ret in runs:
        n = len(signal)
        k = rng.randrange(1, n) if n > 1 else 0
        out.append((signal[k:] + signal[:k], ret))
    return out


def test(runs, rng, draws=2000):
    """The coefficient, and where it sits in the shifted null."""
    observed = edge(runs)
    null = [edge(shifted(runs, rng)) for _ in range(draws)]
    spread = math.sqrt(sum(x * x for x in null) / len(null))
    extreme = sum(1 for x in null if abs(x) >= abs(observed))
    scale = normaliser(runs)
    # A constant signal shifts into itself, so every draw lands on the same
    # (floating-point) zero the observation did. The ratio of two of those is
    # a number with no meaning; `p` already says 1.0, and `z` should agree.
    if spread <= 1e-9 * scale:
        spread = 0.0
    return {
        "edge_bps": observed * 10000,
        "ic": observed / scale if scale else 0.0,
        "z": observed / spread if spread else 0.0,
        "p": (extreme + 1) / (len(null) + 1),
        "n": sum(len(r) for _, r in runs),
    }


def readout_ic(prices, values, base, horizon):
    """One run's readout against the forward return. What fitness is made of.

    No permutation test: this is called for every fly at every start of every
    generation, and the null belongs in the report rather than in the loop.
    """
    signal, ret = aligned(prices, values, base, horizon, centre=True)
    return coefficient([(signal, ret)]) if ret else 0.0
