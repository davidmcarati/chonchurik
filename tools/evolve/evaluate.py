"""Replay a price segment through one fly and report what the account did.

Fitness is profit *in excess of buying and holding the same window*, and it is
computed from the network's own proposals: nothing here overrides, replaces or
second-guesses a BUY, a SELL or a HOLD. What this module does is fill them.

The benchmark is subtracted because absolute profit does not measure trading.
The first generation on real candles produced a champion at
+0.1866, the first positive number in the project -- which proposed BUY at 282
to 289 of 300 observations, had 264 to 276 of those rejected for want of
budget, and lost to buying and holding at five starts out of five. Maximum
exposure is the profit-maximising policy in a window that rises, so profit
fitness selects for it, and the kill criterion then requires beating the very
thing that fitness was pushing the population towards. Subtracting the
benchmark removes the market's own drift from the objective and leaves the
timing, which is the thing being searched for.

Execution here is a deliberately explicit paper simulator, NOT the production
path. It applies budget, inventory, order size and the same fee rate, and it
does not apply the live guard's cooldown, spread, quote-age or STOP checks.
A champion is therefore not a validated trading result until it has been
re-run through `python -m stonkfly run`, and the report says so.
"""

import contextlib
import dataclasses

import numpy as np

from stonkfly.config import Settings
from stonkfly.genome import apply as apply_genome
from stonkfly.genome import pristine_inhibitory
from stonkfly.display import market_frame
from stonkfly.neural.controller import FlyController
from stonkfly.neural.olfaction import FAST
from stonkfly.reinforcement import reinforcement

from .genome import WILD_TYPE
from .series import quotes

PRODUCT = "BTC-USDC"
CHART_WINDOW = 100
# Kenyon activity needs about eight observations to reach 90% of its plateau,
# measured. Evaluating a fly before that measures a warm-up, not a fly.
WARMUP = 15


@contextlib.contextmanager
def configured(controller, genome, pristine_inhibitory):
    """Apply a genome to an already-built brain, then put it back exactly.

    Building the graph costs a second and a gigabyte, so a worker builds one
    brain and reconfigures it per genome. Every change is reversed here, and
    the caller asserts the graph is byte-identical between generations.
    """
    brain = controller.brain
    kc = brain.circuit["kc"]
    saved = {
        "settings": controller.s,
        "rest": brain.rest[kc].copy(),
        "initial_v": brain.initial["v"][kc].copy(),
        "tonic": brain.tonic[brain.lamina].copy(),
        "jump": brain.adaptation_jump,
        "tau": brain.adaptation_tau,
        "eta": brain.eta,
        "dan": brain.dan_baseline_hz.copy(),
        "gain": brain.inhibitory_gain,
        # Kept verbatim, not recomputed from the gain: float32 does not
        # survive a divide-and-multiply round trip, and a restore that is
        # nearly right would drift a little further with every genome the
        # worker sees.
        "inhibitory": brain.weight[brain.inhibitory_edges].copy(),
        "odor": (
            controller.olfaction.current,
            controller.olfaction.sigma,
            controller.olfaction.floor,
        ),
        "taste": (controller.gustation.floor, controller.gustation.span),
    }
    try:
        # One implementation, in the model. A champion loaded by
        # `python -m stonkfly run --genome` has to be the fly the search
        # scored, and two copies of these assignments would drift apart
        # exactly where nobody would look.
        apply_genome(controller, genome, pristine_inhibitory)
        # replace(), not a fresh Settings: capital, order size, fee and
        # learning belong to the experiment, not to the genome.
        controller.s = dataclasses.replace(
            saved["settings"],
            pulse_current=genome["pulse_current"],
            decoder_threshold_hz=genome["decoder_threshold_hz"],
        )
        yield
    finally:
        controller.s = saved["settings"]
        brain.rest[kc] = saved["rest"]
        brain.initial["v"][kc] = saved["initial_v"]
        brain.tonic[brain.lamina] = saved["tonic"]
        brain.adaptation_jump = saved["jump"]
        brain.adaptation_tau = saved["tau"]
        brain.eta = saved["eta"]
        brain.dan_baseline_hz[:] = saved["dan"]
        brain.inhibitory_gain = saved["gain"]
        brain.weight[brain.inhibitory_edges] = saved["inhibitory"]
        # Learning moved the plastic edges; reset puts them back to baseline,
        # so the next genome starts from the same brain this one did.
        brain.reset()
        (
            controller.olfaction.current,
            controller.olfaction.sigma,
            controller.olfaction.floor,
        ) = saved["odor"]
        controller.gustation.floor, controller.gustation.span = saved["taste"]


class Account:
    """Cash, one position, and the same fee the paper broker charges."""

    def __init__(self, capital, order_size, fee):
        self.start = float(capital)
        self.cash = float(capital)
        self.order = float(order_size)
        self.fee = float(fee)
        self.base = 0.0
        self.fills = {"BUY": 0, "SELL": 0}
        self.rejected = 0

    def equity(self, bid):
        return self.cash + self.base * bid

    def apply(self, side, bid, ask):
        """Fill a proposal if the account can take it. Never chooses a trade."""
        if side == "BUY":
            cost = self.order * (1 + self.fee)
            if cost > self.cash:
                self.rejected += 1
                return None
            self.cash -= cost
            self.base += self.order / ask
            self.fills["BUY"] += 1
            return "BUY"
        if side == "SELL":
            size = self.order / bid
            if size > self.base:
                self.rejected += 1
                return None
            self.base -= size
            self.cash += self.order * (1 - self.fee)
            self.fills["SELL"] += 1
            return "SELL"
        return None


def replay(controller, prices, start, observations, propose, account,
           bars=None):
    """One chronological pass. `propose` turns an observation into a side."""
    history = list(prices[start:start + CHART_WINDOW])
    cursor = start + CHART_WINDOW
    controller.brain.reset()
    anchor = str(account.start)
    executed = None
    sides = []
    kc = 0
    for step in range(WARMUP + observations):
        if cursor >= len(prices):
            break
        price = prices[cursor]
        bid, ask = quotes(price)
        equity = account.equity(bid)
        kind, _ = reinforcement(str(equity), anchor, "0.01")
        side, spikes = propose(
            market_frame(PRODUCT, history, bid, ask), kind, history, executed,
            (str(equity), anchor),
            None if bars is None else bars[max(0, cursor - FAST + 1):cursor + 1],
        )
        anchor = str(equity)
        kc += spikes
        if step >= WARMUP:
            sides.append(side)
            executed = account.apply(side, bid, ask)
        else:
            # Warm-up runs the network but never the account, so the fly is
            # judged from a charged mushroom body and a full starting balance.
            executed = None
            account.cash, account.base = account.start, 0.0
        history.append(price)
        cursor += 1
    final = account.equity(quotes(prices[min(cursor, len(prices)) - 1])[0])
    return {
        "profit": float(final - account.start),
        "final_equity": float(final),
        "observations": len(sides),
        "buy": sides.count("BUY"),
        "sell": sides.count("SELL"),
        "hold": sides.count("HOLD"),
        "fills": dict(account.fills),
        "rejected": account.rejected,
        "kc_spikes": int(kc),
    }


def neural_proposal(controller):
    def propose(frame, kind, history, executed, balance, bars=None):
        n = controller.observe(frame, kind, history, executed, balance,
                               bars)
        return n["side"], n["KC_spikes"]

    return propose


def buy_and_hold(prices, start, observations, settings):
    """Deploy the whole budget as fast as the order size allows, then hold.

    Bound by the same account rules as every fly -- same capital, same order
    limit, same fee, same window -- so it is a fair benchmark rather than an
    idealised index. It takes no brain and no rendered chart, so it is cheap
    enough to recompute per evaluation instead of being cached and mismatched.
    """
    account = Account(settings.capital, settings.order_limit, settings.paper_fee)
    window = prices[start + CHART_WINDOW + WARMUP:][:observations]
    if not window:
        return 0.0
    for price in window:
        bid, ask = quotes(price)
        account.apply("BUY", bid, ask)
    return float(account.equity(quotes(window[-1])[0]) - account.start)


def evaluate(controller, pristine, genome, prices, start, observations, settings):
    """One genome, one chronological start. Profit, and profit over benchmark."""
    with configured(controller, genome, pristine):
        account = Account(settings.capital, settings.order_limit, settings.paper_fee)
        row = replay(
            controller, prices, start, observations,
            neural_proposal(controller), account,
        )
    # Both are reported. `profit` is what the kill criterion compares against
    # the baselines; `excess` is what the search is actually ranked on, and the
    # gap between them is how much of a result was the market rather than the
    # fly.
    row["buy_and_hold"] = buy_and_hold(prices, start, observations, settings)
    row["excess"] = row["profit"] - row["buy_and_hold"]
    return row


# A fly proposing one side at or above this fraction of observations is not
# deciding; the risk guard is. Declared as a fraction rather than "all of
# them" because the measured failure sat at 96%: the exact-100% test passed it.
ONE_SIDED = 0.9


def degenerate(row):
    """A fly that cannot act is not a candidate, however its profit looks.

    Screened before the expensive evaluation so a genome that proposes one
    thing forever, or whose mushroom body never fires, does not consume it.

    Filling nothing is disqualifying too. Against an all-cash baseline of
    exactly zero, a fly that never trades ties for first on any segment where
    trading loses money, and the first real generation of the fixture run
    collapsed the whole population onto exactly that. Sitting in cash is a
    legitimate strategy and it is already represented -- by the baseline. What
    is being searched for here is a fly that trades.

    Near-total one-sidedness counts as not acting. The first champion on real
    candles proposed BUY at 96% of observations and spent the rest of each run
    having those rejected for want of budget: it had bought everything it could
    afford and was holding, which is a baseline, not a policy. The original
    test asked for one proposal at *every* observation and let that through.
    """
    if row["observations"] == 0:
        return "no observations"
    share = max(row["buy"], row["sell"], row["hold"]) / row["observations"]
    if share >= ONE_SIDED:
        return f"one proposal for {share:.0%} of observations"
    if row["kc_spikes"] == 0:
        return "silent mushroom body"
    if not row["fills"]["BUY"] and not row["fills"]["SELL"]:
        return "never filled an order"
    return None


# --- baselines -------------------------------------------------------------


def fixed_proposal(side):
    return lambda *_, **__: (side, 0)


def random_proposal(rng):
    return lambda *_, **__: (rng.choice(["BUY", "SELL", "HOLD"]), 0)


def baselines(controller, pristine, prices, start, observations, settings, rng):
    """The four comparisons declared before any evolution was run.

    A champion that does not beat all four out of sample is noise, and that is
    written down here rather than decided afterwards.
    """
    def account():
        return Account(settings.capital, settings.order_limit, settings.paper_fee)

    out = {}
    for name, propose in [
        ("buy_and_hold", fixed_proposal("BUY")),
        ("all_cash", fixed_proposal("HOLD")),
        ("random", random_proposal(rng)),
    ]:
        out[name] = replay(controller, prices, start, observations, propose, account())
    out["wild_type"] = evaluate(
        controller, pristine, WILD_TYPE, prices, start, observations, settings
    )
    return out


def build(settings=None):
    controller = FlyController(settings or Settings())
    return controller, pristine_inhibitory(controller.brain)


def graph_snapshot(brain):
    """Copies of the arrays a genome must never change.

    About 210 MB on top of a 0.6 GB worker, which buys an exact check instead
    of a sampled one. A genome that leaked into the graph would contaminate
    every later genome the worker sees, and the contamination would look like
    evolution working.
    """
    return brain.ptr.copy(), brain.post.copy(), brain.weight.copy()


def graph_unchanged(brain, snapshot):
    return all(
        np.array_equal(before, after)
        for before, after in zip(snapshot, (brain.ptr, brain.post, brain.weight))
    )


def ceiling(prices, start, observations):
    """Profit a trader with perfect foresight could take from this window.

    Not a baseline -- nothing can beat perfect foresight, and it is not there
    to be beaten. It is there because if the ceiling is at or below zero, no
    policy of any kind can profit on this segment at this sampling interval,
    and evolving one is a waste of a night.

    Derivation: with a final bid B, a BUY at step i moves final equity by
    order x (B / ask_i - (1 + fee)) and a SELL by order x ((1 - fee) - B /
    bid_i). Those contributions are independent of each other, so the best
    achievable is the sum of the positive ones -- taking the better of the two
    actions at each step and ignoring budget and inventory, which can only
    make the bound looser and therefore keeps it a true ceiling.
    """
    window = prices[start + CHART_WINDOW: start + CHART_WINDOW + WARMUP + observations]
    traded = window[WARMUP:]
    if len(traded) < 2:
        return {"ceiling": 0.0, "observations": len(traded)}
    settings = Settings()
    order, fee = float(settings.order_limit), float(settings.paper_fee)
    final = quotes(traded[-1])[0]
    best, buys, sells = 0.0, 0, 0
    for price in traded[:-1]:
        bid, ask = quotes(price)
        gain_buy = order * (final / ask - (1 + fee))
        gain_sell = order * ((1 - fee) - final / bid)
        step = max(0.0, gain_buy, gain_sell)
        best += step
        buys += gain_buy > 0 and gain_buy >= gain_sell
        sells += gain_sell > 0 and gain_sell > gain_buy
    return {
        "ceiling": float(best),
        "observations": len(traded),
        "profitable_buys": int(buys),
        "profitable_sells": int(sells),
        "price_range_percent": float(
            (max(traded) - min(traded)) / min(traded) * 100
        ),
        "round_trip_cost_percent": float(2 * fee * 100),
    }
