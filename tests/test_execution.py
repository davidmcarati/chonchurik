"""No tests send orders to Coinbase. SDK calls here are in-memory doubles."""

import dataclasses
import time

import pytest
from pydantic import ValidationError

from stonkfly.actions import StonkflyActions
from stonkfly.broker import CoinbaseBroker, PaperBroker, UnresolvedOrder
from stonkfly.config import D, Settings
from stonkfly.ledger import Ledger
from stonkfly.market import Quote
from stonkfly.reinforcement import reinforcement
from stonkfly.risk import Guard, Veto


def quote(**changes):
    q = Quote(
        "BTC-USDC",
        D("100"),
        D("100.1"),
        time.time(),
        D(".00000001"),
        D(".01"),
        D(".01"),
        D("1"),
        D(".00000001"),
    )
    return dataclasses.replace(q, **changes)


@pytest.fixture
def env(tmp_path):
    s = Settings()
    ledger = Ledger(tmp_path / "ledger.sqlite", s, "paper")
    guard = Guard(s, ledger, tmp_path / "STOP")
    yield s, ledger, guard
    ledger.close()


@pytest.mark.parametrize("value", ["NaN", "Infinity", "-Infinity", True])
def test_nonfinite_money(value):
    with pytest.raises(ValueError):
        D(value)


@pytest.mark.parametrize(
    "changes",
    [
        dict(capital="101"),
        dict(order_limit="11"),
        dict(neural_bin_ms=0.01),
        dict(reward_deadband="0"),
        dict(interval_seconds=float("nan")),
        dict(daily_orders=1.5),
    ],
)
def test_configuration_bounds(changes):
    with pytest.raises(ValueError):
        Settings(**changes)


@pytest.mark.parametrize(
    "changes",
    [
        dict(timestamp=time.time() - 100),
        dict(timestamp=time.time() + 100),
        dict(ask=D("102")),
        dict(product="ETH-USDC"),
    ],
)
def test_quote_veto(env, changes):
    _, _, g = env
    with pytest.raises(Veto):
        g.plan("BTC-USDC", "BUY", {"BTC-USDC": quote(**changes)})


def test_money_limits_and_no_short(env):
    _, l, g = env
    quotes = {"BTC-USDC": quote()}
    p = g.plan("BTC-USDC", "BUY", quotes)
    assert D(p["base_size"]) * D(p["limit_price"]) + D(p["fee_ceiling"]) <= D("10")
    with pytest.raises(Veto):
        g.plan("BTC-USDC", "SELL", quotes)
    l.put("cash", ".1")
    with pytest.raises(Veto):
        g.plan("BTC-USDC", "BUY", quotes)


def test_stop_loss_and_external_stop(env):
    _, l, g = env
    g.stop_file.touch()
    with pytest.raises(Veto):
        g.plan("BTC-USDC", "BUY", {"BTC-USDC": quote()})
    g.stop_file.unlink()
    l.put("cash", "79")
    with pytest.raises(Veto):
        g.check({"BTC-USDC": quote()}, time.time())
    assert "Loss stop" in l.get("halted")


def test_agentkit_paper_accounting_and_cooldown(env):
    s, l, g = env
    provider = StonkflyActions(g, PaperBroker(s, l))
    provider.quotes = {"BTC-USDC": quote()}
    a = provider.get_actions()[0]
    r = a.invoke({"product": "BTC-USDC", "side": "BUY"})
    assert r["status"] == "FILLED"
    assert l.cash == D("100") - D(r["quote"]) - D(r["fee"])
    assert l.positions["BTC-USDC"] == D(r["base"])
    assert reinforcement(l.equity(provider.quotes), "100", ".01")[0] == "aversive"
    with pytest.raises(Veto):
        a.invoke({"product": "BTC-USDC", "side": "BUY"})
    with pytest.raises(ValidationError):
        a.invoke({"product": "BTC-USDC", "side": "BUY", "size": 99})


def test_daily_attempt_limit(env):
    _, l, g = env
    l.attempts_today = lambda now: 24
    with pytest.raises(Veto, match="Daily"):
        g.plan("BTC-USDC", "BUY", {"BTC-USDC": quote()})


def test_crash_recovery_and_exactly_once_paper(env):
    s, l, g = env
    p = l.reserve(g.plan("BTC-USDC", "BUY", {"BTC-USDC": quote()}), time.time())
    path = l.path
    other = Ledger(path, s, "paper")
    broker = PaperBroker(s, other)
    broker.reconcile()
    cash = other.cash
    broker.reconcile()
    assert other.cash == cash
    with pytest.raises(RuntimeError, match="changed"):
        other.settle(p["client_order_id"], "0", "0", "0")
    other.close()


def test_invalid_fill_and_fee_overrun(env):
    _, l, g = env
    p = l.reserve(g.plan("BTC-USDC", "BUY", {"BTC-USDC": quote()}), time.time())
    with pytest.raises(ValueError):
        l.settle(p["client_order_id"], 0, 1, 0)
    with pytest.raises(RuntimeError):
        l.settle(p["client_order_id"], 1, 100, 0)
    size = D(p["base_size"])
    l.settle(p["client_order_id"], size, size * 100, D(p["fee_ceiling"]) + D(".01"))
    assert "fee exceeded" in l.get("halted")
    assert not l.pending()  # Actual fill remains accounted for, despite the halt.


class SDK:
    """Models only documented Advanced REST calls; never opens a socket."""

    def __init__(self):
        self.cash = D(100)
        self.btc = D(0)
        self.orders = []
        self.submissions = 0
        self.fee = "0.01"
        self.warnings = []
        self.uncertain = False
        self.after_preview = lambda: None
        self.permissions = {
            "can_view": True,
            "can_trade": True,
            "can_transfer": False,
            "portfolio_uuid": "test-portfolio",
        }

    def get_api_key_permissions(self):
        return self.permissions

    def get_accounts(self, **kw):
        return {
            "accounts": [
                {
                    "currency": c,
                    "retail_portfolio_id": "test-portfolio",
                    "available_balance": {"value": str(v)},
                    "hold": {"value": "0"},
                }
                for c, v in [("USDC", self.cash), ("BTC", self.btc)]
            ],
            "has_next": False,
        }

    def list_orders(self, **kw):
        return {
            "orders": [] if kw.get("order_status") else self.orders,
            "has_next": False,
        }

    def preview_limit_order_fok(self, **kw):
        self.after_preview()
        return {"commission_total": self.fee, "errs": [], "warning": self.warnings}

    def limit_order_fok(self, **kw):
        self.submissions += 1
        assert kw["retail_portfolio_id"] == "test-portfolio"
        assert "leverage" not in kw and "margin_type" not in kw
        base = D(kw["base_size"])
        value = base * D("100.1")
        fee = D(self.fee)
        self.cash -= value + fee
        self.btc += base
        order = {
            "order_id": "order-1",
            "client_order_id": kw["client_order_id"],
            "product_id": kw["product_id"],
            "side": kw["side"],
            "status": "FILLED",
            "filled_size": str(base),
            "filled_value": str(value),
            "total_fees": str(fee),
        }
        self.orders.append(order)
        if self.uncertain:
            raise TimeoutError("Response lost after accepted order")
        return {"success": True, "success_response": {"order_id": "order-1"}}

    def get_order(self, oid):
        return {"order": next(o for o in self.orders if o["order_id"] == oid)}


def live_double(env):
    s, l, g = env
    sdk = SDK()
    broker = CoinbaseBroker(s, l, sdk, "test-portfolio")
    broker.preflight()
    provider = StonkflyActions(g, broker)
    provider.quotes = {"BTC-USDC": quote()}
    return sdk, broker, provider


def test_advanced_fok_and_reconciled_balances(env):
    sdk, broker, p = live_double(env)
    assert (
        p.get_actions()[0].invoke({"product": "BTC-USDC", "side": "BUY"})["status"]
        == "SETTLED"
    )
    assert sdk.submissions == 1
    broker.verify_balances()


@pytest.mark.parametrize("failure", ["expensive_fee", "warning", "stop", "stale"])
def test_preview_cannot_bypass_final_guard(env, failure):
    sdk, broker, p = live_double(env)
    if failure == "expensive_fee":
        sdk.fee = "2"
    if failure == "warning":
        sdk.warnings = ["unexpected exchange warning"]
    if failure == "stop":
        sdk.after_preview = env[2].stop_file.touch
    if failure == "stale":
        p.quotes = {"BTC-USDC": quote(timestamp=time.time() - 20)}
    with pytest.raises(Veto):
        p.invoke({"product": "BTC-USDC", "side": "BUY"})
    assert sdk.submissions == 0
    assert not env[1].pending()


def test_ambiguous_submission_reconciles_without_duplicate(env):
    sdk, broker, p = live_double(env)
    sdk.uncertain = True
    with pytest.raises(UnresolvedOrder):
        p.invoke({"product": "BTC-USDC", "side": "BUY"})
    assert env[1].pending()[0]["status"] == "UNKNOWN"
    with pytest.raises(Veto):
        p.invoke({"product": "BTC-USDC", "side": "BUY"})
    broker.reconcile()
    broker.verify_balances()
    broker.reconcile()
    assert sdk.submissions == 1 and not env[1].pending()


def test_unfound_submission_remains_stopped(env):
    sdk, broker, p = live_double(env)
    sdk.uncertain = True
    with pytest.raises(UnresolvedOrder):
        p.invoke({"product": "BTC-USDC", "side": "BUY"})
    sdk.orders = []
    with pytest.raises(UnresolvedOrder):
        broker.reconcile()
    assert sdk.submissions == 1 and env[1].pending()


@pytest.mark.parametrize(
    "change", ["transfer", "wrong_portfolio", "overfunded", "external_deposit"]
)
def test_portfolio_boundaries(env, change):
    s, l, _ = env
    sdk = SDK()
    broker = CoinbaseBroker(s, l, sdk, "test-portfolio")
    if change == "transfer":
        sdk.permissions["can_transfer"] = True
    if change == "wrong_portfolio":
        sdk.permissions["portfolio_uuid"] = "other"
    if change == "overfunded":
        sdk.cash = D(101)
    if change == "external_deposit":
        broker.preflight()
        sdk.cash += 1
    with pytest.raises(RuntimeError):
        broker.preflight()
    assert sdk.submissions == 0


def test_real_sdk_methods_exist():
    # Check actual pinned SDK signatures, alongside the isolated exchange double.
    import inspect

    from coinbase.rest import RESTClient

    for method in ["preview_limit_order_fok", "limit_order_fok"]:
        params = inspect.signature(getattr(RESTClient, method)).parameters
        assert {"base_size", "limit_price", "side", "retail_portfolio_id"} <= set(
            params
        )


def test_provenance_covers_every_source_suffix_present():
    """A source file in a suffix the gate does not hash is invisible to it.

    The gate exists so a resumed run refuses to continue against changed code.
    A CUDA kernel is as much the model as kernel.cpp is, and before `.cu` was
    listed it could have been swapped underneath a run without the provenance
    hash moving. This asserts the list keeps up with what is actually there.
    """
    from pathlib import Path

    from stonkfly.cli import SOURCE_SUFFIXES

    root = Path(__file__).resolve().parent.parent / "stonkfly"
    ignored = {".pyc", ".pyd", ".so", ".dll", ".json", ".npz", ".lib", ".exp",
               ".obj", ".pdb", ".partial", ""}
    present = {p.suffix for p in root.rglob("*")
               if p.is_file() and "__pycache__" not in p.parts}
    missed = present - set(SOURCE_SUFFIXES) - ignored
    assert not missed, f"source suffixes not hashed into provenance: {missed}"
