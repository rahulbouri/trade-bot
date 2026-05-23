"""Tests for PaperTrader — 12 tests covering init, buy, sell, P&L."""

import pytest

from src.trading.paper_trader import PaperTrader, _COMMISSION_PCT, _MAX_POSITION_SIZE_PCT


@pytest.fixture
def trader(tmp_path):
    db = str(tmp_path / "test_paper.db")
    return PaperTrader(db_path=db, initial_capital=100_000.0)


# ── Init tests ────────────────────────────────────────────────────────────────

class TestPaperTraderInit:
    def test_initial_cash_equals_capital(self, trader):
        assert trader.get_cash() == 100_000.0

    def test_no_open_positions_on_init(self, trader):
        assert trader.get_all_open_positions() == []

    def test_portfolio_snapshot_on_empty(self, trader):
        snap = trader.get_portfolio_snapshot()
        assert snap["total_value"] == snap["initial_capital"]
        assert snap["invested_value"] == 0.0
        assert snap["open_positions"] == []


# ── Buy tests ─────────────────────────────────────────────────────────────────

class TestExecuteBuy:
    def test_buy_deducts_cash(self, trader):
        price = 1000.0
        result = trader.execute_buy("AAPL", price, "momentum", {"fast": 12}, 1.5, "Bull", "test")
        qty = result["quantity"]
        commission = price * qty * _COMMISSION_PCT
        expected_cash = 100_000.0 - (price * qty + commission)
        assert abs(trader.get_cash() - expected_cash) < 0.01

    def test_buy_creates_open_position(self, trader):
        trader.execute_buy("AAPL", 1000.0, "momentum", {"fast": 12}, 1.5, "Bull", "test")
        pos = trader.get_open_position("AAPL")
        assert pos is not None
        assert pos["ticker"] == "AAPL"

    def test_buy_respects_max_position_size(self, trader):
        price = 500.0
        result = trader.execute_buy("SPY", price, "momentum", {}, 1.2, "Bull", "test")
        max_qty = int((100_000.0 * _MAX_POSITION_SIZE_PCT) / price)
        assert result["quantity"] <= max_qty

    def test_buy_raises_if_already_open(self, trader):
        trader.execute_buy("AAPL", 1000.0, "momentum", {}, 1.5, "Bull", "test")
        with pytest.raises(ValueError, match="already open"):
            trader.execute_buy("AAPL", 1000.0, "momentum", {}, 1.5, "Bull", "test2")

    def test_buy_raises_if_price_zero(self, trader):
        with pytest.raises(ValueError):
            trader.execute_buy("AAPL", 0.0, "momentum", {}, 1.5, "Bull", "test")


# ── Sell tests ────────────────────────────────────────────────────────────────

class TestExecuteSell:
    def _buy(self, trader, ticker="AAPL", price=100.0, quantity_hint=10):
        """Helper: buy at a price that gives ~10 shares."""
        # Use a price that gives a known quantity
        return trader.execute_buy(ticker, price, "momentum", {"fast": 12}, 1.5, "Bull", "test")

    def test_sell_removes_open_position(self, trader):
        trader.execute_buy("AAPL", 1000.0, "momentum", {}, 1.5, "Bull", "test")
        trader.execute_sell("AAPL", 1100.0, "test exit")
        assert trader.get_open_position("AAPL") is None

    def test_sell_records_trade(self, trader):
        trader.execute_buy("AAPL", 1000.0, "momentum", {}, 1.5, "Bull", "test")
        trader.execute_sell("AAPL", 1100.0, "test exit")
        hist = trader.get_trade_history()
        assert len(hist) == 1
        assert hist.iloc[0]["ticker"] == "AAPL"

    def test_sell_raises_if_not_open(self, trader):
        with pytest.raises(ValueError, match="No open position"):
            trader.execute_sell("AAPL", 1000.0, "test")

    def test_pnl_calculation_correct(self, trader):
        """
        We need exactly 10 shares, so price must fit:
        floor(100_000 * 0.20 / price) = 10  →  price in (1000, 2000]
        Use price=1999.0 → qty = floor(20000 / 1999) = 10
        """
        price = 1999.0
        exit_price = 1999.0 + 100.0  # sell at 2099

        trader.execute_buy("TEST", price, "momentum", {}, 1.5, "Bull", "test")
        pos = trader.get_open_position("TEST")
        qty = pos["quantity"]
        assert qty == 10, f"Expected 10 shares but got {qty}"

        cost_basis = price * qty + (price * qty * _COMMISSION_PCT)
        exit_commission = exit_price * qty * _COMMISSION_PCT
        gross_proceeds = exit_price * qty - exit_commission
        expected_pnl = gross_proceeds - cost_basis
        expected_pnl_pct = expected_pnl / cost_basis

        trader.execute_sell("TEST", exit_price, "test exit")
        hist = trader.get_trade_history()
        row = hist.iloc[0]

        assert abs(row["pnl"] - expected_pnl) < 0.01
        assert abs(row["pnl_pct"] - expected_pnl_pct) < 0.0001
