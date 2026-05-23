"""Test PositionManager: position tracking, decision rules, SQLite persistence."""

import os
import tempfile

import pytest

from src.agent.position_manager import Decision, Position, PositionManager


@pytest.fixture
def tmp_db(tmp_path):
    return str(tmp_path / "test_position.db")


@pytest.fixture
def manager(tmp_db):
    return PositionManager(db_path=tmp_db)


def _position(ticker="AAPL", entry_date="2026-01-01", entry_price=150.0, entry_reason="agent_signal", days_held=0):
    return Position(ticker=ticker, entry_date=entry_date, entry_price=entry_price, entry_reason=entry_reason, days_held=days_held)


def _signal(ticker="AAPL", confidence=0.8, reasoning="test", signal_strength=1.0):
    return {"ticker": ticker, "confidence": confidence, "reasoning": reasoning, "signal_strength": signal_strength}


class TestPositionManager:

    def test_get_current_position_none_if_empty(self, manager):
        """Returns None if no position held."""
        assert manager.get_current_position() is None

    def test_should_hold_current_returns_false_if_no_position(self, manager):
        """Always return False if current_position is None."""
        result = manager.should_hold_current(None, _signal())
        assert result is False

    def test_should_hold_current_hold_if_same_ticker_high_confidence(self, manager):
        """HOLD (True) if same ticker + confidence > 70%."""
        pos = _position(ticker="AAPL")
        result = manager.should_hold_current(pos, _signal(ticker="AAPL", confidence=0.75))
        assert result is True

    def test_should_hold_current_switch_if_new_very_high_confidence(self, manager):
        """SWITCH (False) if different ticker + confidence > 80%."""
        pos = _position(ticker="AAPL")
        result = manager.should_hold_current(pos, _signal(ticker="MSFT", confidence=0.85))
        assert result is False

    def test_should_hold_current_exit_if_position_too_old(self, manager):
        """EXIT (False) if position > 5 days old."""
        pos = _position(ticker="AAPL", days_held=6)
        result = manager.should_hold_current(pos, _signal(ticker="AAPL", confidence=0.75))
        assert result is False

    def test_should_hold_current_hold_if_new_signal_weak(self, manager):
        """HOLD (True) if new signal < 50% confidence."""
        pos = _position(ticker="AAPL")
        result = manager.should_hold_current(pos, _signal(ticker="MSFT", confidence=0.30))
        assert result is True

    def test_execute_decision_buy_if_no_position_strong_signal(self, manager):
        """action=BUY if no position + confidence > 50%."""
        decision = manager.execute_decision(None, _signal(ticker="AAPL", confidence=0.7), {})
        assert decision.action == "BUY"
        assert decision.new_ticker == "AAPL"
        assert decision.current_ticker is None

    def test_execute_decision_skip_if_no_position_weak_signal(self, manager):
        """action=SKIP if no position + confidence < 50%."""
        decision = manager.execute_decision(None, _signal(ticker="AAPL", confidence=0.3), {})
        assert decision.action == "SKIP"
        assert decision.new_ticker is None

    def test_execute_decision_hold_same_ticker(self, manager):
        """action=HOLD if current position + same ticker + high confidence."""
        pos = _position(ticker="AAPL", days_held=1)
        decision = manager.execute_decision(pos, _signal(ticker="AAPL", confidence=0.75), {})
        assert decision.action == "HOLD"
        assert decision.current_ticker == "AAPL"

    def test_execute_decision_switch_different_ticker(self, manager):
        """action=SWITCH if different ticker + confidence > 70%."""
        pos = _position(ticker="AAPL", days_held=1)
        decision = manager.execute_decision(pos, _signal(ticker="MSFT", confidence=0.85), {})
        assert decision.action == "SWITCH"
        assert decision.current_ticker == "AAPL"
        assert decision.new_ticker == "MSFT"

    def test_execute_decision_exit_position_too_old(self, manager):
        """action=EXIT if position > 5 days old."""
        pos = _position(ticker="AAPL", days_held=6)
        decision = manager.execute_decision(pos, _signal(ticker="AAPL", confidence=0.75), {})
        assert decision.action == "EXIT"
        assert decision.current_ticker == "AAPL"
        assert decision.new_ticker is None

    def test_store_decision_persists_to_db(self, manager):
        """decision_history record written, retrievable via get_decision_history()."""
        decision = manager.execute_decision(None, _signal(ticker="AAPL", confidence=0.7), {})
        manager.store_decision(decision, entry_price=150.0)

        history = manager.get_decision_history()
        assert len(history) == 1
        assert history[0].action == "BUY"
        assert history[0].new_ticker == "AAPL"

    def test_store_decision_buy_creates_current_position(self, manager):
        """BUY decision creates a current_position row."""
        decision = manager.execute_decision(None, _signal(ticker="AAPL", confidence=0.7), {})
        manager.store_decision(decision, entry_price=150.0)

        pos = manager.get_current_position()
        assert pos is not None
        assert pos.ticker == "AAPL"
        assert pos.entry_price == 150.0

    def test_should_hold_current_default_hold(self, manager):
        """Default HOLD when same ticker + confidence between 0.5 and 0.7."""
        pos = _position(ticker="AAPL", days_held=1)
        # Confidence 0.6: not weak (<0.5), not strong same-ticker (>0.7), no switch
        result = manager.should_hold_current(pos, _signal(ticker="AAPL", confidence=0.6))
        assert result is True

    def test_store_decision_exit_writes_exit_history(self, manager):
        """EXIT decision with existing position writes to exit_history."""
        # First buy AAPL
        buy_dec = manager.execute_decision(None, _signal(ticker="AAPL", confidence=0.7), {})
        manager.store_decision(buy_dec, entry_price=150.0)

        # Now exit it (aged position)
        import sqlite3
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(manager.db_path) as conn:
            conn.execute("DELETE FROM current_position")
            conn.execute(
                "INSERT INTO current_position (ticker, entry_date, entry_price, entry_reason, days_held, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                ("AAPL", "2026-01-01", 150.0, "agent_signal", 6, now),
            )

        pos = manager.get_current_position()
        exit_dec = manager.execute_decision(pos, _signal(ticker="AAPL", confidence=0.75), {})
        assert exit_dec.action == "EXIT"
        manager.store_decision(exit_dec, entry_price=155.0)

        # Check exit_history was written
        with sqlite3.connect(manager.db_path) as conn:
            rows = conn.execute("SELECT * FROM exit_history").fetchall()
        assert len(rows) == 1
        assert rows[0][1] == "AAPL"  # ticker

    def test_decision_to_dict_serialization(self, manager):
        """Decision.to_dict() returns all expected keys."""
        decision = manager.execute_decision(None, _signal(ticker="AAPL", confidence=0.7), {})
        d = decision.to_dict()
        expected_keys = {"timestamp", "action", "current_ticker", "new_ticker", "reasoning", "confidence", "agent_signal_strength"}
        assert expected_keys == set(d.keys())
        assert d["action"] == "BUY"

    def test_store_decision_hold_increments_days_held(self, manager):
        """HOLD decision increments days_held in current_position."""
        # First buy
        buy_decision = manager.execute_decision(None, _signal(ticker="AAPL", confidence=0.7), {})
        manager.store_decision(buy_decision, entry_price=150.0)

        # Then hold
        pos = manager.get_current_position()
        hold_decision = manager.execute_decision(pos, _signal(ticker="AAPL", confidence=0.75), {})
        manager.store_decision(hold_decision, entry_price=150.0)

        updated_pos = manager.get_current_position()
        assert updated_pos is not None
        assert updated_pos.days_held == 1
        assert updated_pos.entry_reason == "hold_from_yesterday"
