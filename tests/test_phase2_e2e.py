"""E2E tests: Phase 2 position lifecycle (BUY → HOLD → EXIT)."""

import json
import sqlite3
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.agent.agent_graph import AgentState, position_management_node
from src.agent.position_manager import Decision, Position, PositionManager


def _make_state(asset_search_results=None, should_continue=True):
    state: AgentState = {
        "ohlcv_data": {"AAPL": pd.DataFrame(), "MSFT": pd.DataFrame()},
        "features_df": pd.DataFrame(),
        "context": None,
        "proposals": [],
        "results": [],
        "best_result": None,
        "iteration": 0,
        "max_iterations": 3,
        "strategy_type": "momentum",
        "asset": "AAPL",
        "should_continue": should_continue,
        "memory_context": "",
        "run_log": [],
        "regime_signals": None,
        "asset_search_results": asset_search_results or [],
        "strategies_ranked": [],
        "llm_decisions": {},
        "validated_edges": [],
        "current_position": None,
        "position_decision": {},
    }
    return state


def _asset(ticker, score=0.8):
    return {"ticker": ticker, "score": score, "reasoning": "strong momentum", "avg_historical_return": 150.0}


class TestPhase2E2E:

    def test_e2e_day1_no_position_buys_new(self, tmp_path):
        """Day 1 with no position → agent runs → BUY decision."""
        db_path = str(tmp_path / "pm.db")
        pm = PositionManager(db_path=db_path)

        state = _make_state(asset_search_results=[_asset("AAPL", score=0.8)])

        with patch("src.agent.agent_graph.PositionManager", return_value=pm):
            result = position_management_node(state)

        assert result["position_decision"]["action"] == "BUY"
        assert result["position_decision"]["new_ticker"] == "AAPL"
        assert result["should_continue"] is True

        # Verify DB has current_position
        pos = pm.get_current_position()
        assert pos is not None
        assert pos.ticker == "AAPL"

    def test_e2e_day2_holds_same_ticker(self, tmp_path):
        """Day 2 with AAPL position + AAPL signal → HOLD."""
        db_path = str(tmp_path / "pm.db")
        pm = PositionManager(db_path=db_path)

        # Day 1: BUY AAPL
        buy_dec = Decision(
            timestamp=datetime.now(timezone.utc).isoformat(),
            action="BUY",
            current_ticker=None,
            new_ticker="AAPL",
            reasoning="Day 1 buy",
            confidence=0.8,
            agent_signal_strength=1.0,
        )
        pm.store_decision(buy_dec, entry_price=150.0)

        # Day 2: Same signal for AAPL
        state = _make_state(asset_search_results=[_asset("AAPL", score=0.8)])

        with patch("src.agent.agent_graph.PositionManager", return_value=pm):
            result = position_management_node(state)

        assert result["position_decision"]["action"] == "HOLD"
        assert result["position_decision"]["current_ticker"] == "AAPL"
        assert result["should_continue"] is True

        # Verify days_held incremented
        pos = pm.get_current_position()
        assert pos is not None
        assert pos.days_held == 1

    def test_e2e_position_lifecycle_buy_hold_exit(self, tmp_path):
        """3-day position lifecycle: Day1=BUY, Day2=HOLD, Day3=EXIT."""
        db_path = str(tmp_path / "pm.db")
        pm = PositionManager(db_path=db_path)

        # Day 1: BUY
        state1 = _make_state(asset_search_results=[_asset("AAPL", score=0.8)])
        with patch("src.agent.agent_graph.PositionManager", return_value=pm):
            result1 = position_management_node(state1)
        assert result1["position_decision"]["action"] == "BUY"

        # Day 2: HOLD (same ticker, strong signal)
        state2 = _make_state(asset_search_results=[_asset("AAPL", score=0.8)])
        with patch("src.agent.agent_graph.PositionManager", return_value=pm):
            result2 = position_management_node(state2)
        assert result2["position_decision"]["action"] == "HOLD"

        # Day 3: Force old position (days_held=6) and weak signal → EXIT
        with sqlite3.connect(db_path) as conn:
            conn.execute("DELETE FROM current_position")
            now = datetime.now(timezone.utc).isoformat()
            conn.execute(
                "INSERT INTO current_position (ticker, entry_date, entry_price, entry_reason, days_held, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                ("AAPL", "2026-01-01", 150.0, "hold_from_yesterday", 6, now),
            )

        state3 = _make_state(asset_search_results=[_asset("AAPL", score=0.3)])
        with patch("src.agent.agent_graph.PositionManager", return_value=pm):
            result3 = position_management_node(state3)
        assert result3["position_decision"]["action"] == "EXIT"
        assert result3["should_continue"] is False

        # Verify all 3 decisions in history
        history = pm.get_decision_history(limit=30)
        actions = [d.action for d in history]
        assert "BUY" in actions
        assert "HOLD" in actions
        assert "EXIT" in actions
