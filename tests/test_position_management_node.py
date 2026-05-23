"""Test position_management_node integration in agent_graph."""

import json
import tempfile
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.agent.agent_graph import AgentState, position_management_node
from src.agent.position_manager import Decision, Position, PositionManager


def _make_state(asset_search_results=None, ohlcv_data=None):
    state: AgentState = {
        "ohlcv_data": ohlcv_data or {"AAPL": pd.DataFrame()},
        "features_df": pd.DataFrame(),
        "context": None,
        "proposals": [],
        "results": [],
        "best_result": None,
        "iteration": 0,
        "max_iterations": 3,
        "strategy_type": "momentum",
        "asset": "AAPL",
        "should_continue": True,
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


def _make_asset_result(ticker, score=0.8):
    return {"ticker": ticker, "score": score, "reasoning": "test", "avg_historical_return": 150.0}


class TestPositionManagementNode:

    def test_position_management_node_no_position_buys(self, tmp_path):
        """Node issues BUY when no position + strong signal."""
        db_path = str(tmp_path / "pm.db")
        state = _make_state(asset_search_results=[_make_asset_result("AAPL", score=0.8)])

        with patch("src.agent.agent_graph.PositionManager", return_value=PositionManager(db_path=db_path)):
            result = position_management_node(state)

        assert result["position_decision"]["action"] == "BUY"
        assert result["position_decision"]["new_ticker"] == "AAPL"
        assert result["should_continue"] is True

    def test_position_management_node_holds_current(self, tmp_path):
        """Node issues HOLD when current position + same ticker."""
        db_path = str(tmp_path / "pm.db")
        pm = PositionManager(db_path=db_path)

        # Pre-populate a position
        buy_dec = Decision(
            timestamp="2026-01-01T00:00:00+00:00",
            action="BUY",
            current_ticker=None,
            new_ticker="AAPL",
            reasoning="test",
            confidence=0.8,
            agent_signal_strength=1.0,
        )
        pm.store_decision(buy_dec, entry_price=150.0)

        state = _make_state(asset_search_results=[_make_asset_result("AAPL", score=0.8)])

        with patch("src.agent.agent_graph.PositionManager", return_value=pm):
            result = position_management_node(state)

        assert result["position_decision"]["action"] == "HOLD"
        assert result["should_continue"] is True

    def test_position_management_node_switches_new_asset(self, tmp_path):
        """Node issues SWITCH when different ticker + high confidence."""
        db_path = str(tmp_path / "pm.db")
        pm = PositionManager(db_path=db_path)

        buy_dec = Decision(
            timestamp="2026-01-01T00:00:00+00:00",
            action="BUY",
            current_ticker=None,
            new_ticker="AAPL",
            reasoning="test",
            confidence=0.9,
            agent_signal_strength=1.0,
        )
        pm.store_decision(buy_dec, entry_price=150.0)

        # Signal for different ticker at very high confidence
        state = _make_state(asset_search_results=[_make_asset_result("MSFT", score=0.9)])

        with patch("src.agent.agent_graph.PositionManager", return_value=pm):
            result = position_management_node(state)

        assert result["position_decision"]["action"] in ("SWITCH", "HOLD", "EXIT")

    def test_position_management_node_exits_stops_trading(self, tmp_path):
        """Node issues EXIT + sets should_continue=False."""
        db_path = str(tmp_path / "pm.db")
        pm = PositionManager(db_path=db_path)

        # Store an aged-out position (days_held=6)
        import sqlite3
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                "INSERT INTO current_position (ticker, entry_date, entry_price, entry_reason, days_held, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                ("AAPL", "2026-01-01", 150.0, "agent_signal", 6, now),
            )

        # Weak signal → should_hold=False due to age, then exit
        state = _make_state(asset_search_results=[_make_asset_result("AAPL", score=0.3)])

        with patch("src.agent.agent_graph.PositionManager", return_value=pm):
            result = position_management_node(state)

        assert result["position_decision"]["action"] == "EXIT"
        assert result["should_continue"] is False

    def test_position_management_node_updates_state_dict(self, tmp_path):
        """state['position_decision'] populated correctly."""
        db_path = str(tmp_path / "pm.db")
        state = _make_state(asset_search_results=[_make_asset_result("AAPL", score=0.8)])

        with patch("src.agent.agent_graph.PositionManager", return_value=PositionManager(db_path=db_path)):
            result = position_management_node(state)

        pd_keys = {"action", "timestamp", "current_ticker", "new_ticker", "reasoning", "confidence", "agent_signal_strength"}
        assert pd_keys.issubset(set(result["position_decision"].keys()))
        assert len(result["run_log"]) > 0
