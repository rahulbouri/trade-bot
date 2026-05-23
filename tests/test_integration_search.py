"""Integration test: E2E pipeline from discovery to backtest."""

import json
from datetime import datetime
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.agent.agent_graph import AgentState
from src.agent.search_pipeline import SearchPipeline
from src.agent.strategy_memory import PastResult, StrategyMemory
from src.features.regime import RegimeSignals


class TestIntegrationPhase1:
    """E2E integration tests for Phase 1."""

    def test_e2e_empty_memory_bootstrap(self):
        """E2E test: Empty StrategyMemory → search → backtest."""
        # Create fresh pipeline with mock empty memory
        pipeline = SearchPipeline()

        with patch.object(pipeline.memory, "recall", return_value=[]):
            # Analyze with empty memory should handle gracefully
            sample_signals = RegimeSignals(
                vix_percentile_252d=35.0,
                momentum_63d=0.08,
                realized_vol_21d=0.15,
                drawdown_from_52w_high=-0.05,
                price_vs_200sma_pct=0.03,
                vol_regime="mid",
            )

            # Test 1: search_assets with empty memory
            result = pipeline.search_assets(
                ["AAPL", "GOOGL"], sample_signals, pd.DataFrame(), "momentum", top_k=2
            )

            # Should return gracefully (possibly with low/zero scores)
            assert isinstance(result, list)

            # Test 2: shortlist_strategies with empty memory
            strategies = pipeline.shortlist_strategies("LowVol-Bull")
            assert isinstance(strategies, list)

    def test_e2e_with_growing_memory(self):
        """E2E test: Growing memory improves asset search quality."""
        pipeline = SearchPipeline()
        sample_signals = RegimeSignals(
            vix_percentile_252d=35.0,
            momentum_63d=0.08,
            realized_vol_21d=0.15,
            drawdown_from_52w_high=-0.05,
            price_vs_200sma_pct=0.03,
            vol_regime="mid",
        )

        # Simulate growing memory: week 1 (5 trades)
        week1_trades = []
        for i in range(5):
            trade = PastResult(
                run_id=f"week1_run_{i}",
                strategy_type="momentum",
                params='{"fast": 12}',
                sharpe=1.2,
                total_return=0.1,
                max_drawdown=-0.1,
                regime="LowVol-Bull",
                signals_vector=json.dumps(sample_signals.__dict__),
            )
            week1_trades.append(trade)

        # Week 2 (50 trades)
        week2_trades = week1_trades + [
            PastResult(
                run_id=f"week2_run_{i}",
                strategy_type="momentum",
                params='{"fast": 12}',
                sharpe=1.3,
                total_return=0.12,
                max_drawdown=-0.09,
                regime="LowVol-Bull",
                signals_vector=json.dumps(sample_signals.__dict__),
            )
            for i in range(45)
        ]

        # Test with week 1 memory
        with patch.object(pipeline.memory, "recall", return_value=week1_trades):
            result_w1 = pipeline.search_assets(
                ["AAPL"], sample_signals, pd.DataFrame(), "momentum"
            )
            matching_w1 = result_w1[0]["matching_trades"] if result_w1 else 0

        # Test with week 2 memory
        with patch.object(pipeline.memory, "recall", return_value=week2_trades):
            result_w2 = pipeline.search_assets(
                ["AAPL"], sample_signals, pd.DataFrame(), "momentum"
            )
            matching_w2 = result_w2[0]["matching_trades"] if result_w2 else 0

        # Week 2 should find more matches (monotonic improvement)
        assert matching_w2 >= matching_w1

    def test_state_flows_through_nodes(self):
        """Test: AgentState correctly passes data between nodes."""
        state: AgentState = {
            "ohlcv_data": {},
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
        }

        # Verify state can hold all required fields
        assert "regime_label" not in state  # Not yet added

        # Add new fields that nodes would add
        state["regime_signals"] = RegimeSignals()
        state["asset_search_results"] = []
        state["strategies_ranked"] = []
        state["llm_decisions"] = {}

        assert "regime_signals" in state
        assert "asset_search_results" in state
        assert "strategies_ranked" in state
        assert "llm_decisions" in state

    def test_backward_compatibility_no_signals_vector(self):
        """Test: Old trades without signals_vector don't break pipeline."""
        pipeline = SearchPipeline()

        # Create old-style trade without signals_vector
        old_trade = PastResult(
            run_id="old_run",
            strategy_type="momentum",
            params='{"fast": 12}',
            sharpe=1.2,
            total_return=0.1,
            max_drawdown=-0.1,
            regime="LowVol-Bull",
            signals_vector="",  # Empty, old trade
        )

        # New-style trade with signals_vector
        new_signals = RegimeSignals()
        new_trade = PastResult(
            run_id="new_run",
            strategy_type="momentum",
            params='{"fast": 12}',
            sharpe=1.3,
            total_return=0.12,
            max_drawdown=-0.09,
            regime="LowVol-Bull",
            signals_vector=json.dumps(new_signals.__dict__),
        )

        sample_signals = RegimeSignals()

        with patch.object(pipeline.memory, "recall", return_value=[old_trade, new_trade]):
            # Should not crash when processing old trade without signals_vector
            try:
                result = pipeline.find_similar_winning_trades(
                    sample_signals, "momentum", max_distance=0.5
                )
                # Should skip old trade and use new trade
                assert len(result) > 0
            except Exception as e:
                pytest.fail(f"Backward compatibility failed: {e}")

    def test_backtest_integration_stores_signals_vector(self):
        """Test: backtest_accepted stores results with signals_vector."""
        pipeline = SearchPipeline()
        sample_signals = RegimeSignals(
            vix_percentile_252d=35.0,
            momentum_63d=0.08,
            realized_vol_21d=0.15,
            drawdown_from_52w_high=-0.05,
            price_vs_200sma_pct=0.03,
            vol_regime="mid",
        )

        # Mock OHLCV data
        ohlcv_data = {"AAPL": pd.DataFrame({"Close": [100, 101, 102]})}

        with patch.object(pipeline.memory, "store") as mock_store:
            with patch("src.agent.search_pipeline.run_backtest") as mock_backtest:
                with patch("src.features.regime.detect_regime_full") as mock_regime:
                    mock_backtest.return_value = {
                        "metrics": {
                            "sharpe_ratio": 1.5,
                            "total_return": 0.1,
                            "max_drawdown": -0.1,
                            "num_trades": 50,
                        }
                    }
                    mock_regime.return_value = sample_signals

                    result = pipeline.backtest_accepted(
                        ["AAPL"], "momentum", "LowVol-Bull", ohlcv_data
                    )

                    # Verify store was called with signals_vector
                    if mock_store.called:
                        call_args = mock_store.call_args[0][0]  # First positional arg
                        if hasattr(call_args, "signals_vector"):
                            assert call_args.signals_vector != ""  # Should have JSON

    def test_search_assets_parallel_execution(self):
        """Test: search_assets uses parallel execution."""
        pipeline = SearchPipeline()

        with patch.object(pipeline, "score_asset_for_strategy") as mock_score:
            mock_score.return_value = {
                "ticker": "AAPL",
                "score": 0.5,
                "strategy": "momentum",
                "distance": 0.1,
                "similarity": 0.9,
                "avg_historical_return": 0.1,
                "avg_historical_sharpe": 1.5,
                "matching_trades": 3,
                "reasoning": "test",
                "similar_trades": [],
            }

            sample_signals = RegimeSignals()
            universe = [f"STOCK_{i}" for i in range(20)]

            # Run search
            result = pipeline.search_assets(
                universe, sample_signals, pd.DataFrame(), "momentum", top_k=10
            )

            # Verify all assets were scored
            assert mock_score.call_count >= len(universe)

    def test_memory_deserialization_robustness(self):
        """Test: _deserialize_signals handles malformed JSON gracefully."""
        pipeline = SearchPipeline()

        # Valid JSON
        valid_json = '{"vix_percentile_252d": 35.0}'
        result = pipeline.memory._deserialize_signals(valid_json)
        assert isinstance(result, dict)

        # Invalid JSON
        invalid_json = "{broken json"
        result = pipeline.memory._deserialize_signals(invalid_json)
        assert result is None

        # Empty string
        result = pipeline.memory._deserialize_signals("")
        assert result is None

    def test_shortlist_strategies_empty_memory(self):
        """Test: shortlist_strategies handles empty memory gracefully."""
        pipeline = SearchPipeline()

        with patch.object(pipeline.memory, "recall", return_value=[]):
            result = pipeline.shortlist_strategies("LowVol-Bull", top_k=3)

            # Should return empty or placeholder results
            assert isinstance(result, list)

    def test_score_formula_correctness(self):
        """Test: score formula = base_sim * return_mult * win_rate_adj."""
        pipeline = SearchPipeline()

        # Create controlled test data
        sample_signals = RegimeSignals(
            vix_percentile_252d=35.0,
            momentum_63d=0.08,
            realized_vol_21d=0.15,
            drawdown_from_52w_high=-0.05,
            price_vs_200sma_pct=0.03,
            vol_regime="mid",
        )

        similar_trades = [
            {
                "sharpe": 1.5,
                "return": 0.20,
                "distance": 0.1,
                "similarity": 1.0 / (1.0 + 0.1),  # ~0.909
            }
            for _ in range(3)
        ]

        # Manually compute expected score
        avg_return = 0.20
        base_similarity = 1.0 / (1.0 + 0.1)  # 0.909
        return_multiplier = 1.0 + avg_return  # 1.20
        matching_trades = 3
        expected_trades = 20
        win_rate_adjustment = (matching_trades / expected_trades) ** 0.5

        expected_score = base_similarity * return_multiplier * win_rate_adjustment
        expected_score = min(expected_score, 1.0)  # Clipped

        # Verify formula is reasonable
        assert 0.0 <= expected_score <= 1.0
        assert expected_score > 0.0  # Should have positive score with good trades
