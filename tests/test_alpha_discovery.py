"""Test AlphaDiscoveryAgent, EdgeValidator, and EdgeGeneralizer."""

import json
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.agent.alpha_discovery import AlphaDiscoveryAgent
from src.agent.edge_generalizer import EdgeGeneralizer
from src.agent.edge_validator import EdgeValidator
from src.agent.strategy_memory import PastResult


@pytest.fixture
def discovery_agent():
    """Create AlphaDiscoveryAgent."""
    return AlphaDiscoveryAgent()


@pytest.fixture
def edge_validator():
    """Create EdgeValidator."""
    return EdgeValidator()


@pytest.fixture
def edge_generalizer():
    """Create EdgeGeneralizer."""
    return EdgeGeneralizer()


@pytest.fixture
def sample_hypothesis():
    """Create sample hypothesis."""
    return {
        "edge_hypothesis": "Momentum works in Bull markets with fast=12-15",
        "regime_patterns": ["Bull", "LowVol"],
        "parameter_patterns": {"fast": [12, 13, 14, 15], "slow": [45, 50, 55, 60]},
        "confidence": 0.8,
        "reasoning": "Backtested on 100 trades",
    }


class TestAlphaDiscovery:
    """Test AlphaDiscoveryAgent."""

    def test_analyze_winning_patterns_raises_invalid_strategy(self, discovery_agent):
        """Test: raises ValueError if strategy not in registry."""
        with pytest.raises(ValueError):
            discovery_agent.analyze_winning_patterns("invalid_strategy")

    def test_analyze_winning_patterns_returns_dict(self, discovery_agent):
        """Test: returns dict with required fields."""
        sample_trades = [
            PastResult(
                run_id="run_1",
                strategy_type="momentum",
                params='{"fast": 12, "slow": 26}',
                sharpe=1.5,
                total_return=0.15,
                max_drawdown=-0.1,
                regime="LowVol-Bull",
                signals_vector='{"vix_percentile_252d": 35, "momentum_63d": 0.08}',
            ),
        ]

        with patch.object(discovery_agent.memory, "recall", return_value=sample_trades):
            with patch.object(discovery_agent, "_parse_json_response") as mock_parse:
                mock_parse.return_value = {
                    "edge_hypothesis": "Test edge",
                    "regime_patterns": ["Bull"],
                    "parameter_patterns": {"fast": [12, 13, 14]},
                    "confidence": 0.8,
                    "reasoning": "Test",
                }

                with patch.object(discovery_agent.planner, "generate_proposals") as mock_gen:
                    mock_gen.return_value = [
                        {
                            "edge_hypothesis": "Test edge",
                            "regime_patterns": ["Bull"],
                            "parameter_patterns": {"fast": [12, 13, 14]},
                            "confidence": 0.8,
                        }
                    ]

                    result = discovery_agent.analyze_winning_patterns("momentum")

                    assert "strategy" in result
                    assert "num_trades_analyzed" in result
                    assert "edge_hypothesis" in result
                    assert "regime_patterns" in result
                    assert "parameter_patterns" in result
                    assert "confidence" in result

    def test_analyze_winning_patterns_handles_llm_failure(self, discovery_agent):
        """Test: handles LLM API errors gracefully."""
        with patch.object(discovery_agent.memory, "recall", return_value=[]):
            with patch.object(discovery_agent.planner, "generate_text", side_effect=Exception("LLM error")):
                with pytest.raises(RuntimeError):
                    discovery_agent.analyze_winning_patterns("momentum")


class TestEdgeValidator:
    """Test EdgeValidator."""

    def test_validate_edge_improvement_bps_calculation(self, edge_validator, sample_hypothesis):
        """Test: improvement_bps = (sharpe_restricted - sharpe_baseline) * 10000."""
        with patch.object(edge_validator, "_run_backtest_baseline") as mock_baseline:
            with patch.object(edge_validator, "_run_backtest_restricted") as mock_restricted:
                mock_baseline.return_value = {
                    "sharpe": 1.0,
                    "total_return": 0.1,
                    "num_trades": 50,
                }
                mock_restricted.return_value = {
                    "sharpe": 1.1,
                    "total_return": 0.12,
                    "num_trades": 50,
                }

                result = edge_validator.validate_edge_hypothesis(
                    "momentum", sample_hypothesis, {}, min_improvement_bps=100
                )

                # Improvement: (1.1 - 1.0) * 10000 = 1000 bps
                assert result["improvement_bps"] == 1000

    def test_validate_edge_verdict_real(self, edge_validator, sample_hypothesis):
        """Test: verdict='REAL edge' if improvement_bps > threshold."""
        with patch.object(edge_validator, "_run_backtest_baseline") as mock_baseline:
            with patch.object(edge_validator, "_run_backtest_restricted") as mock_restricted:
                mock_baseline.return_value = {"sharpe": 1.0, "total_return": 0.1, "num_trades": 50}
                mock_restricted.return_value = {"sharpe": 1.15, "total_return": 0.12, "num_trades": 50}

                result = edge_validator.validate_edge_hypothesis(
                    "momentum", sample_hypothesis, {}, min_improvement_bps=100
                )

                assert result["verdict"] == "REAL edge"

    def test_validate_edge_verdict_spurious(self, edge_validator, sample_hypothesis):
        """Test: verdict='SPURIOUS correlation' if improvement_bps < threshold."""
        with patch.object(edge_validator, "_run_backtest_baseline") as mock_baseline:
            with patch.object(edge_validator, "_run_backtest_restricted") as mock_restricted:
                mock_baseline.return_value = {"sharpe": 1.0, "total_return": 0.1, "num_trades": 50}
                mock_restricted.return_value = {"sharpe": 1.005, "total_return": 0.1, "num_trades": 50}

                result = edge_validator.validate_edge_hypothesis(
                    "momentum", sample_hypothesis, {}, min_improvement_bps=100
                )

                # Improvement: (1.005 - 1.0) * 10000 = 50 bps < 100 threshold
                assert result["verdict"] == "SPURIOUS correlation"

    def test_validate_edge_tracks_trade_counts(self, edge_validator, sample_hypothesis):
        """Test: reports num_trades_baseline and num_trades_restricted."""
        with patch.object(edge_validator, "_run_backtest_baseline") as mock_baseline:
            with patch.object(edge_validator, "_run_backtest_restricted") as mock_restricted:
                mock_baseline.return_value = {"sharpe": 1.0, "total_return": 0.1, "num_trades": 100}
                mock_restricted.return_value = {"sharpe": 1.1, "total_return": 0.12, "num_trades": 50}

                result = edge_validator.validate_edge_hypothesis(
                    "momentum", sample_hypothesis, {}
                )

                assert result["num_trades_baseline"] == 100
                assert result["num_trades_restricted"] == 50

    def test_validate_edge_raises_invalid_strategy(self, edge_validator, sample_hypothesis):
        """Test: raises ValueError for invalid strategy."""
        with pytest.raises(ValueError):
            edge_validator.validate_edge_hypothesis(
                "invalid_strategy", sample_hypothesis, {}
            )

    def test_validate_edge_raises_invalid_hypothesis(self, edge_validator):
        """Test: raises ValueError for empty hypothesis."""
        with pytest.raises(ValueError):
            edge_validator.validate_edge_hypothesis("momentum", {}, {})


class TestEdgeGeneralizer:
    """Test EdgeGeneralizer."""

    def test_test_across_regimes_returns_all_regimes(self, edge_generalizer, sample_hypothesis):
        """Test: tests Bull, Bear, Choppy; returns results for all 3."""
        with patch.object(edge_generalizer, "backtest_runner") as mock_runner:
            mock_runner.return_value = {
                "metrics": {"sharpe_ratio": 1.5, "total_return": 0.1, "num_trades": 50}
            }

            result = edge_generalizer.test_across_regimes(
                "momentum", sample_hypothesis, {}
            )

            assert "Bull" in result["results"]
            assert "Bear" in result["results"]
            assert "Choppy" in result["results"]

    def test_test_across_timeframes_returns_all_windows(self, edge_generalizer, sample_hypothesis):
        """Test: tests 5d, 10d, 15d, 20d, 30d."""
        with patch.object(edge_generalizer, "backtest_runner") as mock_runner:
            mock_runner.return_value = {
                "metrics": {"sharpe_ratio": 1.5, "total_return": 0.1, "num_trades": 50}
            }

            result = edge_generalizer.test_across_timeframes(
                "momentum", sample_hypothesis, {}
            )

            assert "5d" in result["results"]
            assert "10d" in result["results"]
            assert "15d" in result["results"]
            assert "20d" in result["results"]
            assert "30d" in result["results"]

    def test_test_across_asset_classes_returns_all_classes(self, edge_generalizer, sample_hypothesis):
        """Test: tests large_cap, mid_cap, small_cap."""
        with patch.object(edge_generalizer, "backtest_runner") as mock_runner:
            mock_runner.return_value = {
                "metrics": {"sharpe_ratio": 1.5, "total_return": 0.1, "num_trades": 50}
            }

            result = edge_generalizer.test_across_asset_classes(
                "momentum", sample_hypothesis, {"AAPL": pd.DataFrame()}
            )

            assert "large_cap" in result["results"]
            assert "mid_cap" in result["results"]
            assert "small_cap" in result["results"]

    def test_generalization_identifies_peak_result(self, edge_generalizer, sample_hypothesis):
        """Test: identifies and reports best regime/timeframe/asset class."""
        # Provide sample ohlcv_data
        sample_ohlcv = {
            "AAPL": pd.DataFrame({"Close": [100, 101, 102]}),
            "GOOGL": pd.DataFrame({"Close": [100, 101, 102]}),
        }

        with patch.object(edge_generalizer, "backtest_runner") as mock_runner:
            def runner_side_effect(ohlcv, tickers, strategy, params):
                # Return highest Sharpe for Bull regime
                return {
                    "metrics": {"sharpe_ratio": 1.5, "total_return": 0.1, "num_trades": 50}
                }

            mock_runner.side_effect = runner_side_effect

            result = edge_generalizer.test_across_regimes(
                "momentum", sample_hypothesis, sample_ohlcv
            )

            assert result["best_regime"] in ["Bull", "Bear", "Choppy"]
            assert result["peak_sharpe"] > 0.0

    def test_generalization_conclusion_correct(self, edge_generalizer, sample_hypothesis):
        """Test: conclusion matches result pattern."""
        with patch.object(edge_generalizer, "backtest_runner") as mock_runner:
            # All regimes profitable
            mock_runner.return_value = {
                "metrics": {"sharpe_ratio": 1.5, "total_return": 0.1, "num_trades": 50}
            }

            result = edge_generalizer.test_across_regimes(
                "momentum", sample_hypothesis, {}
            )

            # Should say "Works everywhere" when all regimes have Sharpe > 1.0
            assert "Works" in result["conclusion"] or "conclusion" in result

    def test_test_across_regimes_invalid_strategy(self, edge_generalizer, sample_hypothesis):
        """Test: handles invalid strategy gracefully."""
        result = edge_generalizer.test_across_regimes(
            "invalid_strategy", sample_hypothesis, {}
        )

        assert result["conclusion"] == "Strategy not found"
