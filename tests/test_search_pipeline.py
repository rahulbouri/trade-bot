"""Test SearchPipeline: asset search, strategy shortlist, scoring."""

import json
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest

from src.agent.search_pipeline import SearchPipeline
from src.agent.strategy_memory import PastResult, StrategyMemory
from src.features.regime import RegimeSignals, signals_to_vector


@pytest.fixture
def pipeline():
    """Create SearchPipeline instance."""
    return SearchPipeline()


@pytest.fixture
def sample_signals():
    """Create sample RegimeSignals."""
    return RegimeSignals(
        vix_percentile_252d=35.0,
        momentum_63d=0.08,
        realized_vol_21d=0.15,
        drawdown_from_52w_high=-0.05,
        price_vs_200sma_pct=0.03,
        vol_regime="mid",
        regime_label="MidVol-Bull",
        regime_confidence=0.8,
    )


@pytest.fixture
def mock_memory(monkeypatch):
    """Mock StrategyMemory."""
    mock = MagicMock(spec=StrategyMemory)
    return mock


class TestSearchPipeline:
    """Test SearchPipeline methods."""

    def test_find_similar_winning_trades_returns_top_k(self, pipeline, sample_signals):
        """Test: returns up to top_k trades, sorted by similarity descending."""
        # Create mock trades
        trades = []
        for i in range(5):
            signals = RegimeSignals(
                vix_percentile_252d=35.0 + i * 2,
                momentum_63d=0.08 + i * 0.02,
                realized_vol_21d=0.15,
                drawdown_from_52w_high=-0.05,
                price_vs_200sma_pct=0.03,
                vol_regime="mid",
            )
            trade = PastResult(
                run_id=f"run_{i}",
                strategy_type="momentum",
                params='{"fast": 12, "slow": 26}',
                sharpe=1.5 - i * 0.1,
                total_return=0.15,
                max_drawdown=-0.10,
                signals_vector=json.dumps(signals.__dict__),
                regime="MidVol-Bull",
            )
            trades.append(trade)

        # Mock memory.recall to return trades
        with patch.object(pipeline.memory, "recall", return_value=trades):
            result = pipeline.find_similar_winning_trades(sample_signals, "momentum", top_k=3)

            # Assert top_k enforcement
            assert len(result) <= 3

            # Assert similarity descending
            if len(result) > 1:
                for i in range(len(result) - 1):
                    assert result[i]["similarity"] >= result[i + 1]["similarity"]

    def test_find_similar_winning_trades_filters_by_min_sharpe(self, pipeline, sample_signals):
        """Test: only returns trades with sharpe >= min_sharpe."""
        trades = []
        sharpes = [0.8, 1.2, 1.5, 2.0]
        for i, sharpe in enumerate(sharpes):
            signals = sample_signals
            trade = PastResult(
                run_id=f"run_{i}",
                strategy_type="momentum",
                params='{"fast": 12}',
                sharpe=sharpe,
                total_return=0.1,
                max_drawdown=-0.1,
                signals_vector=json.dumps(signals.__dict__),
                regime="MidVol-Bull",
            )
            trades.append(trade)

        with patch.object(pipeline.memory, "recall", return_value=trades):
            result = pipeline.find_similar_winning_trades(sample_signals, "momentum", min_sharpe=1.0)

            # Should only include trades with sharpe >= 1.0
            sharpes_returned = [t["sharpe"] for t in result]
            assert all(s >= 1.0 for s in sharpes_returned)

    def test_find_similar_winning_trades_skips_max_distance(self, pipeline, sample_signals):
        """Test: skips trades with distance > max_distance."""
        # Create trades with different signal patterns (different distances)
        trades = []
        vol_percentiles = [35.0, 40.0, 50.0, 70.0, 85.0]
        for i, vp in enumerate(vol_percentiles):
            signals = RegimeSignals(
                vix_percentile_252d=vp,
                momentum_63d=0.08,
                realized_vol_21d=0.15,
                drawdown_from_52w_high=-0.05,
                price_vs_200sma_pct=0.03,
                vol_regime="mid",
            )
            trade = PastResult(
                run_id=f"run_{i}",
                strategy_type="momentum",
                params='{"fast": 12}',
                sharpe=1.5,
                total_return=0.1,
                max_drawdown=-0.1,
                signals_vector=json.dumps(signals.__dict__),
                regime="MidVol-Bull",
            )
            trades.append(trade)

        with patch.object(pipeline.memory, "recall", return_value=trades):
            result = pipeline.find_similar_winning_trades(sample_signals, "momentum", max_distance=0.2)

            # All returned trades should have distance <= 0.2
            assert all(t["distance"] <= 0.2 for t in result)

    def test_find_similar_winning_trades_raises_on_invalid_strategy(self, pipeline, sample_signals):
        """Test: raises ValueError if strategy not in registry."""
        with pytest.raises(ValueError):
            pipeline.find_similar_winning_trades(sample_signals, "invalid_strategy")

    def test_find_similar_winning_trades_raises_on_empty_memory(self, pipeline, sample_signals):
        """Test: raises KeyError if no trades found."""
        with patch.object(pipeline.memory, "recall", return_value=[]):
            with pytest.raises(KeyError):
                pipeline.find_similar_winning_trades(sample_signals, "momentum")

    def test_score_asset_for_strategy_returns_valid_score(self, pipeline, sample_signals):
        """Test: score ranges [0, 1], formula correct."""
        similar_trades = [
            {
                "run_id": f"run_{i}",
                "strategy": "momentum",
                "params": {"fast": 12},
                "sharpe": 1.5,
                "return": 0.15,
                "max_drawdown": -0.1,
                "distance": 0.1,
                "similarity": 0.9,
                "regime": "MidVol-Bull",
                "win_rate": 0.7,
            }
            for i in range(3)
        ]

        with patch.object(pipeline, "find_similar_winning_trades", return_value=similar_trades):
            result = pipeline.score_asset_for_strategy("AAPL", "momentum", sample_signals, pd.DataFrame())

            assert "score" in result
            assert 0.0 <= result["score"] <= 1.0
            assert result["ticker"] == "AAPL"
            assert result["strategy"] == "momentum"

    def test_score_asset_for_strategy_bootstrap_score_no_matches(self, pipeline, sample_signals):
        """Test: bootstrap regime-based score returned when no similar trades found."""
        with patch.object(pipeline, "find_similar_winning_trades", side_effect=KeyError):
            result = pipeline.score_asset_for_strategy("AAPL", "momentum", sample_signals, pd.DataFrame())

            # Bootstrap score is regime-based (0.0–0.70), not zero
            assert 0.0 <= result["score"] <= 0.70
            assert result["matching_trades"] == 0
            assert "Bootstrap" in result["reasoning"]

    def test_search_assets_returns_top_k_sorted(self, pipeline, sample_signals):
        """Test: search_assets returns top_k by score, descending."""
        with patch.object(pipeline, "score_asset_for_strategy") as mock_score:
            # Return decreasing scores
            def side_effect(ticker, *args, **kwargs):
                scores = {"AAPL": 0.9, "GOOGL": 0.7, "MSFT": 0.5}
                return {
                    "ticker": ticker,
                    "score": scores.get(ticker, 0.0),
                    "strategy": "momentum",
                    "distance": 0.1,
                    "similarity": 0.9,
                    "avg_historical_return": 0.1,
                    "avg_historical_sharpe": 1.5,
                    "matching_trades": 3,
                    "reasoning": "test",
                    "similar_trades": [],
                }

            mock_score.side_effect = side_effect

            result = pipeline.search_assets(
                ["AAPL", "GOOGL", "MSFT"], sample_signals, pd.DataFrame(), "momentum", top_k=2
            )

            assert len(result) <= 2
            # Check scores are descending
            if len(result) > 1:
                for i in range(len(result) - 1):
                    assert result[i]["score"] >= result[i + 1]["score"]

    def test_search_assets_empty_universe(self, pipeline, sample_signals):
        """Test: handles empty asset universe gracefully."""
        result = pipeline.search_assets([], sample_signals, pd.DataFrame(), "momentum")

        assert result == []

    def test_shortlist_strategies_groups_by_regime(self, pipeline):
        """Test: returns strategies ranked by mean sharpe in regime."""
        # Create trades in a specific regime
        trades = [
            PastResult(
                run_id="run_1",
                strategy_type="momentum",
                params='{}',
                sharpe=1.5,
                total_return=0.1,
                max_drawdown=-0.1,
                regime="LowVol-Bull",
                signals_vector="{}",
            ),
            PastResult(
                run_id="run_2",
                strategy_type="momentum",
                params='{}',
                sharpe=1.3,
                total_return=0.1,
                max_drawdown=-0.1,
                regime="LowVol-Bull",
                signals_vector="{}",
            ),
            PastResult(
                run_id="run_3",
                strategy_type="mean_reversion",
                params='{}',
                sharpe=0.9,
                total_return=0.1,
                max_drawdown=-0.1,
                regime="LowVol-Bull",
                signals_vector="{}",
            ),
        ]

        with patch.object(pipeline.memory, "recall", return_value=trades):
            result = pipeline.shortlist_strategies("LowVol-Bull", top_k=2)

            # Should return strategies sorted by mean sharpe
            assert len(result) <= 2
            assert result[0]["strategy"] == "momentum"  # Higher mean Sharpe

    def test_shortlist_strategies_top_k(self, pipeline):
        """Test: returns top_k strategies only."""
        # Create mock return for different strategies
        def mock_recall(*args, **kwargs):
            strategy = kwargs.get("strategy_type", "")
            if strategy == "momentum":
                return [
                    PastResult(
                        run_id="run_1",
                        strategy_type="momentum",
                        params='{}',
                        sharpe=1.5,
                        total_return=0.1,
                        max_drawdown=-0.1,
                        regime="LowVol-Bull",
                    )
                ]
            elif strategy == "mean_reversion":
                return [
                    PastResult(
                        run_id="run_2",
                        strategy_type="mean_reversion",
                        params='{}',
                        sharpe=0.9,
                        total_return=0.1,
                        max_drawdown=-0.1,
                        regime="LowVol-Bull",
                    )
                ]
            return []

        with patch.object(pipeline.memory, "recall", side_effect=mock_recall):
            result = pipeline.shortlist_strategies("LowVol-Bull", top_k=1)

            assert len(result) <= 1

    def test_backtest_accepted_empty_list(self, pipeline):
        """Test: handles empty accepted_tickers list."""
        result = pipeline.backtest_accepted([], "momentum", "LowVol-Bull", {})

        assert result["best_ticker"] == ""
        assert result["best_sharpe"] == 0.0
        assert result["results"] == []

    def test_signals_to_vector_normalized(self, sample_signals):
        """Test: signals_to_vector returns normalized [0, 1] vector."""
        vector = signals_to_vector(sample_signals)

        assert isinstance(vector, np.ndarray)
        assert vector.shape == (6,)
        assert np.all(vector >= 0.0)
        assert np.all(vector <= 1.0)

    def test_distance_calculation(self, sample_signals):
        """Test: Euclidean distance calculation."""
        v1 = signals_to_vector(sample_signals)

        signals2 = RegimeSignals(
            vix_percentile_252d=40.0,  # Different
            momentum_63d=0.08,
            realized_vol_21d=0.15,
            drawdown_from_52w_high=-0.05,
            price_vs_200sma_pct=0.03,
            vol_regime="mid",
        )
        v2 = signals_to_vector(signals2)

        distance = float(np.linalg.norm(v1 - v2))

        assert distance >= 0.0  # Distance is always non-negative
        assert distance < 1.0  # Both vectors are normalized to [0, 1]
