"""Test HistoricalAggregator: trade aggregation and prompt context formatting."""

import json
from unittest.mock import MagicMock

import pytest

from src.agent.historical_aggregator import HistoricalAggregator, StrategyRegimeSummary
from src.agent.strategy_memory import PastResult, StrategyMemory


def _make_trade(strategy_type, regime, sharpe, total_return, max_drawdown=0.0, params=None):
    return PastResult(
        run_id="test",
        strategy_type=strategy_type,
        regime=regime,
        sharpe=sharpe,
        total_return=total_return,
        max_drawdown=max_drawdown,
        params=json.dumps(params or {}),
    )


@pytest.fixture
def mock_memory():
    return MagicMock(spec=StrategyMemory)


@pytest.fixture
def aggregator(mock_memory):
    agg = HistoricalAggregator(memory=mock_memory)
    return agg, mock_memory


class TestHistoricalAggregator:

    def test_aggregate_all_trades_empty_memory(self, aggregator):
        """Returns {} if StrategyMemory has no trades."""
        agg, memory = aggregator
        memory.recall.return_value = []
        result = agg.aggregate_all_trades()
        assert result == {}

    def test_aggregate_all_trades_groups_by_strategy_regime(self, aggregator):
        """Correctly groups trades by (strategy_type, regime)."""
        agg, memory = aggregator
        memory.recall.return_value = [
            _make_trade("momentum", "Bull", 1.5, 0.12),
            _make_trade("momentum", "Bull", 1.2, 0.10),
            _make_trade("trend", "Bull", 1.1, 0.08),
        ]
        result = agg.aggregate_all_trades()
        assert "momentum_Bull" in result
        assert "trend_Bull" in result
        assert result["momentum_Bull"].num_trades == 2
        assert result["trend_Bull"].num_trades == 1

    def test_aggregate_computes_win_rate_correctly(self, aggregator):
        """win_rate = count(sharpe > 1.0) / total_trades."""
        agg, memory = aggregator
        memory.recall.return_value = [
            _make_trade("momentum", "Bull", 1.5, 0.12),
            _make_trade("momentum", "Bull", 0.8, 0.05),
        ]
        result = agg.aggregate_all_trades()
        assert result["momentum_Bull"].win_rate == pytest.approx(0.5)

    def test_aggregate_computes_mean_and_std_sharpe(self, aggregator):
        """mean_sharpe and std_sharpe computed correctly."""
        agg, memory = aggregator
        memory.recall.return_value = [
            _make_trade("momentum", "Bull", 1.0, 0.10),
            _make_trade("momentum", "Bull", 1.2, 0.12),
            _make_trade("momentum", "Bull", 1.4, 0.14),
        ]
        result = agg.aggregate_all_trades()
        s = result["momentum_Bull"]
        assert s.mean_sharpe == pytest.approx(1.2, rel=1e-3)
        assert s.std_sharpe > 0.0

    def test_aggregate_identifies_best_params_from_winners(self, aggregator):
        """best_params = most common params in winners (Sharpe > 1.0)."""
        agg, memory = aggregator
        memory.recall.return_value = [
            _make_trade("momentum", "Bull", 1.5, 0.12, params={"fast": 12, "slow": 26}),
            _make_trade("momentum", "Bull", 1.3, 0.10, params={"fast": 12, "slow": 50}),
            _make_trade("momentum", "Bull", 0.5, 0.02, params={"fast": 14, "slow": 30}),
        ]
        result = agg.aggregate_all_trades()
        best = result["momentum_Bull"].best_params
        assert best.get("fast") == 12

    def test_aggregate_computes_return_percentiles(self, aggregator):
        """25th and 75th percentile returns computed correctly."""
        agg, memory = aggregator
        returns = [0.05, 0.10, 0.15, 0.20]
        memory.recall.return_value = [
            _make_trade("momentum", "Bull", 1.5, r) for r in returns
        ]
        result = agg.aggregate_all_trades()
        s = result["momentum_Bull"]
        assert s.return_percentile_25 == pytest.approx(0.0875, abs=0.01)
        assert s.return_percentile_75 == pytest.approx(0.1625, abs=0.01)

    def test_to_prompt_context_returns_readable_text(self, aggregator):
        """Text output (NO LLM CALL), includes strategy/regime/metrics."""
        agg, memory = aggregator
        memory.recall.return_value = [
            _make_trade("momentum", "Bull", 1.5, 0.12, params={"fast": 12}),
            _make_trade("momentum", "Bull", 1.2, 0.10, params={"fast": 12}),
        ]
        summaries = agg.aggregate_all_trades()
        text = agg.to_prompt_context(summaries)
        assert "MOMENTUM" in text
        assert "Bull" in text
        assert "winners" in text
        assert len(text) < 1000

    def test_to_prompt_context_empty_returns_default(self, aggregator):
        """Empty summaries returns default message."""
        agg, _ = aggregator
        text = agg.to_prompt_context({})
        assert "No historical data available." in text

    def test_aggregate_confidence_grows_with_trades(self, aggregator):
        """confidence = min(num_trades/100, 0.95)."""
        agg, memory = aggregator

        # 50 trades → confidence = 0.50
        memory.recall.return_value = [
            _make_trade("momentum", "Bull", 1.5, 0.12) for _ in range(50)
        ]
        result = agg.aggregate_all_trades()
        assert result["momentum_Bull"].confidence == pytest.approx(0.50)

        # 150 trades → confidence = 0.95 (capped)
        memory.recall.return_value = [
            _make_trade("momentum", "Bull", 1.5, 0.12) for _ in range(150)
        ]
        result = agg.aggregate_all_trades()
        assert result["momentum_Bull"].confidence == pytest.approx(0.95)

    def test_aggregate_handles_invalid_json_params(self, aggregator):
        """Invalid JSON params are skipped gracefully."""
        agg, memory = aggregator
        trade = _make_trade("momentum", "Bull", 1.5, 0.12)
        trade.params = "NOT_VALID_JSON"
        memory.recall.return_value = [trade]
        result = agg.aggregate_all_trades()
        assert "momentum_Bull" in result
        assert result["momentum_Bull"].best_params == {}
        assert result["momentum_Bull"].param_ranges == {}

    def test_to_prompt_context_confidence_labels(self, aggregator):
        """HIGH/MEDIUM/LOW confidence labels appear in output."""
        agg, memory = aggregator
        # 90 trades → confidence=0.9 → HIGH
        memory.recall.return_value = [
            _make_trade("momentum", "Bull", 1.5, 0.12) for _ in range(90)
        ]
        summaries = agg.aggregate_all_trades()
        text = agg.to_prompt_context(summaries)
        assert "HIGH" in text

        # 60 trades → confidence=0.6 → MEDIUM
        memory.recall.return_value = [
            _make_trade("trend", "Bear", 1.1, 0.08) for _ in range(60)
        ]
        summaries2 = agg.aggregate_all_trades()
        text2 = agg.to_prompt_context(summaries2)
        assert "MEDIUM" in text2

    def test_aggregate_no_winners_empty_best_params(self, aggregator):
        """No winners → best_params is empty dict."""
        agg, memory = aggregator
        memory.recall.return_value = [
            _make_trade("momentum", "Bear", 0.5, 0.02, params={"fast": 12}),
        ]
        result = agg.aggregate_all_trades()
        assert result["momentum_Bear"].best_params == {}

    def test_percentile_empty_list(self, aggregator):
        """_percentile returns 0.0 for empty list."""
        agg, _ = aggregator
        assert agg._percentile([], 25) == 0.0
