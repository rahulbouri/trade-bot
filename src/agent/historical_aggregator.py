"""Historical Aggregator — Pre-compute trade summaries by (strategy, regime)."""

import json
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.agent.strategy_memory import PastResult, StrategyMemory

logger = logging.getLogger(__name__)


@dataclass
class StrategyRegimeSummary:
    """Aggregate statistics for a (strategy_type, regime) combination."""
    strategy: str
    regime: str
    num_trades: int
    win_rate: float
    mean_sharpe: float
    std_sharpe: float
    mean_return: float
    return_percentile_25: float
    return_percentile_75: float
    max_return: float
    min_return: float
    avg_max_drawdown: float
    best_params: Dict[str, Any] = field(default_factory=dict)
    param_ranges: Dict[str, List[float]] = field(default_factory=dict)
    confidence: float = 0.0


class HistoricalAggregator:
    """Pre-compute aggregate statistics from StrategyMemory for LLM context."""

    def __init__(self, memory: Optional[StrategyMemory] = None):
        self.memory = memory or StrategyMemory()

    def aggregate_all_trades(self) -> Dict[str, StrategyRegimeSummary]:
        """Query StrategyMemory, group by (strategy, regime), compute summaries."""
        all_trades = self.memory.recall(n=10000)
        if not all_trades:
            return {}

        groups: Dict[str, List[PastResult]] = {}
        for trade in all_trades:
            key = f"{trade.strategy_type}_{trade.regime}"
            groups.setdefault(key, []).append(trade)

        summaries: Dict[str, StrategyRegimeSummary] = {}
        for key, trades in groups.items():
            strategy = trades[0].strategy_type
            regime = trades[0].regime
            num_trades = len(trades)

            sharpe_values = [t.sharpe for t in trades]
            return_values = [t.total_return for t in trades]
            drawdown_values = [t.max_drawdown for t in trades]

            mean_sharpe = sum(sharpe_values) / num_trades
            std_sharpe = self._std_dev(sharpe_values)
            mean_return = sum(return_values) / num_trades

            sorted_returns = sorted(return_values)
            return_percentile_25 = self._percentile(sorted_returns, 25)
            return_percentile_75 = self._percentile(sorted_returns, 75)

            winners = [t for t in trades if t.sharpe > 1.0]
            win_rate = len(winners) / num_trades

            winner_params = []
            for w in winners:
                try:
                    winner_params.append(json.loads(w.params))
                except (json.JSONDecodeError, TypeError):
                    pass

            all_params = []
            for t in trades:
                try:
                    all_params.append(json.loads(t.params))
                except (json.JSONDecodeError, TypeError):
                    pass

            best_params = self._get_best_params(winner_params)
            param_ranges = self._get_param_ranges(all_params)
            avg_max_drawdown = sum(drawdown_values) / num_trades
            confidence = min(num_trades / 100.0, 0.95)

            summaries[key] = StrategyRegimeSummary(
                strategy=strategy,
                regime=regime,
                num_trades=num_trades,
                win_rate=win_rate,
                mean_sharpe=mean_sharpe,
                std_sharpe=std_sharpe,
                mean_return=mean_return,
                return_percentile_25=return_percentile_25,
                return_percentile_75=return_percentile_75,
                max_return=max(return_values),
                min_return=min(return_values),
                avg_max_drawdown=avg_max_drawdown,
                best_params=best_params,
                param_ranges=param_ranges,
                confidence=confidence,
            )

        logger.debug("Aggregated %d strategy-regime groups from %d trades", len(summaries), len(all_trades))
        return summaries

    def to_prompt_context(self, summaries: Dict[str, StrategyRegimeSummary]) -> str:
        """Format summaries as readable text block for LLM (NO LLM CALL)."""
        if not summaries:
            return "No historical data available."

        sorted_items = sorted(summaries.values(), key=lambda s: s.mean_sharpe, reverse=True)

        blocks = ["STRATEGY PERFORMANCE SUMMARY (1-Year History)\n"]
        for s in sorted_items:
            if s.confidence > 0.8:
                confidence_label = "HIGH"
            elif s.confidence > 0.5:
                confidence_label = "MEDIUM"
            else:
                confidence_label = "LOW"

            params_str = ", ".join(f"{k}={v}" for k, v in s.best_params.items()) if s.best_params else "N/A"

            block = (
                f"{s.strategy.upper()} ({s.regime} regime):\n"
                f"  • {s.num_trades} trades, {s.win_rate:.0%} winners\n"
                f"  • Mean Sharpe: {s.mean_sharpe:.2f} (±{s.std_sharpe:.2f})\n"
                f"  • Returns: avg {s.mean_return:+.1%}, range [{s.return_percentile_25:+.1%} to {s.return_percentile_75:+.1%}]\n"
                f"  • Drawdown: avg {s.avg_max_drawdown:.1%}\n"
                f"  • Best params: {params_str}\n"
                f"  • Confidence: {confidence_label} ({s.num_trades} samples)"
            )
            blocks.append(block)

        return "\n\n".join(blocks)

    def _std_dev(self, values: List[float]) -> float:
        """Compute standard deviation for a list of values."""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return math.sqrt(variance)

    def _percentile(self, sorted_values: List[float], pct: int) -> float:
        """Compute percentile from a pre-sorted list."""
        if not sorted_values:
            return 0.0
        n = len(sorted_values)
        index = (pct / 100.0) * (n - 1)
        lower = int(index)
        upper = min(lower + 1, n - 1)
        fraction = index - lower
        return sorted_values[lower] + fraction * (sorted_values[upper] - sorted_values[lower])

    def _get_best_params(self, winner_params: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract most common params from winners (Sharpe > 1.0 trades)."""
        if not winner_params:
            return {}

        param_counts: Dict[str, Dict[Any, int]] = {}
        for params in winner_params:
            for key, value in params.items():
                param_counts.setdefault(key, {})
                hashable = value if not isinstance(value, list) else tuple(value)
                param_counts[key][hashable] = param_counts[key].get(hashable, 0) + 1

        best: Dict[str, Any] = {}
        for key, counts in param_counts.items():
            best[key] = max(counts, key=counts.get)
        return best

    def _get_param_ranges(self, all_params: List[Dict[str, Any]]) -> Dict[str, List[float]]:
        """Extract min/max range for numeric parameters."""
        ranges: Dict[str, List[float]] = {}
        for params in all_params:
            for key, value in params.items():
                if isinstance(value, (int, float)):
                    if key not in ranges:
                        ranges[key] = [float(value), float(value)]
                    else:
                        ranges[key][0] = min(ranges[key][0], float(value))
                        ranges[key][1] = max(ranges[key][1], float(value))
        return ranges
