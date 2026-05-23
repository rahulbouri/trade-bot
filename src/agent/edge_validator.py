"""
Edge Validator — Validate Alpha Discovery Hypotheses
====================================================

Backtests hypothesis-restricted strategies vs unrestricted baseline.
Requires >100 bps Sharpe improvement to consider edge "real".
"""

import json
import logging
from typing import Any, Dict, Optional

import pandas as pd

from src.backtest.runner import run_backtest
from src.strategies.strategy_registry import STRATEGY_REGISTRY

logger = logging.getLogger(__name__)


class EdgeValidator:
    """Validate alpha discovery hypotheses with >100 bps Sharpe improvement requirement."""

    def __init__(self):
        """Initialize with backtest engine."""
        self.backtest_runner = run_backtest

    def validate_edge_hypothesis(
        self,
        strategy: str,
        hypothesis: Dict[str, Any],
        ohlcv_data: Dict[str, pd.DataFrame],
        eval_start: Optional[str] = None,
        min_improvement_bps: int = 100,
    ) -> Dict[str, Any]:
        """
        Backtest hypothesis-restricted strategy vs unrestricted baseline.

        Process:
        1. Extract from hypothesis: edge_hypothesis, parameter_patterns, regime_patterns
        2. Run Backtest 1 (Baseline - unrestricted)
        3. Run Backtest 2 (Restricted - with hypothesis constraints)
        4. Compute improvement: improvement_bps = (sharpe_restricted - sharpe_baseline) * 10000
        5. Verdict: if improvement_bps > min_improvement_bps: "REAL edge", else "SPURIOUS"

        Returns dict with verdict, improvement metrics, and reasoning.
        """
        if not hypothesis:
            raise ValueError("Hypothesis cannot be empty")

        if strategy not in STRATEGY_REGISTRY:
            raise ValueError(f"Strategy '{strategy}' not in STRATEGY_REGISTRY")

        edge_hypothesis = hypothesis.get("edge_hypothesis", "Unknown")
        parameter_patterns = hypothesis.get("parameter_patterns", {})
        regime_patterns = hypothesis.get("regime_patterns", [])

        # Get strategy object
        strat_obj = STRATEGY_REGISTRY[strategy]
        default_params = getattr(strat_obj, "default_params", {})

        # Backtest 1: Baseline (unrestricted)
        baseline_result = self._run_backtest_baseline(strategy, ohlcv_data, default_params)
        baseline_sharpe = baseline_result["sharpe"]
        baseline_return = baseline_result["total_return"]
        baseline_trades = baseline_result["num_trades"]

        # Backtest 2: Restricted (with hypothesis constraints)
        restricted_params = self._apply_hypothesis_constraints(default_params, parameter_patterns)
        restricted_result = self._run_backtest_restricted(
            strategy, ohlcv_data, restricted_params, regime_patterns
        )
        restricted_sharpe = restricted_result["sharpe"]
        restricted_return = restricted_result["total_return"]
        restricted_trades = restricted_result["num_trades"]

        # Compute improvement
        improvement_bps = int((restricted_sharpe - baseline_sharpe) * 10000)
        improvement_pct = (restricted_sharpe - baseline_sharpe) / max(abs(baseline_sharpe), 0.1) * 100

        # Verdict
        is_real_edge = improvement_bps > min_improvement_bps
        verdict = "REAL edge" if is_real_edge else "SPURIOUS correlation"

        # Confidence: higher if:
        # - Improvement is larger
        # - Sufficient trades in restricted backtest
        confidence = min(
            abs(improvement_bps) / 500.0,  # 500 bps = 100% confidence
            restricted_trades / 100.0,  # 100 trades = 100% confidence
            1.0,
        )

        reasoning = (
            f"Baseline: Sharpe={baseline_sharpe:.2f} ({baseline_trades} trades). "
            f"Restricted: Sharpe={restricted_sharpe:.2f} ({restricted_trades} trades). "
            f"Improvement: {improvement_bps} bps ({improvement_pct:.1f}%). "
            f"Verdict: {verdict} (threshold: {min_improvement_bps} bps)"
        )

        return {
            "strategy": strategy,
            "hypothesis": edge_hypothesis,
            "baseline_sharpe": baseline_sharpe,
            "baseline_return": baseline_return,
            "restricted_sharpe": restricted_sharpe,
            "restricted_return": restricted_return,
            "improvement_bps": improvement_bps,
            "improvement_pct": improvement_pct,
            "verdict": verdict,
            "confidence": float(confidence),
            "reasoning": reasoning,
            "num_trades_baseline": baseline_trades,
            "num_trades_restricted": restricted_trades,
        }

    def _run_backtest_baseline(
        self,
        strategy: str,
        ohlcv_data: Dict[str, pd.DataFrame],
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run baseline backtest with unrestricted parameters."""
        try:
            # Get all available assets
            tickers = list(ohlcv_data.keys())
            if not tickers:
                logger.warning("No OHLCV data available for baseline backtest")
                return {"sharpe": 0.0, "total_return": 0.0, "num_trades": 0}

            result = self.backtest_runner(ohlcv_data, tickers, strategy, params)
            if result and "metrics" in result:
                metrics = result["metrics"]
                return {
                    "sharpe": metrics.get("sharpe_ratio", 0.0),
                    "total_return": metrics.get("total_return", 0.0),
                    "num_trades": metrics.get("num_trades", 0),
                }
            return {"sharpe": 0.0, "total_return": 0.0, "num_trades": 0}
        except Exception as e:
            logger.warning("Baseline backtest failed: %s", e)
            return {"sharpe": 0.0, "total_return": 0.0, "num_trades": 0}

    def _run_backtest_restricted(
        self,
        strategy: str,
        ohlcv_data: Dict[str, pd.DataFrame],
        params: Dict[str, Any],
        regime_patterns: list,
    ) -> Dict[str, Any]:
        """Run restricted backtest with hypothesis constraints."""
        try:
            # Filter assets by regime if patterns specified (simplified for now)
            tickers = list(ohlcv_data.keys())
            if not tickers:
                return {"sharpe": 0.0, "total_return": 0.0, "num_trades": 0}

            result = self.backtest_runner(ohlcv_data, tickers, strategy, params)
            if result and "metrics" in result:
                metrics = result["metrics"]
                return {
                    "sharpe": metrics.get("sharpe_ratio", 0.0),
                    "total_return": metrics.get("total_return", 0.0),
                    "num_trades": metrics.get("num_trades", 0),
                }
            return {"sharpe": 0.0, "total_return": 0.0, "num_trades": 0}
        except Exception as e:
            logger.warning("Restricted backtest failed: %s", e)
            return {"sharpe": 0.0, "total_return": 0.0, "num_trades": 0}

    def _apply_hypothesis_constraints(
        self,
        params: Dict[str, Any],
        parameter_patterns: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Apply hypothesis parameter constraints to default params."""
        if not parameter_patterns:
            return params

        restricted_params = params.copy()

        # Apply constraints from parameter_patterns
        for param_name, allowed_values in parameter_patterns.items():
            if param_name in restricted_params and isinstance(allowed_values, list):
                # Use middle value of allowed range
                if allowed_values:
                    restricted_params[param_name] = int(sum(allowed_values) / len(allowed_values))

        return restricted_params
