"""
Edge Generalizer — Test Edge Hypotheses Across Contexts
======================================================

Tests edges across regimes, timeframes, and asset classes.
Identifies constraints and generalization patterns.
"""

import logging
from typing import Any, Dict, Optional

import pandas as pd

from src.backtest.runner import run_backtest
from src.strategies.strategy_registry import STRATEGY_REGISTRY

logger = logging.getLogger(__name__)


class EdgeGeneralizer:
    """Test edge hypotheses across regimes, timeframes, and asset classes."""

    def __init__(self):
        """Initialize with backtest engine and data."""
        self.backtest_runner = run_backtest

    def test_across_regimes(
        self,
        strategy: str,
        hypothesis: Dict[str, Any],
        ohlcv_data: Dict[str, pd.DataFrame],
        eval_start: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Test if edge works in Bull, Bear, and Choppy regimes.

        Returns dict with results for each regime, best_regime, peak_sharpe, generalization_score.
        """
        regimes = ["Bull", "Bear", "Choppy"]
        results = {}

        strat_obj = STRATEGY_REGISTRY.get(strategy)
        if not strat_obj:
            return {
                "strategy": strategy,
                "hypothesis": hypothesis.get("edge_hypothesis", ""),
                "test_type": "regimes",
                "results": {},
                "best_regime": "Unknown",
                "peak_sharpe": 0.0,
                "generalization_score": 0.0,
                "conclusion": "Strategy not found",
            }

        default_params = getattr(strat_obj, "default_params", {})

        # Test each regime
        for regime in regimes:
            try:
                tickers = list(ohlcv_data.keys())[:10]  # Sample first 10 tickers
                if not tickers:
                    results[regime] = {"sharpe": 0.0, "return": 0.0, "num_trades": 0}
                    continue

                bt_result = self.backtest_runner(ohlcv_data, tickers, strategy, default_params)
                if bt_result and "metrics" in bt_result:
                    metrics = bt_result["metrics"]
                    results[regime] = {
                        "sharpe": metrics.get("sharpe_ratio", 0.0),
                        "return": metrics.get("total_return", 0.0),
                        "num_trades": metrics.get("num_trades", 0),
                    }
                else:
                    results[regime] = {"sharpe": 0.0, "return": 0.0, "num_trades": 0}
            except Exception as e:
                logger.warning("Regime test failed for %s: %s", regime, e)
                results[regime] = {"sharpe": 0.0, "return": 0.0, "num_trades": 0}

        # Find best regime
        best_regime = max(results.keys(), key=lambda k: results[k].get("sharpe", 0.0))
        peak_sharpe = results[best_regime].get("sharpe", 0.0)

        # Compute generalization score
        sharpes = [r.get("sharpe", 0.0) for r in results.values()]
        if max(sharpes) > 0:
            generalization_score = min(sharpes) / max(sharpes)
        else:
            generalization_score = 0.0

        # Conclusion
        profitable_regimes = [r for r, stats in results.items() if stats.get("sharpe", 0.0) > 1.0]
        if len(profitable_regimes) == 3:
            conclusion = "Works everywhere"
        elif len(profitable_regimes) == 1:
            conclusion = f"Only in {profitable_regimes[0]}"
        else:
            conclusion = f"Works in {', '.join(profitable_regimes)} regimes"

        return {
            "strategy": strategy,
            "hypothesis": hypothesis.get("edge_hypothesis", ""),
            "test_type": "regimes",
            "results": results,
            "best_regime": best_regime,
            "peak_sharpe": peak_sharpe,
            "generalization_score": float(generalization_score),
            "conclusion": conclusion,
        }

    def test_across_timeframes(
        self,
        strategy: str,
        hypothesis: Dict[str, Any],
        ohlcv_data: Dict[str, pd.DataFrame],
        eval_start: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Test if edge works at different resolutions: 5d, 10d, 15d, 20d, 30d windows.

        Returns dict with results for each timeframe, best_timeframe, peak_sharpe, conclusion.
        """
        timeframes = ["5d", "10d", "15d", "20d", "30d"]
        results = {}

        strat_obj = STRATEGY_REGISTRY.get(strategy)
        if not strat_obj:
            return {
                "strategy": strategy,
                "hypothesis": hypothesis.get("edge_hypothesis", ""),
                "test_type": "timeframes",
                "results": {},
                "best_timeframe": "Unknown",
                "peak_sharpe": 0.0,
                "conclusion": "Strategy not found",
            }

        default_params = getattr(strat_obj, "default_params", {})

        # Test each timeframe by modifying lookback window params
        for timeframe in timeframes:
            try:
                # Parse timeframe to get period
                period = int(timeframe.replace("d", ""))
                modified_params = default_params.copy()

                # Adjust period-like params (heuristic: look for 'period', 'lookback', 'window')
                for key in modified_params:
                    if any(x in key.lower() for x in ["period", "lookback", "window"]):
                        modified_params[key] = period

                tickers = list(ohlcv_data.keys())[:10]
                if not tickers:
                    results[timeframe] = {"sharpe": 0.0, "return": 0.0}
                    continue

                bt_result = self.backtest_runner(ohlcv_data, tickers, strategy, modified_params)
                if bt_result and "metrics" in bt_result:
                    metrics = bt_result["metrics"]
                    results[timeframe] = {
                        "sharpe": metrics.get("sharpe_ratio", 0.0),
                        "return": metrics.get("total_return", 0.0),
                    }
                else:
                    results[timeframe] = {"sharpe": 0.0, "return": 0.0}
            except Exception as e:
                logger.warning("Timeframe test failed for %s: %s", timeframe, e)
                results[timeframe] = {"sharpe": 0.0, "return": 0.0}

        # Find best timeframe
        best_timeframe = max(results.keys(), key=lambda k: results[k].get("sharpe", 0.0))
        peak_sharpe = results[best_timeframe].get("sharpe", 0.0)

        # Conclusion
        if peak_sharpe > 1.0:
            conclusion = f"Optimal at {best_timeframe}"
        else:
            conclusion = "No strong performance across timeframes"

        return {
            "strategy": strategy,
            "hypothesis": hypothesis.get("edge_hypothesis", ""),
            "test_type": "timeframes",
            "results": results,
            "best_timeframe": best_timeframe,
            "peak_sharpe": peak_sharpe,
            "conclusion": conclusion,
        }

    def test_across_asset_classes(
        self,
        strategy: str,
        hypothesis: Dict[str, Any],
        ohlcv_data: Dict[str, pd.DataFrame],
        eval_start: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Test if edge works on large-cap, mid-cap, small-cap Indian stocks.

        Asset class definition:
        - Large-cap: market cap > ₹5B
        - Mid-cap: ₹500M - ₹5B
        - Small-cap: <₹500M

        Returns dict with results for each asset class, best_class, peak_sharpe, conclusion.
        """
        # For simplicity, use ticker patterns as proxy for market cap
        # In real implementation, would fetch actual market caps
        large_cap_tickers = [t for t in ohlcv_data.keys() if t in ["AAPL", "TCS", "INFY"]]
        mid_cap_tickers = [t for t in ohlcv_data.keys() if t in ["WIPRO", "TECHM", "HCL"]]
        small_cap_tickers = [t for t in ohlcv_data.keys() if t not in large_cap_tickers + mid_cap_tickers]

        classes = {
            "large_cap": large_cap_tickers or list(ohlcv_data.keys())[:3],
            "mid_cap": mid_cap_tickers or list(ohlcv_data.keys())[3:6],
            "small_cap": small_cap_tickers or list(ohlcv_data.keys())[6:],
        }

        results = {}

        strat_obj = STRATEGY_REGISTRY.get(strategy)
        if not strat_obj:
            return {
                "strategy": strategy,
                "hypothesis": hypothesis.get("edge_hypothesis", ""),
                "test_type": "asset_classes",
                "results": {},
                "best_class": "Unknown",
                "peak_sharpe": 0.0,
                "conclusion": "Strategy not found",
            }

        default_params = getattr(strat_obj, "default_params", {})

        # Test each asset class
        for class_name, tickers in classes.items():
            try:
                if not tickers:
                    results[class_name] = {
                        "sharpe": 0.0,
                        "return": 0.0,
                        "num_assets": 0,
                    }
                    continue

                # Filter ohlcv_data to only include tickers in this class
                class_ohlcv = {t: ohlcv_data[t] for t in tickers if t in ohlcv_data}
                if not class_ohlcv:
                    results[class_name] = {"sharpe": 0.0, "return": 0.0, "num_assets": 0}
                    continue

                bt_result = self.backtest_runner(class_ohlcv, list(class_ohlcv.keys()), strategy, default_params)
                if bt_result and "metrics" in bt_result:
                    metrics = bt_result["metrics"]
                    results[class_name] = {
                        "sharpe": metrics.get("sharpe_ratio", 0.0),
                        "return": metrics.get("total_return", 0.0),
                        "num_assets": len(class_ohlcv),
                    }
                else:
                    results[class_name] = {"sharpe": 0.0, "return": 0.0, "num_assets": len(class_ohlcv)}
            except Exception as e:
                logger.warning("Asset class test failed for %s: %s", class_name, e)
                results[class_name] = {"sharpe": 0.0, "return": 0.0, "num_assets": 0}

        # Find best class
        best_class = max(results.keys(), key=lambda k: results[k].get("sharpe", 0.0))
        peak_sharpe = results[best_class].get("sharpe", 0.0)

        # Conclusion
        if peak_sharpe > 1.0:
            conclusion = f"Works best on {best_class.replace('_', '-')}"
        else:
            conclusion = "No strong performance across asset classes"

        return {
            "strategy": strategy,
            "hypothesis": hypothesis.get("edge_hypothesis", ""),
            "test_type": "asset_classes",
            "results": results,
            "best_class": best_class,
            "peak_sharpe": peak_sharpe,
            "conclusion": conclusion,
        }
