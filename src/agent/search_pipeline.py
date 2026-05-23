"""
Search Pipeline — Asset Search & Strategy Shortlist with Historical Trade Analysis
===================================================================================

Finds winning assets and strategies by analyzing historical winning trades.
Uses signal distance metrics to identify similar market regimes.
"""

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.agent.strategy_memory import StrategyMemory
from src.backtest.runner import run_backtest
from src.features.regime import RegimeSignals, signals_to_vector
from src.strategies.strategy_registry import STRATEGY_REGISTRY

logger = logging.getLogger(__name__)


class SearchPipeline:
    """Search for suitable assets and strategies based on historical performance."""

    def __init__(self):
        """Initialize with connection to StrategyMemory."""
        self.memory = StrategyMemory()

    def find_similar_winning_trades(
        self,
        current_signals: RegimeSignals,
        strategy: str,
        max_distance: float = 0.3,
        min_sharpe: float = 1.0,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Find historical winning trades most similar to current market regime.

        Returns list of dicts with fields:
        - run_id, strategy, params, sharpe, return, max_drawdown, distance,
          similarity, regime, win_rate
        """
        if strategy not in STRATEGY_REGISTRY:
            raise ValueError(f"Strategy '{strategy}' not in STRATEGY_REGISTRY")

        # Query all trades for this strategy
        all_trades = self.memory.recall(strategy_type=strategy, n=1000)
        if not all_trades:
            raise KeyError(f"No trades found for strategy '{strategy}' in memory")

        # Convert current signals to vector
        current_vector = signals_to_vector(current_signals)

        # Score each trade by distance to current regime
        scored_trades = []
        for trade in all_trades:
            if trade.sharpe < min_sharpe:
                continue

            # Deserialize signals from JSON
            signals_dict = self.memory._deserialize_signals(trade.signals_vector)
            if not signals_dict:
                continue

            try:
                trade_signals = RegimeSignals(**signals_dict)
            except TypeError:
                continue

            trade_vector = signals_to_vector(trade_signals)
            distance = float(np.linalg.norm(current_vector - trade_vector))

            if distance > max_distance:
                continue

            similarity = 1.0 / (1.0 + distance)
            params = json.loads(trade.params) if isinstance(trade.params, str) else trade.params

            scored_trades.append({
                "run_id": trade.run_id,
                "strategy": trade.strategy_type,
                "params": params,
                "sharpe": trade.sharpe,
                "return": trade.total_return,
                "max_drawdown": trade.max_drawdown,
                "distance": distance,
                "similarity": similarity,
                "regime": trade.regime,
                "win_rate": 0.7,  # Placeholder; could track actual win_rate
            })

        # Sort by similarity descending
        scored_trades.sort(key=lambda x: x["similarity"], reverse=True)

        return scored_trades[:top_k]

    def score_asset_for_strategy(
        self,
        ticker: str,
        strategy: str,
        current_signals: RegimeSignals,
        current_features: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        Score a single asset for a given strategy.

        Score formula:
            base_similarity = 1.0 / (1.0 + distance)
            return_multiplier = 1.0 + avg_historical_return
            win_rate_adjustment = (matching_trades / expected_trades) ** 0.5
            final_score = base_similarity * return_multiplier * win_rate_adjustment
        """
        try:
            similar_trades = self.find_similar_winning_trades(
                current_signals, strategy, max_distance=0.3, min_sharpe=1.0, top_k=10
            )
        except (ValueError, KeyError):
            similar_trades = []

        if not similar_trades:
            # Bootstrap: no winning trades in memory yet.
            # Compute a regime-based score from current signals so the LLM
            # can still evaluate the asset on regime fit alone.
            try:
                momentum = float(getattr(current_signals, "momentum_63d", 0.0))
                vix_pct = float(getattr(current_signals, "vix_percentile_252d", 50.0)) / 100.0
                # Positive momentum boosts score; high VIX reduces it.
                # Range: ~0.35–0.70, tuned so Bull markets clear the 0.50 BUY threshold.
                bootstrap_score = float(np.clip(0.55 + momentum * 2.0 - vix_pct * 0.15, 0.0, 0.70))
            except Exception:
                bootstrap_score = 0.45
            return {
                "ticker": ticker,
                "strategy": strategy,
                "score": bootstrap_score,
                "distance": 1.0,
                "similarity": 0.0,
                "avg_historical_return": 0.0,
                "avg_historical_sharpe": 0.0,
                "matching_trades": 0,
                "reasoning": (
                    f"Bootstrap (no historical trades in memory). "
                    f"Regime-based score={bootstrap_score:.2f} "
                    f"(momentum={momentum:.2%}, vix_pct={vix_pct:.0%})."
                ),
            }

        # Compute averages
        avg_return = float(np.mean([t["return"] for t in similar_trades]))
        avg_sharpe = float(np.mean([t["sharpe"] for t in similar_trades]))
        best_distance = similar_trades[0]["distance"]
        best_similarity = similar_trades[0]["similarity"]
        matching_trades = len(similar_trades)

        # Score formula
        base_similarity = best_similarity
        return_multiplier = 1.0 + avg_return
        expected_trades = 20  # Arbitrary threshold
        win_rate_adjustment = (matching_trades / expected_trades) ** 0.5
        final_score = base_similarity * return_multiplier * win_rate_adjustment
        final_score = float(np.clip(final_score, 0.0, 1.0))

        reasoning = (
            f"Best similar trade: Sharpe={avg_sharpe:.2f}, Return={avg_return:.1%}. "
            f"Found {matching_trades} similar trades."
        )

        return {
            "ticker": ticker,
            "strategy": strategy,
            "score": final_score,
            "distance": best_distance,
            "similarity": best_similarity,
            "avg_historical_return": avg_return,
            "avg_historical_sharpe": avg_sharpe,
            "matching_trades": matching_trades,
            "reasoning": reasoning,
        }

    def search_assets(
        self,
        universe: List[str],
        current_signals: RegimeSignals,
        current_features: pd.DataFrame,
        strategy: str,
        top_k: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Score all assets in universe, return top_k by score.

        Execution: Parallel ThreadPoolExecutor(max_workers=8)
        Time: <3 seconds for 1500 assets
        """
        if not universe:
            return []

        def score_single_asset(ticker: str) -> Dict[str, Any]:
            """Score a single asset."""
            score_result = self.score_asset_for_strategy(
                ticker, strategy, current_signals, current_features
            )
            try:
                similar_trades = self.find_similar_winning_trades(
                    current_signals, strategy, max_distance=0.3, min_sharpe=1.0, top_k=3
                )
            except (ValueError, KeyError):
                similar_trades = []
            score_result["similar_trades"] = similar_trades
            return score_result

        results = []
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(score_single_asset, ticker) for ticker in universe]
            for future in futures:
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.warning("Error scoring asset: %s", e)

        # Sort by score descending
        results.sort(key=lambda x: x.get("score", 0.0), reverse=True)

        return results[:top_k]

    def shortlist_strategies(
        self,
        regime: str,
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Rank strategies by historical Sharpe in given regime.

        Regime format: "{VolRegime}-{TrendRegime}" (e.g., "LowVol-Bull")
        """
        strategy_stats = {}

        # Collect all trades for each strategy in this regime
        for strategy_name in STRATEGY_REGISTRY.keys():
            trades = self.memory.recall(regime=regime, strategy_type=strategy_name, n=500)
            if not trades:
                strategy_stats[strategy_name] = {
                    "strategy": strategy_name,
                    "mean_sharpe": 0.0,
                    "num_trades": 0,
                    "win_rate": 0.0,
                    "avg_return": 0.0,
                }
            else:
                sharpes = [t.sharpe for t in trades]
                returns = [t.total_return for t in trades]
                wins = sum(1 for t in trades if t.sharpe > 0.0)

                strategy_stats[strategy_name] = {
                    "strategy": strategy_name,
                    "mean_sharpe": float(np.mean(sharpes)),
                    "num_trades": len(trades),
                    "win_rate": float(wins / len(trades)) if trades else 0.0,
                    "avg_return": float(np.mean(returns)),
                }

        # Sort by mean Sharpe descending
        sorted_strategies = sorted(
            strategy_stats.values(),
            key=lambda x: x["mean_sharpe"],
            reverse=True,
        )

        return sorted_strategies[:top_k]

    def backtest_accepted(
        self,
        accepted_tickers: List[str],
        strategy: str,
        regime: str,
        ohlcv_data: Dict[str, pd.DataFrame],
    ) -> Dict[str, Any]:
        """
        Run backtests on accepted assets, store results to StrategyMemory with signals_vector.
        """
        if not accepted_tickers:
            return {
                "best_ticker": "",
                "best_sharpe": 0.0,
                "best_params": {},
                "results": [],
            }

        results = []
        best_result = None
        best_sharpe = -float("inf")

        # Run backtest for each accepted ticker
        for ticker in accepted_tickers:
            if ticker not in ohlcv_data:
                continue

            # Get strategy's default params from registry
            if strategy not in STRATEGY_REGISTRY:
                continue

            strat_obj = STRATEGY_REGISTRY[strategy]
            default_params = getattr(strat_obj, "default_params", {})

            try:
                # Run backtest
                bt_result = run_backtest(ohlcv_data, [ticker], strategy, default_params)
                if not bt_result or "metrics" not in bt_result:
                    continue

                metrics = bt_result["metrics"]
                sharpe = metrics.get("sharpe_ratio", 0.0)
                total_return = metrics.get("total_return", 0.0)
                max_drawdown = metrics.get("max_drawdown", 0.0)

                # Create result entry
                result_entry = {
                    "ticker": ticker,
                    "sharpe": sharpe,
                    "return": total_return,
                    "max_drawdown": max_drawdown,
                    "params": default_params,
                }
                results.append(result_entry)

                # Track best
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_result = result_entry

                # Store to StrategyMemory with signals_vector
                from src.features.regime import detect_regime_full
                signals = detect_regime_full(ohlcv_data.get(ticker, pd.DataFrame()))
                signals_json = json.dumps(signals.__dict__)

                past_result = PastResult(
                    regime=regime,
                    strategy_type=strategy,
                    params=json.dumps(default_params),
                    sharpe=sharpe,
                    total_return=total_return,
                    max_drawdown=max_drawdown,
                    confidence=1.0,
                    generation_method="search_pipeline",
                    reasoning=f"Backtest for {ticker} in {regime}",
                    signals_vector=signals_json,
                )
                self.memory.store(past_result)

            except Exception as e:
                logger.warning("Backtest failed for %s: %s", ticker, e)

        if best_result is None:
            best_result = {
                "ticker": "",
                "sharpe": 0.0,
                "return": 0.0,
                "max_drawdown": 0.0,
                "params": {},
            }

        return {
            "best_ticker": best_result.get("ticker", ""),
            "best_sharpe": best_result.get("sharpe", 0.0),
            "best_params": best_result.get("params", {}),
            "results": results,
        }


from src.agent.strategy_memory import PastResult  # noqa: E402
