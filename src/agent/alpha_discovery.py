"""
Alpha Discovery Agent — Find Trading Edges via LLM Analysis
===========================================================

Analyzes winning historical trades to discover patterns and edge hypotheses.
Uses LLM to identify regime conditions, parameter ranges, and edge rules.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional

from src.agent.base_planner import create_planner
from src.agent.strategy_memory import StrategyMemory
from src.features.regime import RegimeSignals
from src.strategies.strategy_registry import STRATEGY_REGISTRY

logger = logging.getLogger(__name__)


class AlphaDiscoveryAgent:
    """Discover trading edges from historical winning trades using LLM analysis."""

    def __init__(self):
        """Initialize with StrategyMemory and LLM planner."""
        self.memory = StrategyMemory()
        self.planner = create_planner()

    def analyze_winning_patterns(
        self,
        strategy: str,
        min_sharpe: float = 1.0,
        max_trades: int = 100,
    ) -> Dict[str, Any]:
        """
        Analyze 100+ winning trades to discover edge patterns.

        Process:
        1. Query StrategyMemory: all trades where strategy_type==strategy AND sharpe>=min_sharpe
        2. Fetch up to max_trades, sort by sharpe descending
        3. For each trade, extract params, sharpe, return, regime, signals
        4. Format as JSON for LLM
        5. Call LLM with prompt
        6. Return raw LLM response + parsed structure
        """
        if strategy not in STRATEGY_REGISTRY:
            raise ValueError(f"Strategy '{strategy}' not in STRATEGY_REGISTRY")

        # Query trades from memory
        all_trades = self.memory.recall(strategy_type=strategy, n=max_trades)
        winning_trades = [t for t in all_trades if t.sharpe >= min_sharpe]

        if not winning_trades:
            logger.warning("No winning trades found for strategy %s with min_sharpe %.2f", strategy, min_sharpe)
            winning_trades = all_trades  # Use all trades if no winners

        # Format trades for LLM
        formatted_trades = []
        for trade in winning_trades[:max_trades]:
            params = json.loads(trade.params) if isinstance(trade.params, str) else trade.params
            signals_dict = self.memory._deserialize_signals(trade.signals_vector)

            trade_data = {
                "params": params,
                "sharpe": trade.sharpe,
                "return": trade.total_return,
                "regime": trade.regime,
                "signals": signals_dict or {},
            }
            formatted_trades.append(trade_data)

        # Create LLM prompt
        trades_json = json.dumps(formatted_trades, indent=2)
        prompt = f"""You are analyzing {len(formatted_trades)} winning trades for the '{strategy}' strategy.
Each trade has parameters, returns, Sharpe ratio, and market regime at entry.

Trades data:
{trades_json}

Analyze and answer these questions:
1. Which regimes had the most winners? (e.g., Bull, Bear, Choppy)
2. Which parameter values appear in >70% of winners?
3. Are there specific regime conditions that boost Sharpe? (e.g., low VIX, high momentum)
4. What ONE specific edge hypothesis explains the winners?
   Format: "Edge: [Strategy name] with [parameters] works ONLY in [regime conditions]"
   Example: "Edge: Momentum with fast=12-15, slow=45-60 works in Bull markets, avg Sharpe 1.5 vs 0.8"

IMPORTANT: Return ONLY valid JSON in this format, with no extra text:
{{
    "edge_hypothesis": "...",
    "regime_patterns": ["Bull", "LowVol"],
    "parameter_patterns": {{"fast": [12, 13, 14, 15], "slow": [45, 50, 55, 60]}},
    "confidence": 0.8,
    "reasoning": "..."
}}"""

        # Call LLM — use generate_text() for free-form JSON (not generate_proposals
        # which expects a list of parameter dicts; discovery needs a single object)
        try:
            llm_text, token_counts = self.planner.generate_text(prompt)
            logger.info(
                "LLM TOKENS [alpha_discovery | %s]:  "
                "prompt=%d  completion=%d  thinking=%d  total=%d",
                self.planner.__class__.__name__,
                token_counts["prompt_tokens"],
                token_counts["completion_tokens"],
                token_counts["thinking_tokens"],
                token_counts["total_tokens"],
            )
            if not llm_text:
                raise RuntimeError("LLM returned empty response")
            parsed_hypothesis = self._parse_json_response(llm_text)
        except Exception as e:
            logger.warning("LLM call failed: %s", e)
            raise RuntimeError(f"LLM analysis failed: {e}")

        # Extract fields from parsed hypothesis
        edge_hypothesis = parsed_hypothesis.get("edge_hypothesis", "Unknown edge")
        regime_patterns = parsed_hypothesis.get("regime_patterns", [])
        parameter_patterns = parsed_hypothesis.get("parameter_patterns", {})
        confidence = parsed_hypothesis.get("confidence", 0.5)

        return {
            "strategy": strategy,
            "num_trades_analyzed": len(formatted_trades),
            "llm_response": json.dumps(parsed_hypothesis),
            "parsed_hypothesis": parsed_hypothesis,
            "edge_hypothesis": edge_hypothesis,
            "regime_patterns": regime_patterns,
            "parameter_patterns": parameter_patterns,
            "confidence": confidence,
        }

    def _parse_json_response(self, text: str) -> Dict[str, Any]:
        """Extract JSON from LLM response text."""
        # Try to find JSON object in response
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass

        # Return default if parsing fails
        logger.warning("Could not parse JSON response, using default")
        return {
            "edge_hypothesis": "Unknown",
            "regime_patterns": [],
            "parameter_patterns": {},
            "confidence": 0.0,
            "reasoning": "Failed to parse LLM response",
        }
