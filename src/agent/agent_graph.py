"""
Agent Graph — ReAct Agent Loop with LangGraph StateGraph
==========================================================

Implements the real agentic loop:
  [discovery] → analyze → shortlist → llm_analysis → position_management
              → hypothesize → backtest → reflect → (loop or store)

Phase 2 integration:
  - llm_analysis_node: real LLM call via Gemini + HistoricalAggregator context
  - position_management_node: filters by llm_decisions before picking ticker
  - backtest_node: uses position-decided ticker, not config.reference_asset
  - discovery_node: runs conditionally when StrategyMemory >= 100 records
"""

import json
import logging
from typing import Any, Dict, List, Optional, TypedDict

import pandas as pd

from src.agent.base_planner import create_planner
from src.agent.context_builder import RegimeContext, build_context
from src.agent.historical_aggregator import HistoricalAggregator
from src.agent.position_manager import PositionManager
from src.agent.proposal_generator import Proposal, ProposalGenerator
from src.agent.search_pipeline import SearchPipeline
from src.agent.strategy_memory import PastResult, StrategyMemory
from src.backtest.metrics import PerformanceMetrics
from src.features.regime import detect_regime_full
from src.utils.config import config

logger = logging.getLogger(__name__)

# ── Minimum records in StrategyMemory before discovery runs ──────────────────
_DISCOVERY_MIN_RECORDS = 100

# ── Max assets to send to LLM per analysis call (controls token cost) ────────
_LLM_ANALYSIS_TOP_N = 10

# ── Prompt template for batch asset analysis ─────────────────────────────────
_ASSET_ANALYSIS_PROMPT = """\
You are a quantitative analyst evaluating trade candidates for today.

CURRENT MARKET REGIME: {regime_label} (confidence: {confidence:.0%})
  VIX Percentile (1Y): {vix_pct:.0f}th
  3-Month Momentum:    {momentum_63d:+.1%}
  Realized Vol (21d):  {realized_vol:.1%}
  Drawdown from Peak:  {drawdown:+.1%}

HISTORICAL STRATEGY PERFORMANCE (aggregated from StrategyMemory):
{historical_context}

CANDIDATE ASSETS (ranked by similarity to historical winners):
{asset_lines}

TASK:
For each candidate, decide:
  YES   — strong historical fit + regime aligned → take position
  NO    — poor fit or regime mismatch → skip
  MAYBE — moderate fit → monitor, prefer others

Rules:
- Prefer YES only when score > 0.5 AND historical context supports it
- Downgrade to NO if regime is Bear/Crisis and strategy is momentum-based
- Give one short sentence of reasoning per asset

Return ONLY a JSON array, no markdown fences or extra text:
[
  {{"ticker": "X", "decision": "YES", "confidence": 0.85, "reasoning": "one sentence"}},
  {{"ticker": "Y", "decision": "NO",  "confidence": 0.20, "reasoning": "one sentence"}},
  ...
]"""


class AgentState(TypedDict, total=False):
    """State flowing through the agent graph."""
    ohlcv_data: Dict[str, pd.DataFrame]
    features_df: pd.DataFrame
    context: Optional[RegimeContext]
    proposals: List[Proposal]
    results: List[Dict[str, Any]]
    best_result: Optional[Dict[str, Any]]
    iteration: int
    max_iterations: int
    strategy_type: str
    asset: str
    should_continue: bool
    memory_context: str
    run_log: List[str]
    # Phase 1: Search & Discovery
    regime_signals: Optional[Any]
    asset_search_results: List[Dict[str, Any]]
    strategies_ranked: List[Dict[str, Any]]
    llm_decisions: Dict[str, str]
    validated_edges: List[Dict[str, Any]]
    # Phase 2: Position Management
    current_position: Optional[Dict[str, Any]]
    position_decision: Dict[str, Any]
    # Phase 3: Paper Trading
    paper_trade_result: Optional[Dict[str, Any]]


# ─── Node: discovery ──────────────────────────────────────────────────────────

def discovery_node(state: AgentState) -> AgentState:
    """
    Offline alpha discovery: find edges in StrategyMemory via LLM analysis.

    Only runs when StrategyMemory has >= _DISCOVERY_MIN_RECORDS records.
    Designed for weekly/manual trigger; run_agent() calls it on every run
    but it self-gates on data sufficiency.
    """
    logger.info("=== DISCOVERY: Checking data sufficiency for alpha discovery ===")

    memory = StrategyMemory()
    all_trades = memory.recall(n=10000)
    n_records = len(all_trades)

    if n_records < _DISCOVERY_MIN_RECORDS:
        logger.info(
            "DISCOVERY: Skipping — %d/%d records in StrategyMemory (need %d to run)",
            n_records, _DISCOVERY_MIN_RECORDS, _DISCOVERY_MIN_RECORDS,
        )
        state["validated_edges"] = []
        state["run_log"].append(
            f"Discovery: Skipped ({n_records}/{_DISCOVERY_MIN_RECORDS} records needed)"
        )
        return state

    logger.info("DISCOVERY: Running with %d records in StrategyMemory", n_records)

    try:
        from src.agent.alpha_discovery import AlphaDiscoveryAgent
        from src.agent.edge_validator import EdgeValidator
        from src.agent.edge_generalizer import EdgeGeneralizer

        strategy_type = state.get("strategy_type", "momentum")
        agent = AlphaDiscoveryAgent()
        hypothesis = agent.analyze_winning_patterns(strategy_type)

        logger.info(
            "DISCOVERY: Hypothesis found — %s (confidence=%.0f%%)",
            hypothesis.get("edge_hypothesis", "unknown"),
            hypothesis.get("confidence", 0) * 100,
        )

        state["validated_edges"] = [hypothesis]
        state["run_log"].append(
            f"Discovery: Found edge hypothesis (confidence={hypothesis.get('confidence', 0):.0%})"
        )
    except Exception as e:
        logger.warning("DISCOVERY: Failed — %s", e)
        state["validated_edges"] = []
        state["run_log"].append(f"Discovery: Failed ({e})")

    return state


# ─── Node: analyze ────────────────────────────────────────────────────────────

def analyze_node(state: AgentState) -> AgentState:
    """Build regime context from market data and search for candidate assets."""
    logger.info("=== ANALYZE: Building market context & searching assets ===")
    from src.features.engine import compute_features
    from src.features.regime import detect_regime

    ohlcv = state["ohlcv_data"]
    asset = state.get("asset", config.reference_asset)

    features_df = compute_features(ohlcv, asset, config.vix_ticker)
    context = build_context(features_df)
    regime_label = detect_regime(features_df)
    context.regime_label = regime_label

    regime_signals = detect_regime_full(features_df)

    memory = StrategyMemory()
    memory_ctx = memory.to_prompt_context(regime_label, state.get("strategy_type", "momentum"))

    pipeline = SearchPipeline()
    strategy_type = state.get("strategy_type", "momentum")
    universe = list(ohlcv.keys())

    try:
        asset_search_results = pipeline.search_assets(
            universe, regime_signals, features_df, strategy_type, top_k=50
        )
        state["asset_search_results"] = asset_search_results
        num_found = len(asset_search_results)
        logger.info("ANALYZE: Asset search found %d candidates from universe of %d", num_found, len(universe))
        state["run_log"].append(f"Asset Search: Found {num_found} suitable candidates")
    except Exception as e:
        logger.warning("Asset search failed: %s", e)
        state["asset_search_results"] = []
        state["run_log"].append(f"Asset Search: Failed ({e})")

    state["features_df"] = features_df
    state["context"] = context
    state["regime_signals"] = regime_signals
    state["memory_context"] = memory_ctx
    state["run_log"] = state.get("run_log", [])
    state["run_log"].append(f"Regime: {regime_label} (confidence: {context.regime_confidence:.0%})")

    logger.info("ANALYZE: Regime=%s  Confidence=%.0f%%", regime_label, context.regime_confidence * 100)
    return state


# ─── Node: shortlist ──────────────────────────────────────────────────────────

def shortlist_node(state: AgentState) -> AgentState:
    """Rank strategies by historical Sharpe in detected regime."""
    logger.info("=== SHORTLIST: Ranking strategies by historical Sharpe ===")

    pipeline = SearchPipeline()
    context = state.get("context")
    regime = context.regime_label if context else "Unknown"

    try:
        strategies_ranked = pipeline.shortlist_strategies(regime, top_k=3)
        state["strategies_ranked"] = strategies_ranked
        strategy_names = [f"{s.get('strategy')} (Sharpe={s.get('mean_sharpe', 0):.2f})" for s in strategies_ranked]
        logger.info("SHORTLIST: Top 3 strategies: %s", strategy_names)
        state["run_log"].append(f"Shortlist: {strategy_names}")
    except Exception as e:
        logger.warning("Shortlist failed: %s", e)
        state["strategies_ranked"] = []
        state["run_log"].append(f"Shortlist: Failed ({e})")

    return state


# ─── Node: llm_analysis ───────────────────────────────────────────────────────

def llm_analysis_node(state: AgentState) -> AgentState:
    """
    Batch LLM analysis of top-N candidate assets.

    Steps:
      1. HistoricalAggregator computes regime+strategy summaries (0 tokens)
      2. Build batch prompt: regime context + historical summaries + top-N assets
      3. Single Gemini call → YES/NO/MAYBE per asset
      4. Log input/output/thinking token counts
      5. Fallback to score threshold if LLM unavailable or parse fails
    """
    logger.info("=== LLM ANALYSIS: Batch risk/reward assessment ===")

    # ── Step 1: Historical aggregation (0 tokens) ─────────────────────────────
    aggregator = HistoricalAggregator()
    summaries = aggregator.aggregate_all_trades()
    historical_context = aggregator.to_prompt_context(summaries)
    logger.info(
        "LLM ANALYSIS: HistoricalAggregator produced %d strategy-regime groups",
        len(summaries),
    )

    # ── Step 2: Pick top-N candidates ────────────────────────────────────────
    asset_search_results = state.get("asset_search_results", [])
    top_assets = asset_search_results[:_LLM_ANALYSIS_TOP_N]

    if not top_assets:
        logger.warning("LLM ANALYSIS: No candidate assets — skipping LLM call")
        state["llm_decisions"] = {}
        state["run_log"].append("LLM Analysis: No candidates found, skipped")
        return state

    # ── Step 3: Build regime context for prompt ───────────────────────────────
    context = state.get("context")
    regime_signals = state.get("regime_signals")

    regime_label = context.regime_label if context else "Unknown"
    confidence = context.regime_confidence if context else 0.0
    vix_pct = getattr(regime_signals, "vix_percentile_252d", 50.0) if regime_signals else 50.0
    momentum_63d = getattr(regime_signals, "momentum_63d", 0.0) if regime_signals else 0.0
    realized_vol = getattr(regime_signals, "realized_vol_21d", 0.15) if regime_signals else 0.15
    drawdown = getattr(regime_signals, "drawdown_from_52w_high", 0.0) if regime_signals else 0.0

    asset_lines = []
    for i, asset in enumerate(top_assets, 1):
        ticker = asset.get("ticker", "?")
        score = asset.get("score", 0.0)
        n_trades = asset.get("matching_trades", 0)
        avg_sharpe = asset.get("avg_historical_sharpe", 0.0)
        avg_return = asset.get("avg_historical_return", 0.0)
        reasoning = asset.get("reasoning", "")
        asset_lines.append(
            f"{i}. {ticker}: score={score:.3f}, {n_trades} similar historical trades"
            f" (avg Sharpe={avg_sharpe:.2f}, avg return={avg_return:+.1%})\n"
            f"   {reasoning}"
        )

    prompt = _ASSET_ANALYSIS_PROMPT.format(
        regime_label=regime_label,
        confidence=confidence,
        vix_pct=vix_pct,
        momentum_63d=momentum_63d,
        realized_vol=realized_vol,
        drawdown=drawdown,
        historical_context=historical_context,
        asset_lines="\n".join(asset_lines),
    )

    logger.debug("LLM ANALYSIS PROMPT (%d chars):\n%s", len(prompt), prompt)

    # ── Step 4: LLM call ─────────────────────────────────────────────────────
    planner = create_planner()
    llm_decisions: Dict[str, str] = {}

    if not planner.is_available():
        logger.warning(
            "LLM ANALYSIS: No LLM available (%s) — using score-threshold fallback",
            planner.__class__.__name__,
        )
    else:
        logger.info(
            "LLM ANALYSIS: Calling %s for %d assets...",
            planner.__class__.__name__,
            len(top_assets),
        )
        try:
            raw_text, token_counts = planner.generate_text(prompt)

            # Log token counts prominently
            logger.info(
                "LLM TOKENS [llm_analysis_node | %s]:  "
                "prompt=%d  completion=%d  thinking=%d  total=%d",
                config.llm.model,
                token_counts["prompt_tokens"],
                token_counts["completion_tokens"],
                token_counts["thinking_tokens"],
                token_counts["total_tokens"],
            )
            logger.debug("LLM ANALYSIS RAW RESPONSE:\n%s", raw_text)

            # ── Step 5: Parse decisions ───────────────────────────────────────
            start = raw_text.find("[")
            end = raw_text.rfind("]")
            if start != -1 and end != -1:
                decisions_list = json.loads(raw_text[start:end + 1])
                for item in decisions_list:
                    ticker = item.get("ticker", "")
                    decision = item.get("decision", "NO").upper()
                    conf = float(item.get("confidence", 0.5))
                    reasoning = item.get("reasoning", "")
                    if ticker:
                        llm_decisions[ticker] = decision
                        logger.info(
                            "LLM DECISION: %-15s → %-5s  conf=%.0f%%  reason: %s",
                            ticker, decision, conf * 100, reasoning,
                        )
            else:
                logger.warning("LLM ANALYSIS: Could not find JSON array in response — using fallback")

        except (json.JSONDecodeError, ValueError, Exception) as e:
            logger.warning("LLM ANALYSIS: Parse/call failed (%s) — using score-threshold fallback", e)

    # ── Fallback: score threshold ─────────────────────────────────────────────
    if not llm_decisions:
        logger.info("LLM ANALYSIS: Applying score-threshold fallback (>0.5=YES, 0.3-0.5=MAYBE, <0.3=NO)")
        for asset in asset_search_results:
            ticker = asset.get("ticker", "")
            score = asset.get("score", 0.0)
            if score > 0.5:
                llm_decisions[ticker] = "YES"
            elif score > 0.3:
                llm_decisions[ticker] = "MAYBE"
            else:
                llm_decisions[ticker] = "NO"

    state["llm_decisions"] = llm_decisions

    num_yes = sum(1 for d in llm_decisions.values() if d == "YES")
    num_maybe = sum(1 for d in llm_decisions.values() if d == "MAYBE")
    num_no = sum(1 for d in llm_decisions.values() if d == "NO")

    logger.info("LLM ANALYSIS COMPLETE: %d YES  %d MAYBE  %d NO", num_yes, num_maybe, num_no)
    state["run_log"].append(
        f"LLM Analysis: {num_yes} YES, {num_maybe} MAYBE, {num_no} NO "
        f"(model={config.llm.model})"
    )
    return state


# ─── Node: position_management ────────────────────────────────────────────────

def position_management_node(state: AgentState) -> AgentState:
    """
    Check overnight position, decide hold/exit/switch/buy/skip.

    FIX: filters asset_search_results to LLM-approved (YES/MAYBE) tickers
    before selecting the top asset signal. Previously used raw search rank,
    ignoring llm_decisions entirely.
    """
    logger.info("=== POSITION MANAGEMENT: Reviewing overnight position ===")

    position_manager = PositionManager()
    current_position = position_manager.get_current_position()

    if current_position:
        logger.info(
            "POSITION MGMT: Overnight position found — %s (held %d days, entry=%.4f)",
            current_position.ticker, current_position.days_held, current_position.entry_price,
        )
    else:
        logger.info("POSITION MGMT: No overnight position held")

    # ── Filter search results to LLM-approved assets ──────────────────────────
    asset_search_results = state.get("asset_search_results", [])
    llm_decisions = state.get("llm_decisions", {})

    if llm_decisions:
        # LLM analysis ran — filter to YES/MAYBE assets only
        approved_assets = [
            a for a in asset_search_results
            if llm_decisions.get(a.get("ticker", ""), "NO") in ("YES", "MAYBE")
        ]
        logger.info(
            "POSITION MGMT: %d/%d assets LLM-approved (YES/MAYBE) after filter",
            len(approved_assets), len(asset_search_results),
        )
    else:
        # LLM analysis did not run — no filter, use raw search results
        approved_assets = asset_search_results
        logger.info(
            "POSITION MGMT: llm_decisions empty — using all %d search results (no LLM filter)",
            len(approved_assets),
        )

    if approved_assets:
        top_asset = approved_assets[0]
        llm_label = llm_decisions.get(top_asset.get("ticker", ""), "YES" if not llm_decisions else "NO")
        logger.info(
            "POSITION MGMT: Top asset — %s (score=%.3f, LLM=%s)",
            top_asset.get("ticker"), top_asset.get("score", 0.0), llm_label,
        )
    elif asset_search_results:
        top_asset = asset_search_results[0]
        llm_label = "NO"
        logger.warning(
            "POSITION MGMT: No LLM-approved assets. Falling back to top search result: %s (LLM=NO, penalised)",
            top_asset.get("ticker"),
        )
    else:
        top_asset = {}
        llm_label = "NO"
        logger.warning("POSITION MGMT: No candidates at all — signal confidence=0")

    # ── Build agent signal ────────────────────────────────────────────────────
    if top_asset:
        raw_score = top_asset.get("score", 0.0)
        # Confidence multiplier: YES=1.0, MAYBE=0.8, NO (fallback)=0.5
        multiplier = 1.0 if llm_label == "YES" else (0.8 if llm_label == "MAYBE" else 0.5)
        confidence = min(raw_score * multiplier, 1.0)

        agent_signal = {
            "ticker": top_asset.get("ticker"),
            "confidence": confidence,
            "reasoning": top_asset.get("reasoning", ""),
            "score": top_asset.get("score", 0.0),
            "llm_decision": llm_label,
            "signal_strength": 1.0 if confidence > 0.5 else -0.5,
        }
    else:
        agent_signal = {
            "ticker": None,
            "confidence": 0.0,
            "reasoning": "No assets found",
            "signal_strength": 0.0,
            "llm_decision": "NO",
        }

    logger.info(
        "POSITION MGMT: Agent signal — ticker=%s  confidence=%.0f%%  llm=%s",
        agent_signal["ticker"], agent_signal["confidence"] * 100, agent_signal.get("llm_decision"),
    )

    # ── Execute decision ──────────────────────────────────────────────────────
    decision = position_manager.execute_decision(
        current_position, agent_signal, state["ohlcv_data"],
    )

    entry_price = top_asset.get("avg_historical_return", 0.0) if top_asset else 0.0
    position_manager.store_decision(decision, entry_price)

    state["current_position"] = current_position.__dict__ if current_position else None
    state["position_decision"] = decision.to_dict()
    state["run_log"].append(
        f"Position Management: {decision.action} — {decision.reasoning}"
    )

    logger.info(
        "POSITION MGMT: Decision=%s  current=%s → new=%s  confidence=%.0f%%",
        decision.action, decision.current_ticker, decision.new_ticker, decision.confidence * 100,
    )

    if decision.action == "EXIT":
        state["should_continue"] = False
        state["run_log"].append("Position exited. Skipping today's backtest.")
        logger.info("POSITION MGMT: EXIT decision — skipping hypothesize/backtest loop")

    return state


# ─── Node: hypothesize ────────────────────────────────────────────────────────

def hypothesize_node(state: AgentState) -> AgentState:
    """Generate strategy proposals via LLM or grid search."""
    iteration = state.get("iteration", 0) + 1
    state["iteration"] = iteration
    logger.info("=== HYPOTHESIZE (iteration %d): Generating proposals ===", iteration)

    generator = ProposalGenerator()
    strategy_type = state.get("strategy_type", "momentum")
    context = state["context"]

    proposals = generator.generate(
        context=context,
        n_proposals=5,
        strategy_type=strategy_type,
    )

    state["proposals"] = proposals
    methods = [p.generation_method for p in proposals]
    logger.info("HYPOTHESIZE: Generated %d proposals — methods: %s", len(proposals), methods)
    state["run_log"].append(
        f"Iteration {iteration}: Generated {len(proposals)} proposals ({methods})"
    )

    for i, p in enumerate(proposals):
        logger.info("  Proposal %d: %s  confidence=%.2f  method=%s", i + 1, p.params, p.confidence, p.generation_method)

    return state


# ─── Node: backtest ───────────────────────────────────────────────────────────

def backtest_node(state: AgentState) -> AgentState:
    """
    Run backtests on proposals for the position-decided ticker.

    FIX: Uses the ticker from position_decision (new_ticker or current_ticker)
    instead of always using config.reference_asset. Falls back gracefully
    if position_decision or OHLCV data isn't available.
    """
    logger.info("=== BACKTEST: Running tournament ===")
    from src.backtest.runner import run_backtest

    ohlcv = state["ohlcv_data"]
    strategy_type = state.get("strategy_type", "momentum")
    context = state.get("context")
    regime = context.regime_label if context else "Unknown"
    regime_signals = state.get("regime_signals")

    # ── Determine target ticker from position decision ────────────────────────
    position_decision = state.get("position_decision", {})
    decided_action = position_decision.get("action", "")

    # Priority: new_ticker (BUY/SWITCH) > current_ticker (HOLD) > top search > config default
    decided_ticker = (
        position_decision.get("new_ticker")
        or position_decision.get("current_ticker")
    )

    if decided_ticker and decided_ticker in ohlcv:
        asset = decided_ticker
        logger.info(
            "BACKTEST: Using position-decided ticker=%s (action=%s)", asset, decided_action
        )
    else:
        # Fallback: top LLM-approved search result
        asset_search = state.get("asset_search_results", [])
        llm_decisions = state.get("llm_decisions", {})
        fallback = next(
            (a["ticker"] for a in asset_search
             if a.get("ticker") in ohlcv and llm_decisions.get(a.get("ticker"), "NO") in ("YES", "MAYBE")),
            None,
        )
        asset = fallback or config.reference_asset
        logger.info(
            "BACKTEST: Position decision ticker not in OHLCV data. Falling back to ticker=%s", asset
        )

    logger.info("BACKTEST: Final ticker=%s  strategy=%s  regime=%s", asset, strategy_type, regime)

    results = []
    proposals = state.get("proposals", [])

    for i, proposal in enumerate(proposals):
        try:
            bt_result = run_backtest(ohlcv, [asset], strategy_type, proposal.params)
            if bt_result and "metrics" in bt_result:
                metrics = bt_result["metrics"]
                sharpe = metrics.get("sharpe_ratio", 0.0)
                results.append({
                    "proposal_idx": i,
                    "params": proposal.params,
                    "sharpe": sharpe,
                    "total_return": metrics.get("total_return", 0.0),
                    "max_drawdown": metrics.get("max_drawdown", 0.0),
                    "num_trades": metrics.get("num_trades", 0),
                    "generation_method": proposal.generation_method,
                    "confidence": proposal.confidence,
                    "reasoning": proposal.reasoning,
                    "ticker": asset,
                })
                logger.debug("BACKTEST: Proposal %d  Sharpe=%.3f  params=%s", i, sharpe, proposal.params)
        except Exception as e:
            logger.warning("BACKTEST: Proposal %d failed — %s", i, e)

    results.sort(key=lambda x: x.get("sharpe", 0.0), reverse=True)
    state["results"] = results

    if results:
        best = results[0]
        state["best_result"] = best
        logger.info(
            "BACKTEST: Best — ticker=%s  Sharpe=%.3f  Return=%.1f%%  params=%s",
            best.get("ticker"), best["sharpe"], best["total_return"] * 100, best["params"],
        )
        state["run_log"].append(
            f"Backtest [{asset}]: Sharpe={best['sharpe']:.3f}  Return={best['total_return']:.1%}  "
            f"Params={best['params']}"
        )
    else:
        state["best_result"] = None
        state["run_log"].append(f"Backtest [{asset}]: No valid results")
        logger.warning("BACKTEST: No valid results produced")

    return state


# ─── Node: reflect ────────────────────────────────────────────────────────────

def reflect_node(state: AgentState) -> AgentState:
    """Evaluate results. Decide if acceptable or should retry."""
    logger.info("=== REFLECT: Evaluating results ===")

    best = state.get("best_result")
    iteration = state.get("iteration", 1)
    max_iter = state.get("max_iterations", config.agent.max_iterations)
    min_sharpe = config.agent.min_acceptable_sharpe

    if best is None:
        state["should_continue"] = iteration < max_iter
        state["run_log"].append(f"Reflect: No results. {'Retrying...' if state['should_continue'] else 'Stopping.'}")
        return state

    sharpe = best.get("sharpe", 0.0)

    if sharpe >= min_sharpe:
        state["should_continue"] = False
        logger.info("REFLECT: Sharpe %.3f >= threshold %.2f — ACCEPTED", sharpe, min_sharpe)
        state["run_log"].append(f"Reflect: Sharpe {sharpe:.3f} >= {min_sharpe:.2f}. ACCEPTED.")
    elif iteration >= max_iter:
        state["should_continue"] = False
        logger.info("REFLECT: Max iterations reached — accepting best Sharpe %.3f", sharpe)
        state["run_log"].append(f"Reflect: Max iterations reached. Accepting best Sharpe {sharpe:.3f}.")
    else:
        state["should_continue"] = True
        logger.info("REFLECT: Sharpe %.3f < %.2f — retrying (iter %d/%d)", sharpe, min_sharpe, iteration, max_iter)
        state["run_log"].append(f"Reflect: Sharpe {sharpe:.3f} < {min_sharpe:.2f}. Retrying ({iteration}/{max_iter}).")

    return state


# ─── Node: store ─────────────────────────────────────────────────────────────

def store_node(state: AgentState) -> AgentState:
    """Persist best result to strategy memory with signals_vector."""
    logger.info("=== STORE: Persisting results ===")

    best = state.get("best_result")
    if best is None:
        state["run_log"].append("Store: Nothing to persist.")
        return state

    context = state.get("context")
    regime = context.regime_label if context else "Unknown"
    regime_signals = state.get("regime_signals")

    signals_vector = ""
    if regime_signals:
        try:
            signals_vector = json.dumps(regime_signals.__dict__)
        except Exception as e:
            logger.warning("Failed to serialize signals_vector: %s", e)

    memory = StrategyMemory()
    result = PastResult(
        regime=regime,
        strategy_type=state.get("strategy_type", "momentum"),
        params=json.dumps(best["params"]),
        sharpe=best.get("sharpe", 0.0),
        total_return=best.get("total_return", 0.0),
        max_drawdown=best.get("max_drawdown", 0.0),
        confidence=best.get("confidence", 0.0),
        generation_method=best.get("generation_method", ""),
        reasoning=best.get("reasoning", ""),
        signals_vector=signals_vector,
    )
    run_id = memory.store(result)
    logger.info(
        "STORE: Persisted run_id=%s  regime=%s  sharpe=%.3f  ticker=%s",
        run_id, regime, result.sharpe, best.get("ticker", "?"),
    )
    state["run_log"].append(f"Store: Persisted {run_id} (regime={regime}, sharpe={result.sharpe:.3f})")
    state["_store_run_id"] = run_id
    return state


# ─── Paper trading execution (always runs, regardless of EXIT/SKIP) ──────────

def _execute_paper_trades(state: AgentState) -> AgentState:
    """
    Execute paper trades and record the agent run.

    Called unconditionally at the end of run_agent() so that EXIT actions
    (which skip store_node) still trigger paper SELLs.
    """
    from src.trading.paper_trader import PaperTrader

    context = state.get("context")
    regime = context.regime_label if context else "Unknown"
    best = state.get("best_result") or {}
    position_decision = state.get("position_decision", {})
    action = position_decision.get("action", "SKIP")
    best_ticker = best.get("ticker") or position_decision.get("new_ticker") or config.reference_asset
    run_id = state.get("_store_run_id", "no-store")

    paper_trader = PaperTrader()
    ohlcv = state.get("ohlcv_data", {})

    def _last_close(ticker: str) -> float:
        df = ohlcv.get(ticker)
        if df is not None and not df.empty:
            try:
                return float(df["Close"].iloc[-1])
            except Exception:
                pass
        return 0.0

    if action == "SWITCH":
        old_ticker = position_decision.get("current_ticker")
        if old_ticker:
            try:
                paper_trader.execute_sell(old_ticker, _last_close(old_ticker), "SWITCH")
                logger.info("PAPER TRADE: SELL %s (SWITCH)", old_ticker)
            except Exception as e:
                logger.warning("PAPER TRADE: SELL failed for %s — %s", old_ticker, e)

    if action in ("BUY", "SWITCH"):
        try:
            entry_price = _last_close(best_ticker)
            buy_result = paper_trader.execute_buy(
                ticker=best_ticker,
                price=entry_price,
                strategy_type=state.get("strategy_type", "momentum"),
                params=best.get("params", {}),
                sharpe_at_entry=best.get("sharpe", 0.0),
                regime=regime,
                reason=position_decision.get("reasoning", ""),
            )
            state["paper_trade_result"] = buy_result
            logger.info(
                "PAPER TRADE: BUY %s  qty=%d  price=%.4f  cash_after=%.2f",
                best_ticker, buy_result["quantity"], buy_result["price"], buy_result["cash_after"],
            )
        except Exception as e:
            logger.warning("PAPER TRADE: BUY failed for %s — %s", best_ticker, e)

    elif action == "EXIT":
        exit_ticker = position_decision.get("current_ticker", best_ticker)
        try:
            sell_result = paper_trader.execute_sell(exit_ticker, _last_close(exit_ticker), "EXIT")
            state["paper_trade_result"] = sell_result
            logger.info(
                "PAPER TRADE: SELL %s  pnl=%.2f (%.1f%%)  days_held=%d",
                exit_ticker, sell_result["pnl"], sell_result["pnl_pct"] * 100, sell_result["days_held"],
            )
        except Exception as e:
            logger.warning("PAPER TRADE: SELL failed for %s — %s", exit_ticker, e)

    try:
        paper_trader.record_agent_run(
            run_id=run_id,
            regime=regime,
            ticker=best_ticker,
            action=action,
            strategy_type=state.get("strategy_type", "momentum"),
            params=best.get("params", {}),
            sharpe=best.get("sharpe", 0.0),
            total_return=best.get("total_return", 0.0),
            llm_decisions=state.get("llm_decisions", {}),
            run_log=state.get("run_log", []),
        )
        logger.info("PAPER TRADE: Agent run recorded (action=%s, ticker=%s)", action, best_ticker)
    except Exception as e:
        logger.warning("PAPER TRADE: record_agent_run failed — %s", e)

    return state


# ─── Main agent entry point ───────────────────────────────────────────────────

def run_agent(
    ohlcv_data: Dict[str, pd.DataFrame],
    strategy_type: str = "momentum",
    asset: str = None,
    max_iterations: int = None,
) -> AgentState:
    """
    Run the full agent loop:
      discovery → analyze → shortlist → llm_analysis → position_management
      → [hypothesize → backtest → reflect] × N → store

    Phase 2 changes:
    - discovery_node: auto-gates on 100-record minimum in StrategyMemory
    - llm_analysis_node: real Gemini call with HistoricalAggregator context
    - position_management_node: filters by llm_decisions before picking ticker
    - backtest_node: uses position-decided ticker, not config.reference_asset
    """
    state: AgentState = {
        "ohlcv_data": ohlcv_data,
        "features_df": pd.DataFrame(),
        "context": None,
        "proposals": [],
        "results": [],
        "best_result": None,
        "iteration": 0,
        "max_iterations": max_iterations or config.agent.max_iterations,
        "strategy_type": strategy_type,
        "asset": asset or config.reference_asset,
        "should_continue": True,
        "memory_context": "",
        "run_log": [],
        "regime_signals": None,
        "asset_search_results": [],
        "strategies_ranked": [],
        "llm_decisions": {},
        "validated_edges": [],
        "current_position": None,
        "position_decision": {},
        "paper_trade_result": None,
    }

    logger.info("=== AGENT RUN START: strategy=%s  asset=%s ===", strategy_type, state["asset"])

    # Step 0: Discovery (self-gates: skips if < 100 records in memory)
    state = discovery_node(state)

    # Step 1: Analyze market + search candidates
    state = analyze_node(state)

    # Step 2: Shortlist strategies by historical Sharpe
    state = shortlist_node(state)

    # Step 3: LLM risk/reward analysis of top-N candidates
    state = llm_analysis_node(state)

    # Step 4: Position management (hold/exit/switch/buy/skip)
    state = position_management_node(state)

    # Step 5-7: Hypothesize → Backtest → Reflect (only if not exiting)
    if state["should_continue"]:
        while state["should_continue"] and state["iteration"] < state["max_iterations"]:
            state = hypothesize_node(state)
            state = backtest_node(state)
            state = reflect_node(state)
        state = store_node(state)
    else:
        state["run_log"].append("Agent run ended: position management decision was EXIT.")

    # Step 8: Paper trading — always runs regardless of EXIT/SKIP/HOLD
    state = _execute_paper_trades(state)

    logger.info("=== AGENT RUN COMPLETE ===")
    logger.info("Run log:")
    for line in state.get("run_log", []):
        logger.info("  %s", line)

    return state
