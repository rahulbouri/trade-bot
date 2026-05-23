"""
simulate_history.py — Replay N trading days through the full agent + paper trader pipeline.

For each day d in the last N trading days:
  1. Slice all parquet data to rows <= d  (masks the future)
  2. Map VIX parquet (keyed "VIX") → "^VIX" so compute_features finds it
  3. Run run_agent() — discovery → analyze → shortlist → llm_analysis →
     position_management → hypothesize → backtest → store + paper trade
  4. Print a one-line summary per day

Run from project root:
  python scripts/simulate_history.py [--days N]
"""

import argparse
import logging
import os
import sqlite3
import sys
from pathlib import Path

import pandas as pd

# ── Path setup ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logging import setup_logging
from src.utils.config import config
from src.trading.paper_trader import PaperTrader

setup_logging("INFO")
logger = logging.getLogger("simulate_history")

# Suppress noisy sub-loggers during simulation
for noisy in ("httpx", "httpcore", "google", "urllib3"):
    logging.getLogger(noisy).setLevel(logging.WARNING)


# ── Reset helpers ─────────────────────────────────────────────────────────────

def _reset_db(path: str, label: str) -> None:
    p = Path(path)
    if p.exists():
        p.unlink()
        logger.info("RESET: Deleted %s (%s)", path, label)
    else:
        logger.info("RESET: %s not found (already clean)", path)


def reset_all_dbs() -> None:
    """Wipe paper trader, position manager, and strategy memory DBs."""
    _reset_db(".cache/paper_trades.db", "PaperTrader")
    _reset_db(".cache/position_manager.db", "PositionManager")

    # Reset StrategyMemory (experiments/results.db) — delete all rows, keep schema
    results_path = Path(config.results_db_path)
    if results_path.exists():
        with sqlite3.connect(str(results_path)) as conn:
            tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
            for table in tables:
                conn.execute(f"DELETE FROM {table}")
            try:
                conn.execute("DELETE FROM sqlite_sequence")
            except Exception:
                pass
        logger.info("RESET: Cleared StrategyMemory (%s) tables=%s", results_path, tables)
    else:
        logger.info("RESET: StrategyMemory DB not found (will be created fresh)")


# ── Data loading ──────────────────────────────────────────────────────────────

def load_all_parquet() -> dict[str, pd.DataFrame]:
    """Load every parquet file from data_store/ into a dict keyed by ticker."""
    data_path = Path(config.data_path)
    raw: dict[str, pd.DataFrame] = {}
    for f in sorted(data_path.glob("*.parquet")):
        stem = f.stem  # e.g. "SPY", "VIX", "GLD"
        df = pd.read_parquet(f)
        raw[stem] = df
    logger.info("Loaded parquets: %s", list(raw.keys()))
    return raw


def slice_to_day(raw: dict[str, pd.DataFrame], cutoff: pd.Timestamp) -> dict[str, pd.DataFrame]:
    """
    Return a copy of raw where every DataFrame is sliced to rows <= cutoff.

    Also remaps the "VIX" key → "^VIX" (the ticker config uses) so that
    compute_features() can find it.
    """
    sliced: dict[str, pd.DataFrame] = {}
    for ticker, df in raw.items():
        sliced_df = df[df.index <= cutoff]
        if ticker == "VIX":
            sliced["^VIX"] = sliced_df
        else:
            sliced[ticker] = sliced_df
    return sliced


# ── Simulation loop ───────────────────────────────────────────────────────────

def run_simulation(n_days: int = 10) -> None:
    logger.info("=" * 60)
    logger.info("SIMULATION: %d days  |  real Gemini calls  |  fresh DBs", n_days)
    logger.info("=" * 60)

    reset_all_dbs()

    raw = load_all_parquet()

    # Trading dates from SPY (the reference asset)
    all_dates = raw["SPY"].index
    sim_dates = all_dates[-n_days:]

    logger.info(
        "Simulating %d days: %s → %s",
        n_days,
        sim_dates[0].date(),
        sim_dates[-1].date(),
    )

    from src.agent.agent_graph import run_agent

    results_summary: list[dict] = []

    for i, sim_date in enumerate(sim_dates, 1):
        logger.info("")
        logger.info("── DAY %d/%d  (%s) ──────────────────────────────", i, n_days, sim_date.date())

        ohlcv = slice_to_day(raw, sim_date)

        # Sanity check: each ticker should have data up to sim_date
        for ticker, df in ohlcv.items():
            if df.empty:
                logger.warning("  %s: empty slice for %s", ticker, sim_date.date())

        try:
            state = run_agent(
                ohlcv_data=ohlcv,
                strategy_type="momentum",
                asset=config.reference_asset,
            )

            best = state.get("best_result") or {}
            decision = state.get("position_decision", {})
            action = decision.get("action", "SKIP")
            ticker_decided = decision.get("new_ticker") or decision.get("current_ticker") or "—"
            sharpe = best.get("sharpe", 0.0)
            ret = best.get("total_return", 0.0)
            paper = state.get("paper_trade_result")

            paper_note = ""
            if paper:
                if paper.get("action") == "BUY":
                    paper_note = f"  → PAPER BUY {paper['ticker']} qty={paper['quantity']} @ {paper['price']:.2f}"
                elif paper.get("action") == "SELL":
                    paper_note = f"  → PAPER SELL {paper['ticker']} P&L={paper['pnl']:+.2f}"

            logger.info(
                "  RESULT: action=%-8s ticker=%-6s sharpe=%.3f  return=%.1f%%%s",
                action, ticker_decided, sharpe, ret * 100, paper_note,
            )

            results_summary.append({
                "date": str(sim_date.date()),
                "action": action,
                "ticker": ticker_decided,
                "sharpe": round(sharpe, 3),
                "return_pct": round(ret * 100, 2),
            })

        except Exception as e:
            logger.exception("  FAILED on %s: %s", sim_date.date(), e)
            results_summary.append({"date": str(sim_date.date()), "action": "ERROR", "ticker": "—", "sharpe": 0, "return_pct": 0})

    # ── Final summary ─────────────────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info("SIMULATION COMPLETE")
    logger.info("=" * 60)
    logger.info("")

    logger.info("Day-by-day summary:")
    for r in results_summary:
        logger.info("  %s  %-8s  %-6s  Sharpe=%.3f  Return=%+.1f%%",
                    r["date"], r["action"], r["ticker"], r["sharpe"], r["return_pct"])

    pt = PaperTrader()
    snap = pt.get_portfolio_snapshot()
    logger.info("")
    logger.info("PAPER PORTFOLIO FINAL STATE:")
    logger.info("  Cash:           %s", f"₹{snap['cash']:,.2f}")
    logger.info("  Invested:       %s", f"₹{snap['invested_value']:,.2f}")
    logger.info("  Total value:    %s", f"₹{snap['total_value']:,.2f}")
    logger.info("  Total P&L:      %s  (%s)", f"₹{snap['total_pnl']:+,.2f}", f"{snap['total_pnl_pct']:+.2%}")
    logger.info("  Open positions: %d", len(snap["open_positions"]))
    logger.info("  Closed trades:  %d", snap["num_trades"])

    if snap["open_positions"]:
        logger.info("")
        logger.info("  Open positions detail:")
        for pos in snap["open_positions"]:
            logger.info(
                "    %s  entry=%s  qty=%d  cost_basis=%s",
                pos["ticker"],
                f"₹{pos['entry_price']:,.2f}",
                pos["quantity"],
                f"₹{pos['cost_basis']:,.2f}",
            )

    agent_runs_df = pt.get_agent_runs()
    if not agent_runs_df.empty:
        logger.info("")
        logger.info("  Agent runs recorded: %d", len(agent_runs_df))

    logger.info("")
    logger.info("Run the dashboard:")
    logger.info("  python run_app.py")
    logger.info("  → open http://localhost:8501")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate N days of paper trading")
    parser.add_argument("--days", type=int, default=10, help="Number of trading days to simulate (default: 10)")
    args = parser.parse_args()
    run_simulation(n_days=args.days)
