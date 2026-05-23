"""Paper Trader — SQLite-backed virtual portfolio for paper trading simulation."""

import json
import logging
import math
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

_DEFAULT_DB_PATH = ".cache/paper_trades.db"
_DEFAULT_CAPITAL = 100_000.0
_COMMISSION_PCT = 0.001
_MAX_POSITION_SIZE_PCT = 0.20


class PaperTrader:
    """SQLite-backed virtual portfolio for paper trading simulation."""

    def __init__(self, db_path: str = _DEFAULT_DB_PATH, initial_capital: float = _DEFAULT_CAPITAL):
        self._db_path = db_path
        self._initial_capital = initial_capital
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS portfolio (
                    id INTEGER PRIMARY KEY,
                    cash REAL NOT NULL,
                    initial_capital REAL NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS open_positions (
                    id INTEGER PRIMARY KEY,
                    ticker TEXT NOT NULL UNIQUE,
                    entry_date TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    quantity INTEGER NOT NULL,
                    cost_basis REAL NOT NULL,
                    strategy_type TEXT NOT NULL,
                    params TEXT NOT NULL,
                    sharpe_at_entry REAL NOT NULL,
                    regime_at_entry TEXT NOT NULL,
                    reason TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY,
                    ticker TEXT NOT NULL,
                    entry_date TEXT NOT NULL,
                    exit_date TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    exit_price REAL NOT NULL,
                    quantity INTEGER NOT NULL,
                    pnl REAL NOT NULL,
                    pnl_pct REAL NOT NULL,
                    days_held INTEGER NOT NULL,
                    strategy_type TEXT NOT NULL,
                    params TEXT NOT NULL,
                    sharpe_at_entry REAL NOT NULL,
                    regime_at_entry TEXT NOT NULL,
                    exit_reason TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS agent_runs (
                    id INTEGER PRIMARY KEY,
                    run_id TEXT NOT NULL UNIQUE,
                    timestamp TEXT NOT NULL,
                    regime TEXT NOT NULL,
                    ticker TEXT,
                    action TEXT NOT NULL,
                    strategy_type TEXT NOT NULL,
                    params TEXT NOT NULL,
                    sharpe REAL NOT NULL,
                    total_return REAL NOT NULL,
                    llm_decisions TEXT NOT NULL,
                    run_log TEXT NOT NULL
                );
            """)
            # Seed portfolio row if empty
            row = conn.execute("SELECT COUNT(*) FROM portfolio").fetchone()
            if row[0] == 0:
                now = self._today()
                conn.execute(
                    "INSERT INTO portfolio (cash, initial_capital, created_at, updated_at) VALUES (?, ?, ?, ?)",
                    (self._initial_capital, self._initial_capital, now, now),
                )

    def _today(self) -> str:
        return datetime.utcnow().date().isoformat()

    def _compute_quantity(self, price: float) -> int:
        cash = self.get_cash()
        max_spend = min(cash * _MAX_POSITION_SIZE_PCT, cash)
        return math.floor(max_spend / price)

    def get_cash(self) -> float:
        with sqlite3.connect(self._db_path) as conn:
            row = conn.execute("SELECT cash FROM portfolio WHERE id = 1").fetchone()
            return row[0] if row else 0.0

    def get_open_position(self, ticker: str) -> Optional[Dict[str, Any]]:
        with sqlite3.connect(self._db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM open_positions WHERE ticker = ?", (ticker,)
            ).fetchone()
            return dict(row) if row else None

    def get_all_open_positions(self) -> List[Dict[str, Any]]:
        with sqlite3.connect(self._db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("SELECT * FROM open_positions").fetchall()
            return [dict(r) for r in rows]

    def execute_buy(
        self,
        ticker: str,
        price: float,
        strategy_type: str,
        params: Dict[str, Any],
        sharpe_at_entry: float,
        regime: str,
        reason: str,
    ) -> Dict[str, Any]:
        """Buy position in ticker at given price."""
        if price <= 0:
            raise ValueError(f"Price must be > 0, got {price}")

        if self.get_open_position(ticker) is not None:
            raise ValueError(f"Position already open for {ticker}")

        quantity = self._compute_quantity(price)
        if quantity == 0:
            raise ValueError(f"Insufficient cash to buy {ticker} at {price}")

        commission = price * quantity * _COMMISSION_PCT
        cost_basis = price * quantity + commission
        today = self._today()

        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                """INSERT INTO open_positions
                   (ticker, entry_date, entry_price, quantity, cost_basis,
                    strategy_type, params, sharpe_at_entry, regime_at_entry, reason)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (ticker, today, price, quantity, cost_basis,
                 strategy_type, json.dumps(params), sharpe_at_entry, regime, reason),
            )
            cash_before = conn.execute("SELECT cash FROM portfolio WHERE id = 1").fetchone()[0]
            cash_after = cash_before - cost_basis
            conn.execute(
                "UPDATE portfolio SET cash = ?, updated_at = ? WHERE id = 1",
                (cash_after, today),
            )

        logger.info("BUY %s  qty=%d  price=%.4f  commission=%.2f  cash_after=%.2f",
                    ticker, quantity, price, commission, cash_after)
        return {
            "action": "BUY",
            "ticker": ticker,
            "quantity": quantity,
            "price": price,
            "commission": commission,
            "cash_after": cash_after,
        }

    def execute_sell(
        self,
        ticker: str,
        exit_price: float,
        exit_reason: str,
    ) -> Dict[str, Any]:
        """Sell open position in ticker at exit_price."""
        position = self.get_open_position(ticker)
        if position is None:
            raise ValueError(f"No open position for {ticker}")

        quantity = position["quantity"]
        cost_basis = position["cost_basis"]
        entry_date = position["entry_date"]
        today = self._today()

        entry_dt = datetime.fromisoformat(entry_date)
        exit_dt = datetime.fromisoformat(today)
        days_held = max((exit_dt - entry_dt).days, 0)

        commission = exit_price * quantity * _COMMISSION_PCT
        gross_proceeds = exit_price * quantity - commission
        pnl = gross_proceeds - cost_basis
        pnl_pct = pnl / cost_basis if cost_basis > 0 else 0.0

        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                """INSERT INTO trades
                   (ticker, entry_date, exit_date, entry_price, exit_price, quantity,
                    pnl, pnl_pct, days_held, strategy_type, params, sharpe_at_entry,
                    regime_at_entry, exit_reason)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (ticker, entry_date, today, position["entry_price"], exit_price, quantity,
                 pnl, pnl_pct, days_held, position["strategy_type"], position["params"],
                 position["sharpe_at_entry"], position["regime_at_entry"], exit_reason),
            )
            conn.execute("DELETE FROM open_positions WHERE ticker = ?", (ticker,))
            cash_before = conn.execute("SELECT cash FROM portfolio WHERE id = 1").fetchone()[0]
            cash_after = cash_before + gross_proceeds
            conn.execute(
                "UPDATE portfolio SET cash = ?, updated_at = ? WHERE id = 1",
                (cash_after, today),
            )

        logger.info("SELL %s  pnl=%.2f (%.1f%%)  days_held=%d",
                    ticker, pnl, pnl_pct * 100, days_held)
        return {
            "action": "SELL",
            "ticker": ticker,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "days_held": days_held,
        }

    def record_agent_run(
        self,
        run_id: str,
        regime: str,
        ticker: Optional[str],
        action: str,
        strategy_type: str,
        params: Dict[str, Any],
        sharpe: float,
        total_return: float,
        llm_decisions: Dict[str, str],
        run_log: List[str],
    ) -> None:
        """Insert row into agent_runs table. Idempotent on run_id."""
        now = datetime.utcnow().isoformat()
        with sqlite3.connect(self._db_path) as conn:
            try:
                conn.execute(
                    """INSERT INTO agent_runs
                       (run_id, timestamp, regime, ticker, action, strategy_type, params,
                        sharpe, total_return, llm_decisions, run_log)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (run_id, now, regime, ticker, action, strategy_type,
                     json.dumps(params), sharpe, total_return,
                     json.dumps(llm_decisions), json.dumps(run_log)),
                )
            except sqlite3.IntegrityError:
                pass  # Idempotent: ignore duplicate run_id

    def get_portfolio_snapshot(self, current_prices: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """Return current portfolio state."""
        with sqlite3.connect(self._db_path) as conn:
            conn.row_factory = sqlite3.Row
            port_row = conn.execute("SELECT * FROM portfolio WHERE id = 1").fetchone()
            cash = port_row["cash"] if port_row else self._initial_capital
            initial_capital = port_row["initial_capital"] if port_row else self._initial_capital

            trade_count = conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]

        open_positions = self.get_all_open_positions()

        invested_value = 0.0
        enriched_positions = []
        for pos in open_positions:
            ticker = pos["ticker"]
            current_price = (current_prices or {}).get(ticker, pos["entry_price"])
            market_value = current_price * pos["quantity"]
            unrealized_pnl = market_value - pos["cost_basis"]
            unrealized_pnl_pct = unrealized_pnl / pos["cost_basis"] if pos["cost_basis"] > 0 else 0.0
            invested_value += market_value
            enriched_positions.append({
                **pos,
                "current_price": current_price,
                "market_value": market_value,
                "unrealized_pnl": unrealized_pnl,
                "unrealized_pnl_pct": unrealized_pnl_pct,
            })

        total_value = cash + invested_value
        total_pnl = total_value - initial_capital
        total_pnl_pct = total_pnl / initial_capital if initial_capital > 0 else 0.0

        return {
            "cash": cash,
            "initial_capital": initial_capital,
            "invested_value": invested_value,
            "total_value": total_value,
            "total_pnl": total_pnl,
            "total_pnl_pct": total_pnl_pct,
            "open_positions": enriched_positions,
            "num_trades": trade_count,
        }

    def get_trade_history(self) -> pd.DataFrame:
        """Return all closed trades as DataFrame."""
        columns = [
            "ticker", "entry_date", "exit_date", "entry_price", "exit_price",
            "quantity", "pnl", "pnl_pct", "days_held", "strategy_type", "params",
            "sharpe_at_entry", "regime_at_entry", "exit_reason",
        ]
        with sqlite3.connect(self._db_path) as conn:
            rows = conn.execute(
                "SELECT ticker, entry_date, exit_date, entry_price, exit_price, quantity, "
                "pnl, pnl_pct, days_held, strategy_type, params, sharpe_at_entry, "
                "regime_at_entry, exit_reason FROM trades ORDER BY exit_date DESC"
            ).fetchall()

        if not rows:
            return pd.DataFrame(columns=columns)
        return pd.DataFrame(rows, columns=columns)

    def get_agent_runs(self) -> pd.DataFrame:
        """Return all agent run records as DataFrame, sorted newest first."""
        columns = [
            "run_id", "timestamp", "regime", "ticker", "action", "strategy_type",
            "params", "sharpe", "total_return", "llm_decisions", "run_log",
        ]
        with sqlite3.connect(self._db_path) as conn:
            rows = conn.execute(
                "SELECT run_id, timestamp, regime, ticker, action, strategy_type, params, "
                "sharpe, total_return, llm_decisions, run_log FROM agent_runs ORDER BY timestamp DESC"
            ).fetchall()

        if not rows:
            return pd.DataFrame(columns=columns)
        return pd.DataFrame(rows, columns=columns)

    def get_performance_metrics(self, benchmark_returns: Optional[pd.Series] = None) -> Dict[str, Any]:
        """Compute portfolio performance from closed trades."""
        df = self.get_trade_history()

        empty_metrics: Dict[str, Any] = {
            "total_pnl": 0.0,
            "total_pnl_pct": 0.0,
            "num_trades": 0,
            "win_rate": 0.0,
            "avg_pnl_per_trade": 0.0,
            "avg_days_held": 0.0,
            "best_trade_pnl": 0.0,
            "worst_trade_pnl": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
        }

        if df.empty:
            if benchmark_returns is not None:
                empty_metrics["benchmark_return"] = float(benchmark_returns.sum())
            return empty_metrics

        total_pnl = float(df["pnl"].sum())
        with sqlite3.connect(self._db_path) as conn:
            row = conn.execute("SELECT initial_capital FROM portfolio WHERE id = 1").fetchone()
            initial_capital = row[0] if row else self._initial_capital
        total_pnl_pct = total_pnl / initial_capital if initial_capital > 0 else 0.0

        wins = df[df["pnl"] > 0]
        win_rate = len(wins) / len(df) if len(df) > 0 else 0.0

        pnl_series = df["pnl_pct"]
        if len(df) < 2:
            sharpe = 0.0
        else:
            std = pnl_series.std()
            sharpe = float((pnl_series.mean() / std) * (252 ** 0.5)) if std > 0 else 0.0

        cumulative = (1 + pnl_series).cumprod()
        rolling_max = cumulative.cummax()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = float(drawdown.min()) if not drawdown.empty else 0.0

        metrics: Dict[str, Any] = {
            "total_pnl": total_pnl,
            "total_pnl_pct": total_pnl_pct,
            "num_trades": len(df),
            "win_rate": win_rate,
            "avg_pnl_per_trade": float(df["pnl"].mean()),
            "avg_days_held": float(df["days_held"].mean()),
            "best_trade_pnl": float(df["pnl"].max()),
            "worst_trade_pnl": float(df["pnl"].min()),
            "sharpe_ratio": sharpe,
            "max_drawdown": max_drawdown,
        }

        if benchmark_returns is not None:
            metrics["benchmark_return"] = float(benchmark_returns.sum())

        return metrics
