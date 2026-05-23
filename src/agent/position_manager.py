"""Position Manager — Track overnight positions, make daily hold/exit/switch decisions."""

import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

_HOLD_CONFIDENCE_THRESHOLD = 0.70
_SWITCH_CONFIDENCE_THRESHOLD = 0.80
_BUY_CONFIDENCE_THRESHOLD = 0.50
_MAX_POSITION_AGE_DAYS = 5


@dataclass
class Position:
    """A currently held overnight position."""
    ticker: str
    entry_date: str
    entry_price: float
    entry_reason: str
    days_held: int

    @classmethod
    def from_row(cls, row: tuple) -> Optional["Position"]:
        """Reconstruct Position from DB row."""
        if not row:
            return None
        return cls(
            ticker=row[0],
            entry_date=row[1],
            entry_price=row[2],
            entry_reason=row[3],
            days_held=row[4],
        )


@dataclass
class Decision:
    """A hold/exit/switch/buy/skip decision."""
    timestamp: str
    action: str
    current_ticker: Optional[str]
    new_ticker: Optional[str]
    reasoning: str
    confidence: float
    agent_signal_strength: float

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to state dict."""
        return {
            "timestamp": self.timestamp,
            "action": self.action,
            "current_ticker": self.current_ticker,
            "new_ticker": self.new_ticker,
            "reasoning": self.reasoning,
            "confidence": self.confidence,
            "agent_signal_strength": self.agent_signal_strength,
        }


class PositionManager:
    """Track overnight positions and make daily hold/exit/switch decisions."""

    def __init__(self, db_path: str = ".cache/position_manager.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        """Create tables if not exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS current_position (
                    id INTEGER PRIMARY KEY,
                    ticker TEXT NOT NULL,
                    entry_date TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    entry_reason TEXT NOT NULL,
                    days_held INTEGER NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS decision_history (
                    id INTEGER PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    action TEXT NOT NULL,
                    current_ticker TEXT,
                    new_ticker TEXT,
                    reasoning TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    agent_signal_strength REAL NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS exit_history (
                    id INTEGER PRIMARY KEY,
                    ticker TEXT NOT NULL,
                    exit_date TEXT NOT NULL,
                    exit_price REAL NOT NULL,
                    days_held INTEGER NOT NULL,
                    entry_price REAL NOT NULL,
                    daily_pnl REAL NOT NULL,
                    daily_pnl_pct REAL NOT NULL
                )
            """)
            conn.commit()

    def get_current_position(self) -> Optional[Position]:
        """Fetch position held from overnight (if any)."""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT ticker, entry_date, entry_price, entry_reason, days_held "
                "FROM current_position ORDER BY updated_at DESC LIMIT 1"
            ).fetchone()
        return Position.from_row(row)

    def should_hold_current(
        self,
        current_position: Optional[Position],
        new_agent_signal: Dict[str, Any],
    ) -> bool:
        """Decide whether to hold current position based on signal strength/direction."""
        # Rule 1: No position to hold
        if current_position is None:
            return False

        new_ticker = new_agent_signal.get("ticker")
        confidence = new_agent_signal.get("confidence", 0.0)

        # Rule 2: Position aged out → EXIT (checked before same-ticker rule)
        if current_position.days_held > _MAX_POSITION_AGE_DAYS:
            return False

        # Rule 3: Same ticker + strong confidence → HOLD
        if new_ticker == current_position.ticker and confidence > _HOLD_CONFIDENCE_THRESHOLD:
            return True

        # Rule 4: New signal too weak → HOLD (conservative)
        if confidence < _BUY_CONFIDENCE_THRESHOLD:
            return True

        # Rule 5: Different ticker + very high confidence → SWITCH
        if new_ticker != current_position.ticker and confidence > _SWITCH_CONFIDENCE_THRESHOLD:
            return False

        # Default: HOLD
        return True

    def execute_decision(
        self,
        current_position: Optional[Position],
        agent_signal: Dict[str, Any],
        ohlcv_data: Dict[str, pd.DataFrame],
    ) -> Decision:
        """Generate decision object with action and reasoning (does NOT update DB)."""
        now = datetime.now(timezone.utc).isoformat()
        ticker = agent_signal.get("ticker")
        confidence = agent_signal.get("confidence", 0.0)
        reasoning_hint = agent_signal.get("reasoning", "")
        signal_strength = agent_signal.get("signal_strength", 0.0)

        # Case 1: No current position
        if current_position is None:
            if confidence > _BUY_CONFIDENCE_THRESHOLD:
                return Decision(
                    timestamp=now,
                    action="BUY",
                    current_ticker=None,
                    new_ticker=ticker,
                    reasoning=f"No current position. New signal {ticker} at {confidence:.0%}. {reasoning_hint}",
                    confidence=confidence,
                    agent_signal_strength=signal_strength,
                )
            else:
                return Decision(
                    timestamp=now,
                    action="SKIP",
                    current_ticker=None,
                    new_ticker=None,
                    reasoning="No position + new signal too weak (<50%)",
                    confidence=confidence,
                    agent_signal_strength=signal_strength,
                )

        # Case 2: Current position exists
        should_hold = self.should_hold_current(current_position, agent_signal)

        if should_hold:
            return Decision(
                timestamp=now,
                action="HOLD",
                current_ticker=current_position.ticker,
                new_ticker=None,
                reasoning=(
                    f"Holding {current_position.ticker} ({current_position.days_held}d old). "
                    "New signal weak or same."
                ),
                confidence=confidence,
                agent_signal_strength=signal_strength,
            )
        elif ticker != current_position.ticker and confidence > _HOLD_CONFIDENCE_THRESHOLD:
            return Decision(
                timestamp=now,
                action="SWITCH",
                current_ticker=current_position.ticker,
                new_ticker=ticker,
                reasoning=f"Switch from {current_position.ticker} to {ticker} ({confidence:.0%})",
                confidence=confidence,
                agent_signal_strength=signal_strength,
            )
        else:
            return Decision(
                timestamp=now,
                action="EXIT",
                current_ticker=current_position.ticker,
                new_ticker=None,
                reasoning=(
                    f"Exit {current_position.ticker} ({current_position.days_held}d old). "
                    "New signal too weak."
                ),
                confidence=confidence,
                agent_signal_strength=signal_strength,
            )

    def store_decision(self, decision: Decision, entry_price: float) -> None:
        """Persist decision to SQLite (decision_history + current_position)."""
        now = datetime.now(timezone.utc).isoformat()
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        with sqlite3.connect(self.db_path) as conn:
            # 1. Append to decision_history (immutable audit trail)
            conn.execute(
                """INSERT INTO decision_history
                   (timestamp, action, current_ticker, new_ticker, reasoning, confidence, agent_signal_strength)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    decision.timestamp,
                    decision.action,
                    decision.current_ticker,
                    decision.new_ticker,
                    decision.reasoning,
                    decision.confidence,
                    decision.agent_signal_strength,
                ),
            )

            # 2. Fetch previous position for exit/hold reference
            prev_row = conn.execute(
                "SELECT ticker, entry_date, entry_price, entry_reason, days_held "
                "FROM current_position ORDER BY updated_at DESC LIMIT 1"
            ).fetchone()
            prev_position = Position.from_row(prev_row) if prev_row else None

            # 3. Clear current position
            conn.execute("DELETE FROM current_position")

            # 4. Record exit if applicable
            if decision.action in ("EXIT", "SWITCH") and decision.current_ticker:
                if prev_position and prev_position.entry_price > 0:
                    daily_pnl = entry_price - prev_position.entry_price
                    daily_pnl_pct = daily_pnl / prev_position.entry_price
                    conn.execute(
                        """INSERT INTO exit_history
                           (ticker, exit_date, exit_price, days_held, entry_price, daily_pnl, daily_pnl_pct)
                           VALUES (?, ?, ?, ?, ?, ?, ?)""",
                        (
                            decision.current_ticker,
                            today,
                            entry_price,
                            prev_position.days_held,
                            prev_position.entry_price,
                            daily_pnl,
                            daily_pnl_pct,
                        ),
                    )

            # 5. Insert new current_position for BUY or SWITCH
            if decision.action in ("BUY", "SWITCH") and decision.new_ticker:
                conn.execute(
                    """INSERT INTO current_position
                       (ticker, entry_date, entry_price, entry_reason, days_held, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (decision.new_ticker, today, entry_price, "agent_signal", 0, now),
                )
            elif decision.action == "HOLD" and decision.current_ticker:
                days_held = (prev_position.days_held + 1) if prev_position else 1
                original_entry_date = prev_position.entry_date if prev_position else today
                original_entry_price = prev_position.entry_price if prev_position else entry_price
                conn.execute(
                    """INSERT INTO current_position
                       (ticker, entry_date, entry_price, entry_reason, days_held, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (
                        decision.current_ticker,
                        original_entry_date,
                        original_entry_price,
                        "hold_from_yesterday",
                        days_held,
                        now,
                    ),
                )

            conn.commit()
        logger.debug("Stored decision: %s (%s → %s)", decision.action, decision.current_ticker, decision.new_ticker)

    def get_decision_history(self, limit: int = 30) -> List[Decision]:
        """Fetch recent decisions for analysis (newest first)."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT timestamp, action, current_ticker, new_ticker, reasoning, confidence, agent_signal_strength "
                "FROM decision_history ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            ).fetchall()

        return [
            Decision(
                timestamp=row[0],
                action=row[1],
                current_ticker=row[2],
                new_ticker=row[3],
                reasoning=row[4],
                confidence=row[5],
                agent_signal_strength=row[6],
            )
            for row in rows
        ]
