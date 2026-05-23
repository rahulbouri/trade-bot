"""Background agent runner — module-level singleton shared across all Streamlit pages."""

import threading
from datetime import datetime, timezone
from typing import Any, Dict

# Module-level state: persists across page navigations in the same Streamlit process
_state: Dict[str, Any] = {
    "running": False,
    "started_at": None,
    "completed_at": None,
    "action": None,
    "error": None,
}
_lock = threading.Lock()


def is_running() -> bool:
    return _state["running"]


def get_state() -> Dict[str, Any]:
    with _lock:
        return dict(_state)


def start_agent_run() -> bool:
    """Fire agent in a daemon thread. Returns False if already running."""
    with _lock:
        if _state["running"]:
            return False
        _state.update({
            "running": True,
            "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "completed_at": None,
            "action": None,
            "error": None,
        })

    thread = threading.Thread(target=_run_agent, daemon=True, name="agent-runner")
    thread.start()
    return True


def _run_agent() -> None:
    try:
        from dotenv import load_dotenv

        from src.agent.agent_graph import run_agent
        from src.data.ingest import fetch_ohlcv_data
        from src.utils.config import config

        load_dotenv()
        ohlcv_data = fetch_ohlcv_data()
        strategy_type = config.strategies[0].name if config.strategies else "momentum"
        result_state = run_agent(
            ohlcv_data=ohlcv_data,
            strategy_type=strategy_type,
            asset=config.reference_asset,
        )
        action = result_state.get("position_decision", {}).get("action", "N/A")
        with _lock:
            _state["action"] = action
    except Exception as exc:  # noqa: BLE001
        with _lock:
            _state["error"] = str(exc)
    finally:
        with _lock:
            _state["running"] = False
            _state["completed_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
