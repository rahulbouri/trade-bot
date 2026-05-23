"""Agent Runs page — run history, LLM reasoning, Run Agent Now button."""

import sys
import time
from pathlib import Path

_repo_root = Path(__file__).parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import json

import pandas as pd
import streamlit as st

from src.app.background_runner import get_state, is_running, start_agent_run
from src.trading.paper_trader import PaperTrader

st.set_page_config(page_title="Agent Runs — AgentQuant", layout="wide")
st.title("Agent Run History")

# ── Run Agent button ──────────────────────────────────────────────────────────
st.subheader("Run Agent")

runner_state = get_state()

if runner_state["running"]:
    started = runner_state.get("started_at", "")
    st.warning(f"Agent is running... (started {started} UTC). You can switch pages freely.")
    # Auto-refresh this page every 5s while the agent is running
    time.sleep(5)
    st.rerun()
else:
    col_btn, _ = st.columns([1, 3])
    with col_btn:
        clicked = st.button("Run Agent Now", type="primary", use_container_width=True)

    # Guard: only fire on the *first* rerun after the click, not on subsequent reruns.
    # st.button returns True on the rerun immediately following the click; session state
    # prevents a second fire if the page reruns again before the thread sets running=True.
    if clicked and not st.session_state.get("_agent_just_fired"):
        st.session_state["_agent_just_fired"] = True
        start_agent_run()
        st.rerun()
    elif not clicked:
        st.session_state.pop("_agent_just_fired", None)

    # Show result of last run
    if runner_state["completed_at"]:
        if runner_state["error"]:
            st.error(f"Last run failed at {runner_state['completed_at']} UTC: {runner_state['error']}")
        else:
            st.success(
                f"Last run completed at {runner_state['completed_at']} UTC — "
                f"Action: **{runner_state['action']}**"
            )

st.divider()

# ── Agent Runs table ──────────────────────────────────────────────────────────
st.subheader("Run History")
pt = PaperTrader()
runs_df = pt.get_agent_runs()

if runs_df.empty:
    st.info("No agent runs yet. Click **Run Agent Now** to start.")
else:
    display_df = runs_df[["timestamp", "regime", "ticker", "action", "sharpe", "total_return"]].copy()
    display_df["total_return"] = display_df["total_return"].map(lambda v: f"{v:.1%}")
    display_df["sharpe"] = display_df["sharpe"].map(lambda v: f"{v:.3f}")
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.divider()

    # ── Run Detail expanders (last 10) ────────────────────────────────────────
    st.subheader("Run Details")
    for _, row in runs_df.head(10).iterrows():
        label = f"{row['timestamp']} — {row['action']} {row.get('ticker', 'N/A')} (Sharpe {float(row['sharpe']):.3f})"
        with st.expander(label):
            st.markdown("**Run Log**")
            try:
                log_lines = json.loads(row["run_log"]) if isinstance(row["run_log"], str) else row["run_log"]
                for line in (log_lines or []):
                    st.markdown(f"- {line}")
            except Exception:
                st.text(row["run_log"])

            st.markdown("**LLM Decisions**")
            try:
                decisions = json.loads(row["llm_decisions"]) if isinstance(row["llm_decisions"], str) else row["llm_decisions"]
                if decisions:
                    dec_df = pd.DataFrame(
                        [{"Ticker": t, "LLM Decision": d} for t, d in decisions.items()]
                    )
                    st.dataframe(dec_df, use_container_width=True, hide_index=True)
                else:
                    st.caption("No LLM decisions recorded.")
            except Exception:
                st.text(row["llm_decisions"])

            st.markdown("**Parameters**")
            try:
                params = json.loads(row["params"]) if isinstance(row["params"], str) else row["params"]
                st.json(params)
            except Exception:
                st.text(row["params"])
