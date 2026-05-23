"""Agent Runs page — run history, LLM reasoning, Run Agent Now button."""

import json

import pandas as pd
import streamlit as st

from src.trading.paper_trader import PaperTrader

st.set_page_config(page_title="Agent Runs — AgentQuant", layout="wide")
st.title("Agent Run History")

# ── Run Agent button ──────────────────────────────────────────────────────────
st.subheader("Run Agent")
col_btn, _ = st.columns([1, 3])
with col_btn:
    if st.button("🚀 Run Agent Now", type="primary", use_container_width=True):
        with st.spinner("Running agent — this takes ~30 seconds..."):
            try:
                from dotenv import load_dotenv
                from src.agent.agent_graph import run_agent
                from src.data.ingest import fetch_ohlcv_data
                from src.utils.config import config

                load_dotenv()
                ohlcv_data = fetch_ohlcv_data()
                strategy_type = config.strategies[0].name if config.strategies else "momentum"
                state = run_agent(
                    ohlcv_data=ohlcv_data,
                    strategy_type=strategy_type,
                    asset=config.reference_asset,
                )
                action = state.get("position_decision", {}).get("action", "N/A")
                st.success(f"Agent run complete! Action: {action}")
            except Exception as e:
                st.error(f"Agent run failed: {e}")
        st.rerun()

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
            # Run log
            st.markdown("**Run Log**")
            try:
                log_lines = json.loads(row["run_log"]) if isinstance(row["run_log"], str) else row["run_log"]
                for line in (log_lines or []):
                    st.markdown(f"- {line}")
            except Exception:
                st.text(row["run_log"])

            # LLM decisions
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

            # Params
            st.markdown("**Parameters**")
            try:
                params = json.loads(row["params"]) if isinstance(row["params"], str) else row["params"]
                st.json(params)
            except Exception:
                st.text(row["params"])
