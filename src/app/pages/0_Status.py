"""Status / keepalive page — open this in a browser tab to prevent the app sleeping."""

from datetime import datetime

import streamlit as st

st.set_page_config(page_title="Status — AgentQuant", layout="centered")

# Auto-refresh every 5 minutes (300 seconds) via HTML meta tag
st.markdown(
    '<meta http-equiv="refresh" content="300">',
    unsafe_allow_html=True,
)

st.title("AgentQuant — Status")

now = datetime.utcnow()
st.success(f"App is running — {now.strftime('%Y-%m-%d %H:%M:%S')} UTC")

from src.trading.paper_trader import PaperTrader
try:
    pt = PaperTrader()
    snap = pt.get_portfolio_snapshot()
    runs = pt.get_agent_runs()

    col1, col2, col3 = st.columns(3)
    col1.metric("Portfolio Value", f"₹{snap['total_value']:,.0f}")
    col2.metric("Open Positions", len(snap["open_positions"]))
    col3.metric("Agent Runs", len(runs))

    st.caption("This page refreshes every 5 minutes to keep the app alive.")
except Exception:
    st.info("No portfolio data yet.")
    st.caption("This page refreshes every 5 minutes to keep the app alive.")
