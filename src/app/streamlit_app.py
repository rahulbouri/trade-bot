"""AgentQuant Dashboard — Paper Trading Monitor."""

import streamlit as st

st.set_page_config(
    page_title="AgentQuant",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("AgentQuant — AI Paper Trading Dashboard")
st.markdown("""
Welcome to AgentQuant. Use the sidebar to navigate:

- **Portfolio** — Current positions, P&L, performance metrics
- **Agent Runs** — History of agent decisions and LLM reasoning
- **Trade History** — All completed paper trades
- **Strategy Research** — Backtest explorer and regime analysis
""")

from src.trading.paper_trader import PaperTrader
try:
    pt = PaperTrader()
    snap = pt.get_portfolio_snapshot()
    st.sidebar.metric("Portfolio Value", f"₹{snap['total_value']:,.0f}",
                      delta=f"₹{snap['total_pnl']:+,.0f}")
    st.sidebar.metric("Cash Available", f"₹{snap['cash']:,.0f}")
    st.sidebar.metric("Open Positions", str(len(snap["open_positions"])))
except Exception:
    st.sidebar.info("No portfolio data yet. Run the agent first.")
