"""Portfolio page — current positions, P&L, performance metrics."""

import pandas as pd
import streamlit as st

from src.trading.paper_trader import PaperTrader

st.set_page_config(page_title="Portfolio — AgentQuant", layout="wide")
st.title("Current Portfolio")

pt = PaperTrader()
snap = pt.get_portfolio_snapshot()
metrics = pt.get_performance_metrics()

# Last updated timestamp
runs_df = pt.get_agent_runs()
last_updated = runs_df["timestamp"].iloc[0] if not runs_df.empty else "Never"
st.caption(f"Last updated: {last_updated}")

# ── Top metrics row ───────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric(
    "Total Value",
    f"₹{snap['total_value']:,.0f}",
    delta=f"₹{snap['total_value'] - snap['initial_capital']:+,.0f}",
)
c2.metric(
    "Total P&L",
    f"₹{snap['total_pnl']:+,.0f}",
    delta=f"{snap['total_pnl_pct']:+.1%}",
)
c3.metric("Win Rate", f"{metrics['win_rate']:.0%}")
c4.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")

st.divider()

# ── Open Positions table ──────────────────────────────────────────────────────
st.subheader("Open Positions")
open_positions = snap["open_positions"]

if not open_positions:
    st.info("No open positions. Run the agent to generate trades.")
else:
    # Fetch current prices for mark-to-market
    @st.cache_data(ttl=3600, show_spinner="Fetching current prices…")
    def _get_current_prices(tickers: tuple):
        from src.data.ingest import fetch_ohlcv_data
        prices = {}
        for ticker in tickers:
            try:
                data = fetch_ohlcv_data(ticker)
                if ticker in data and not data[ticker].empty:
                    prices[ticker] = float(data[ticker]["Close"].iloc[-1])
            except Exception:
                pass
        return prices

    tickers = tuple(p["ticker"] for p in open_positions)
    current_prices = _get_current_prices(tickers)

    # Rebuild snapshot with live prices
    snap_live = pt.get_portfolio_snapshot(current_prices)
    rows = []
    for pos in snap_live["open_positions"]:
        rows.append({
            "Ticker": pos["ticker"],
            "Entry Date": pos["entry_date"],
            "Entry Price": f"₹{pos['entry_price']:.2f}",
            "Current Price": f"₹{pos.get('current_price', pos['entry_price']):.2f}",
            "Qty": pos["quantity"],
            "Cost Basis": f"₹{pos['cost_basis']:,.2f}",
            "Unrealized P&L": pos.get("unrealized_pnl", 0.0),
            "P&L %": pos.get("unrealized_pnl_pct", 0.0),
        })

    df_pos = pd.DataFrame(rows)
    st.dataframe(
        df_pos.style.format({
            "Unrealized P&L": "₹{:+,.2f}",
            "P&L %": "{:+.1%}",
        }).applymap(
            lambda v: "color: green" if isinstance(v, (int, float)) and v > 0
            else ("color: red" if isinstance(v, (int, float)) and v < 0 else ""),
            subset=["Unrealized P&L", "P&L %"],
        ),
        use_container_width=True,
    )

st.divider()

# ── P&L Over Time chart ───────────────────────────────────────────────────────
st.subheader("P&L Over Time")
trades_df = pt.get_trade_history()

if trades_df.empty:
    st.info("No closed trades yet.")
else:
    trades_df = trades_df.sort_values("exit_date")
    trades_df["cumulative_pnl"] = trades_df["pnl"].cumsum()
    chart_df = trades_df.set_index("exit_date")[["cumulative_pnl"]]
    st.line_chart(chart_df, y="cumulative_pnl")

st.divider()

# ── Performance Summary ───────────────────────────────────────────────────────
st.subheader("Performance Summary")
summary_rows = []
labels = {
    "total_pnl": ("Total P&L", lambda v: f"₹{v:+,.2f}"),
    "total_pnl_pct": ("Total P&L %", lambda v: f"{v:+.2%}"),
    "num_trades": ("Number of Trades", lambda v: str(v)),
    "win_rate": ("Win Rate", lambda v: f"{v:.0%}"),
    "avg_pnl_per_trade": ("Avg P&L / Trade", lambda v: f"₹{v:+,.2f}"),
    "avg_days_held": ("Avg Days Held", lambda v: f"{v:.1f}"),
    "best_trade_pnl": ("Best Trade P&L", lambda v: f"₹{v:+,.2f}"),
    "worst_trade_pnl": ("Worst Trade P&L", lambda v: f"₹{v:+,.2f}"),
    "sharpe_ratio": ("Sharpe Ratio", lambda v: f"{v:.2f}"),
    "max_drawdown": ("Max Drawdown", lambda v: f"{v:.2%}"),
}
for key, (label, fmt) in labels.items():
    summary_rows.append({"Metric": label, "Value": fmt(metrics.get(key, 0))})

st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)
