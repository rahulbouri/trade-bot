"""History page — trade history (round-trips) and transaction ledger."""

import sys
from pathlib import Path
_repo_root = Path(__file__).parent.parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import numpy as np
import pandas as pd
import streamlit as st

from src.trading.paper_trader import PaperTrader

st.set_page_config(page_title="History — AgentQuant", layout="wide")
st.title("History")

pt = PaperTrader()
metrics = pt.get_performance_metrics()

# ── Summary metrics ───────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Closed Trades", str(metrics["num_trades"]))
c2.metric("Total P&L", f"₹{metrics['total_pnl']:+,.2f}")
c3.metric("Win Rate", f"{metrics['win_rate']:.0%}")
c4.metric("Best Trade", f"₹{metrics['best_trade_pnl']:+,.2f}")

st.divider()

tab_trades, tab_txns = st.tabs(["Trade History", "Transactions"])

# ── Tab 1: Trade History (closed round-trips) ─────────────────────────────────
with tab_trades:
    df = pt.get_trade_history()

    if df.empty:
        st.info("No closed trades yet. Run the agent and let it BUY then EXIT a position.")
    else:
        display = df.copy()
        display["pnl"] = display["pnl"].map(lambda v: f"₹{v:+,.0f}")
        display["pnl_pct"] = display["pnl_pct"].map(lambda v: f"{v:+.1%}")
        display["entry_price"] = display["entry_price"].map(lambda v: f"₹{v:.2f}")
        display["exit_price"] = display["exit_price"].map(lambda v: f"₹{v:.2f}")
        display["sharpe_at_entry"] = display["sharpe_at_entry"].map(lambda v: f"{v:.2f}")

        st.dataframe(display, use_container_width=True, hide_index=True)

        st.divider()

        if len(df) >= 3:
            st.subheader("P&L Distribution")
            pnl_values = df["pnl_pct"].values
            bins = np.linspace(pnl_values.min(), pnl_values.max(), 11)
            hist, edges = np.histogram(pnl_values, bins=bins)
            bin_labels = [f"{e:.1%}" for e in edges[:-1]]
            hist_df = pd.DataFrame({"P&L Range": bin_labels, "Count": hist}).set_index("P&L Range")
            st.bar_chart(hist_df)
            st.divider()

        st.download_button("Download Trade History CSV", df.to_csv(index=False), "trades.csv", "text/csv")

# ── Tab 2: Transaction Ledger (every cash event) ──────────────────────────────
with tab_txns:
    txns = pt.get_transaction_history()

    if txns.empty:
        st.info("No transactions yet. Each BUY and SELL appears here as a separate line.")
    else:
        display_txns = txns.copy()
        display_txns["price"] = display_txns["price"].map(lambda v: f"₹{v:.2f}")
        display_txns["amount"] = display_txns["amount"].map(lambda v: f"₹{v:+,.2f}")
        display_txns["commission"] = display_txns["commission"].map(lambda v: f"₹{v:,.2f}")

        # Colour BUY rows green, SELL rows red via background
        def _row_colour(row):
            colour = "background-color: #d4edda" if row["type"] == "BUY" else "background-color: #f8d7da"
            return [colour] * len(row)

        st.dataframe(
            display_txns.style.apply(_row_colour, axis=1),
            use_container_width=True,
            hide_index=True,
        )

        st.divider()

        # Running cash balance
        st.subheader("Cash Flow Over Time")
        cash_flow = txns["amount"].cumsum()
        cf_df = pd.DataFrame({"Cumulative Cash Flow": cash_flow.values}, index=txns["date"])
        st.line_chart(cf_df)

        st.divider()
        st.download_button("Download Transactions CSV", txns.to_csv(index=False), "transactions.csv", "text/csv")
