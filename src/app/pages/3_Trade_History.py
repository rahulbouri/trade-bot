"""Trade History page — all closed paper trades."""

import sys
from pathlib import Path
_repo_root = Path(__file__).parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import numpy as np
import pandas as pd
import streamlit as st

from src.trading.paper_trader import PaperTrader

st.set_page_config(page_title="Trade History — AgentQuant", layout="wide")
st.title("Paper Trade History")

pt = PaperTrader()
df = pt.get_trade_history()
metrics = pt.get_performance_metrics()

# ── Summary metrics ───────────────────────────────────────────────────────────
c1, c2, c3 = st.columns(3)
c1.metric("Total Trades", str(metrics["num_trades"]))
c2.metric("Total P&L", f"₹{metrics['total_pnl']:+,.2f}")
best = metrics["best_trade_pnl"]
c3.metric("Best Trade P&L", f"₹{best:+,.2f}")

st.divider()

# ── Trade table ───────────────────────────────────────────────────────────────
st.subheader("All Trades")

if df.empty:
    st.info("No closed trades yet. Run the agent and let it BUY then EXIT a position.")
else:
    styled_df = df.copy()
    styled_df["pnl"] = styled_df["pnl"].map(lambda v: f"₹{v:+,.0f}")
    styled_df["pnl_pct"] = styled_df["pnl_pct"].map(lambda v: f"{v:+.1%}")
    styled_df["entry_price"] = styled_df["entry_price"].map(lambda v: f"₹{v:.2f}")
    styled_df["exit_price"] = styled_df["exit_price"].map(lambda v: f"₹{v:.2f}")

    st.dataframe(styled_df, use_container_width=True, hide_index=True)

    st.divider()

    # ── P&L Distribution ─────────────────────────────────────────────────────
    st.subheader("P&L Distribution")
    if len(df) < 3:
        st.info("Need more trades for distribution chart.")
    else:
        pnl_values = df["pnl_pct"].values
        bins = np.linspace(pnl_values.min(), pnl_values.max(), 11)
        hist, edges = np.histogram(pnl_values, bins=bins)
        bin_labels = [f"{e:.1%}" for e in edges[:-1]]
        hist_df = pd.DataFrame({"P&L Range": bin_labels, "Count": hist}).set_index("P&L Range")
        st.bar_chart(hist_df)

    st.divider()

    # ── Download button ───────────────────────────────────────────────────────
    st.download_button(
        "Download CSV",
        df.to_csv(index=False),
        "trades.csv",
        "text/csv",
    )
