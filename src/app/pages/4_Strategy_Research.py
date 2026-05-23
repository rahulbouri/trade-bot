"""Strategy Research page — regime detection, backtest explorer, strategy memory browser."""

import sys
from pathlib import Path
_repo_root = Path(__file__).parent.parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import logging
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.agent.strategy_memory import StrategyMemory
from src.backtest.runner import run_backtest
from src.data.ingest import fetch_ohlcv_data
from src.features.engine import compute_features
from src.features.regime import detect_regime, detect_regime_full
from src.strategies.strategy_registry import STRATEGY_REGISTRY
from src.utils.config import config

logger = logging.getLogger(__name__)
st.set_page_config(page_title="Strategy Research — AgentQuant", layout="wide")
st.title("Strategy Research")


@st.cache_data(ttl=3600, show_spinner="Fetching market data…")
def _fetch_cached(ticker: str, start: str, end: str):
    return fetch_ohlcv_data(ticker, start, end)


@st.cache_data(ttl=3600, show_spinner="Computing features…")
def _features_cached(ticker: str, start: str, end: str) -> pd.DataFrame:
    data = _fetch_cached(ticker, start, end)
    return compute_features(data, ticker, config.vix_ticker)


@st.cache_data(ttl=3600, show_spinner="Loading strategy memory…")
def _recall_cached():
    return StrategyMemory().recall(n=50)


# ── Section 1: Regime Detection ───────────────────────────────────────────────
st.subheader("Regime Detection")

today = datetime.now()
end_default = today - timedelta(days=1)
start_default = end_default - timedelta(days=365)

col_a, col_b, col_c = st.columns([2, 1, 1])
with col_a:
    ticker_input = st.text_input("Ticker", value=config.reference_asset)
with col_b:
    start_date = st.date_input("Start Date", value=start_default)
with col_c:
    end_date = st.date_input("End Date", value=end_default)

if st.button("Detect Regime"):
    try:
        with st.spinner("Detecting regime…"):
            features_df = _features_cached(
                ticker_input,
                start_date.strftime("%Y-%m-%d"),
                end_date.strftime("%Y-%m-%d"),
            )
            regime_label = detect_regime(features_df)
            signals = detect_regime_full(features_df)

        st.success(f"Regime: **{regime_label}**  (confidence: {signals.regime_confidence:.0%})")

        sr1, sr2, sr3, sr4 = st.columns(4)
        sr1.metric("VIX Level", f"{signals.vix_level:.1f}")
        sr2.metric("VIX Percentile (1Y)", f"{signals.vix_percentile_252d:.0f}th")
        sr3.metric("3M Momentum", f"{signals.momentum_63d * 100:.1f}%")
        sr4.metric("Realized Vol (21d)", f"{signals.realized_vol_21d * 100:.1f}%")
    except Exception as e:
        st.error(f"Regime detection failed: {e}")

st.divider()

# ── Section 2: Strategy Backtest ──────────────────────────────────────────────
st.subheader("Strategy Backtest")

col_s1, col_s2 = st.columns([2, 2])
with col_s1:
    strategy_sel = st.selectbox("Strategy", options=list(STRATEGY_REGISTRY.keys()))
with col_s2:
    bt_ticker = st.text_input("Ticker for Backtest", value=config.reference_asset, key="bt_ticker")

col_p1, col_p2, col_p3, col_p4 = st.columns(4)
with col_p1:
    fast_window = st.number_input("Fast Window", min_value=2, max_value=50, value=12)
with col_p2:
    slow_window = st.number_input("Slow Window", min_value=10, max_value=200, value=26)
with col_p3:
    bt_start = st.date_input("Backtest Start", value=start_default, key="bt_start")
with col_p4:
    bt_end = st.date_input("Backtest End", value=end_default, key="bt_end")

if st.button("Run Backtest"):
    try:
        with st.spinner("Running backtest…"):
            data = _fetch_cached(
                bt_ticker,
                bt_start.strftime("%Y-%m-%d"),
                bt_end.strftime("%Y-%m-%d"),
            )
            params = {"fast_window": int(fast_window), "slow_window": int(slow_window)}
            result = run_backtest(data, [bt_ticker], strategy_sel, params)

        if result and "metrics" in result:
            m = result["metrics"]
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("Sharpe Ratio", f"{m.get('sharpe_ratio', 0):.3f}")
            mc2.metric("Total Return", f"{m.get('total_return', 0) * 100:.1f}%")
            mc3.metric("Max Drawdown", f"{m.get('max_drawdown', 0) * 100:.1f}%")
            mc4.metric("Num Trades", str(m.get("num_trades", 0)))

            equity = result.get("equity_curve")
            if equity is not None and not equity.empty:
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(equity.index, equity.values, linewidth=1.5, color="#1f77b4")
                ax.fill_between(equity.index, equity.values, equity.min(), alpha=0.08, color="#1f77b4")
                ax.set_title(f"{strategy_sel} — Equity Curve ({bt_ticker})")
                ax.set_ylabel("Portfolio Value")
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                plt.close(fig)
        else:
            st.warning("Backtest returned no results.")
    except Exception as e:
        st.error(f"Backtest failed: {e}")

st.divider()

# ── Section 3: Strategy Memory Browser ───────────────────────────────────────
st.subheader("Strategy Memory Browser")

try:
    past_results = _recall_cached()
    if not past_results:
        st.info("Strategy memory is empty. Run the agent to populate it.")
    else:
        rows = []
        for r in past_results:
            rows.append({
                "Timestamp": r.timestamp if hasattr(r, "timestamp") else "",
                "Regime": r.regime,
                "Strategy": r.strategy_type,
                "Sharpe": round(r.sharpe, 3),
                "Return": f"{r.total_return * 100:.1f}%",
                "Params": r.params,
            })
        mem_df = pd.DataFrame(rows)

        regimes = ["All"] + sorted(mem_df["Regime"].unique().tolist())
        regime_filter = st.selectbox("Filter by Regime", options=regimes)
        if regime_filter != "All":
            mem_df = mem_df[mem_df["Regime"] == regime_filter]

        st.dataframe(mem_df, use_container_width=True, hide_index=True)
except Exception as e:
    st.error(f"Failed to load strategy memory: {e}")
