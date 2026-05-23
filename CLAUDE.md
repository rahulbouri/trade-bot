# AgentQuant 2.0 — Development Context & Plan

**Last Updated:** 2026-05-23 (UPDATED: Phase 3.1 — Dashboard refinements + deployment fixes)  
**Current Focus:** Phase 3.1: Dashboard UX refinements, deployment fixes, universe expansion planning

---

## Executive Context

AgentQuant is transforming into a **production-grade autonomous trading system** from its current state (strategy generation + backtesting). Vision: AI-driven strategy discovery → RL-based position sizing → live trading with monitoring.

**Status:** Core agent + backtest engine working. Building Phase 2: intelligent asset discovery + position optimization.

---

## Design Decision: Asset Search Implementation

### Evaluated Option: Deep Research Agent (AI Alliance)
❌ **Decision:** NOT suitable (but studied for inspiration)

**Why rejected:**
- Built for deep single-company analysis (10K-level financials), not asset screening
- Token cost: 50-100k per asset × 1500 stocks = $7.50+ (vs our $0.50 budget)
- Speed: 5-10 min per stock vs needed 10-30 sec
- Output: Complex JSON (1000+ lines) vs our simple score + reasoning

**What we learned:**
- Planning + replanning pattern useful for agent loops
- Budget enforcement (tokens, cost, time) critical for production
- Structured output with source tracking valuable for validation

### Selected Approach: 4-Step Pipeline (Math → LLM → Backtest)
✅ **Decision:** Intelligent filtering with LLM for risk/reward analysis

**Why NOT centroid?** Centroids lose individual trade variance and stochastic details. Different tickers in similar regimes need nuanced analysis.

**Architecture: 4 Steps**

**Step 1 - Asset Search (Math, 0 tokens, <3 sec)**
- Query StrategyMemory for ALL winning trades (Sharpe > 1.0)
- For each asset: find similar historical trades using **signal distance**
- Score: `similarity × (1 + historical_return) × win_rate`
- Output: Top 50 assets ranked by best_strategy score
- Preserves **individual trade details** for LLM analysis

**Step 2 - Strategy Shortlist (Math, 0 tokens, <100ms, cached)**
- Group StrategyMemory by (strategy_type, regime_label)
- For current regime: rank strategies by mean Sharpe > 1.0
- Output: Top 3 strategies for this regime

**Step 3 - LLM Risk/Reward Analysis (LLM, ~$0.25, ~30 sec)**
- For each top 50 asset:
  - MCP: Fetch financial news (sentiment, risk events)
  - Show: 3 most similar historical winning trades + their details
  - Ask LLM: "Given historical trades + news, should we take this position? YES/NO/MAYBE"
- Output: Decision + confidence for each asset
- **Important:** LLM analyzes actual data (trades + news), not abstract parameters. This is risk/reward analysis, not signal generation.

**Step 4 - Backtest & Store (Math, 0 tokens, <60 sec)**
- Backtest only LLM-accepted positions
- Store results + **signals_vector** to StrategyMemory
- Feed best 5 to execution layer

**Why This Approach:**
| Aspect | 4-Step Pipeline | Pure Centroid |
|--------|---|---|
| Token cost | $0.25 (filtered) | $0 |
| Preserves variance | ✅ YES (shows actual trades) | ❌ NO (averaged) |
| LLM role | ✅ Enhanced (real data) | ❌ Removed (poor fit) |
| Bootstrap friendly | ✅ YES (fallback to grid) | ✅ YES |
| Speed | ✅ <4 sec total | ✅ <3 sec |
| Financial insight | ✅ YES (news + sentiment) | ❌ NO (pure math) |

**Expected ROI:** +30-40% improvement via informed asset selection + LLM risk filtering

---

## System Architecture: 4 Main Components

### 1️⃣ SEARCH & PLANNING MODULE (Current Focus) — `src/agent/`
**Owner:** None (being built)  
**Dependencies:** Data ingestion, feature engine, strategy registry  
**Linked to:** Agent loop, parameter grid  

**Purpose:** Intelligently identify assets & strategies suitable for current market regime instead of hardcoded universe.

```
Input: Full stock universe (1500+ NSE stocks)
       ├─ fetch all OHLCV data
       ├─ score by: liquidity, volatility, trend, correlation, volatility
       └─ return top 50 suitable for current regime

Output: Ranked asset list → feeds into agent's backtest tournament
        + Strategy affinity map (e.g., Bull market → 0.9 momentum, 0.3 reversion)
```

**New Files (TODO):**
- `src/agent/search_pipeline.py` — SearchPipeline class (4-step: search → shortlist → LLM → backtest)
- `src/agent/strategy_memory.py` (modify) — Add `signals_vector: str` field to store RegimeSignals
- `tests/test_search_pipeline.py` — Unit tests for each step
- `tests/test_integration_search.py` — E2E tests (bootstrap → backtest)

**Integration Points:**
```
Step 1: analyze_node()
├─ detect_regime_full() → RegimeSignals
├─ SearchPipeline.search_assets() → top 50 assets + strategies
└─ state["asset_search_results"]

Step 2: shortlist_node()
├─ SearchPipeline.shortlist_strategies() → top 3 by Sharpe
└─ state["strategies_ranked"]

Step 3: llm_analysis_node()
├─ SearchPipeline.analyze_with_llm() + MCPToolset
├─ MCP: news, sentiment, risk
└─ state["llm_decisions"]

Step 4: backtest_node()
├─ SearchPipeline.backtest_accepted() → run backtests
├─ Store to StrategyMemory (with signals_vector)
└─ state["best_result"]
```

---

### 1b️⃣ ALPHA DISCOVERY MODULE (Offline Research) — `src/agent/`
**Owner:** None (being built)  
**Dependencies:** StrategyMemory, backtest runner, LLM  
**Linked to:** Search & Planning (feeds validated edges)  

**Purpose:** Use LLM to discover real edges from historical winning trades, validate them, and encode as rules.

```
Phase 1 (Discovery): Analyze winning trades
├─ LLM reads 100+ winning trades per strategy
├─ Identifies patterns: "momentum works only in Bull + mid-caps"
└─ Output: Testable hypothesis

Phase 2 (Validation): Backtest the hypothesis
├─ Compare: original strategy vs hypothesis-restricted strategy
├─ Require: >100 bps Sharpe improvement
└─ Output: Real edge or spurious correlation?

Phase 3 (Generalization): Test across contexts
├─ Does edge work in Bull/Bear/Choppy?
├─ Does edge work on large/mid/small caps?
├─ Does edge work on different timeframes?
└─ Output: Precise edge definition with constraints
```

**New Files (TODO):**
- `src/agent/alpha_discovery.py` — AlphaDiscoveryAgent (Phase 1: analyze patterns)
- `src/agent/edge_validator.py` — EdgeValidator (Phase 2: backtest hypothesis)
- `src/agent/edge_generalizer.py` — EdgeGeneralizer (Phase 3: test constraints)
- `tests/test_alpha_discovery.py` — Unit tests for discovery pipeline
- `tests/test_edge_validation.py` — Integration tests for validation

**Integration Points:**
```
discovery_node() (weekly/monthly, offline)
├─ AlphaDiscoveryAgent.analyze_winning_patterns(strategy)
├─ EdgeValidator.validate_edge_hypothesis()
├─ EdgeGeneralizer.test_across_regimes/timeframes/assets()
└─ state["validated_edges"] ← List of real, tested edges

search_node() (daily, real-time)
├─ SearchPipeline uses only validated edges
├─ Parameters from discovery (not grid search)
└─ Regime filters applied from generalization
```

**Why This Matters:**
- Discovers "why" winners win (not just "that" they win)
- Validates edges before deployment (prevents overfitting)
- Shifts from "find similar assets" to "find real alpha sources"
- Expected impact: +5-10% Sharpe from tested edges

---

### 2️⃣ RL MODEL (Position Sizing & Execution) — `src/rl/`
**Owner:** None (being built)  
**Dependencies:** Agent signals, backtest metrics, training data  
**Linked to:** Dashboard, live execution, monitoring  

**Purpose:** Learn optimal position sizing given agent signal + market state. Replace fixed 1/N allocation with learned dynamic allocation.

```
Input State: {agent_signal, confidence, regime, vix, portfolio_sharpe, 
              drawdown, position_current, cash, rsi, momentum, volatility}
                        ↓
         PPO Policy Network (state → action)
                        ↓
Output Action: {position_size: 0.0-1.0, leverage: 1.0-2.0, hedge_ratio: 0.0-1.0}
                        ↓
Reward: (daily_return - risk_penalty - transaction_cost + sharpe_bonus)
```

**New Files (TODO):**
- `src/rl/training/environment.py` — Gym-like trading env
- `src/rl/training/dataset.py` — Historical data + agent signals loader
- `src/rl/models/policy_network.py` — PPO policy (PyTorch)
- `src/rl/models/value_network.py` — Value function
- `src/rl/training/training_loop.py` — PPO training loop
- `src/rl/backtest_rl.py` — Validate RL on historical data
- `src/rl/live_execution.py` — Deploy RL agent to broker API
- `tests/test_rl_*.py` — Unit tests

**Integration Points:**
```
agent_graph.py (backtest_node)  →  signal: {-1, 0, 1}
                                    confidence: float
                                    ↓
                            RLExecutor.decide(state)
                                    ↓
                            position_size, leverage
                                    ↓
                            live_execution.py  (broker API)
                                    ↓
monitoring_dashboard.py  ←  live P&L, positions, alerts
```

---

### 3️⃣ MCP INTEGRATION (Real-time Data & Signals) — `src/agent/`, `src/features/`
**Owner:** None (optional add-on)  
**Dependencies:** MCP client libs, existing tools.py  
**Linked to:** Context builder, feature engine, strategy library  

**Purpose:** Replace yfinance batch pulls with real-time MCPs. Add news sentiment, economic calendar, alternative data for richer regime detection.

```
Current: yfinance (EOD batch)  →  features  →  context
New:     MCP market data (RT)  ┐
         MCP sentiment (News)  ├→ features + MCP features → enhanced context
         MCP calendar (Events) ┘
```

**New Files (TODO):**
- `src/agent/mcp_tools.py` — MCPToolset wrapper class
- `src/mcp_config.yaml` — MCP endpoint configs
- `src/features/mcp_features.py` — sentiment, event indicator columns

**Relevant MCPs (from `~/.claude/skills/`):**
- `langchain` — LLM abstraction (can wrap MCP calls)
- `llamaindex` — Data framework (RAG for sentiment/events)
- `modal` — Distributed MCP execution

**Integration Points:**
```
build_context() (current)  →  compute_features()  →  RegimeContext
                |
                └─ (NEW) MCPToolset.get_sentiment() ──┐
                └─ (NEW) MCPToolset.get_risk_events() ─┼→ context enrichment
                └─ (NEW) MCPToolset.get_realtime_ohlcv() ┘
```

---

### 4️⃣ MONITORING DASHBOARD (Live Trading Validation) — `src/app/`
**Owner:** None (being built)  
**Dependencies:** Live execution, backtest metrics, RL agent  
**Linked to:** Everything (observability)  

**Purpose:** Real-time P&L tracking, backtest vs live gap detection, alert system. Prevents blowups by validating RL agent performance.

```
Live Broker API  →  RLExecutor  →  positions, P&L, fills
                                    ↓
                          Metrics calculator
                                    ↓
monitoring_dashboard.py (Streamlit)  ←  live metrics
                                    ├─ Portfolio value, Sharpe, drawdown
                                    ├─ RL decisions (allocation, leverage)
                                    ├─ Backtest vs reality gap
                                    └─ Alerts (gap > 20%, drawdown > limit)
```

**New Files (TODO):**
- `src/app/rl_monitoring_dashboard.py` — Streamlit dashboard (Tabs: Live Performance, RL Decisions, Gap Analysis, Alerts)
- `src/rl/validation.py` — BacktestRealityValidator class (check gaps, raise alerts)
- `tests/test_dashboard.py` — Mock streaming metrics

**Integration Points:**
```
live_execution.py  →  monitoring_dashboard.py  ←  backtest_metrics
                                          ↓
                              validation.py  (gate: allow/stop trading)
```

---

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│ PHASE 1: ASSET SEARCH (Math-driven, 0 tokens, <3 sec)      │
├─────────────────────────────────────────────────────────────┤
│ fetch_ohlcv_data() ──→ compute_features()                   │
│     ↓                                                       │
│ detect_regime_full() ──→ RegimeSignals                      │
│     ↓                                                       │
│ SearchPipeline.search_assets(universe=1500)                 │
│ ├─ Query StrategyMemory (Sharpe > 1.0)                      │
│ ├─ For each asset: find similar trades                      │
│ └─ Return: Top 50 assets + best_strategy                    │
│     ↓                                                       │
│ state["asset_search_results"] ← 50 candidates               │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ PHASE 2: STRATEGY SHORTLIST (Math, cached, <100ms)         │
├─────────────────────────────────────────────────────────────┤
│ SearchPipeline.shortlist_strategies(regime)                 │
│ ├─ Group StrategyMemory by (strategy, regime)               │
│ ├─ Rank by mean Sharpe                                      │
│ └─ Return: [(momentum, 1.5), (trend, 1.2), ...]             │
│     ↓                                                       │
│ state["strategies_ranked"] ← top 3 strategies                │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ PHASE 3: LLM ANALYSIS (AI-driven, ~$0.25, ~30 sec)         │
├─────────────────────────────────────────────────────────────┤
│ For each top 50 asset:                                      │
│ ├─ MCPToolset.get_sentiment() ──→ news, risk events         │
│ ├─ SearchPipeline.find_similar_trades() ──→ 3 examples     │
│ ├─ LLM prompt: regime + historical trades + news            │
│ └─ LLM decision: "YES" / "NO" / "MAYBE"                     │
│     ↓                                                       │
│ state["llm_decisions"] ← {ticker: decision, reason}         │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ PHASE 4: BACKTEST & STORE (Math, 0 tokens, <60 sec)        │
├─────────────────────────────────────────────────────────────┤
│ SearchPipeline.backtest_accepted()                          │
│ ├─ Run backtest only on YES decisions (~20-30 assets)       │
│ ├─ Store to StrategyMemory (with signals_vector)            │
│ └─ Rank by Sharpe                                           │
│     ↓                                                       │
│ state["best_result"] ← top backtested strategy              │
│     ↓                                                       │
│ RLExecutor.decide(state) ──→ position_size, leverage        │
│     ↓                                                       │
│ live_execution.py ──→ broker API ──→ place trades           │
│     ↓                                                       │
│ monitoring_dashboard.py ←─ live P&L, validation             │
│                             ↓                              │
│                   Alert: "Stop trading" or continue        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Implementation Roadmap

### Phase 1: 4-Step Search Pipeline + Alpha Discovery (Weeks 1-4) ← **COMPLETED** ✅

**Status:** All files created, tested, integrated into agent_graph.py. Phase 1 fully functional.

**Files Completed:**
- src/agent/search_pipeline.py (12 KB)
- src/agent/alpha_discovery.py (5.8 KB)
- src/agent/edge_validator.py (7.5 KB)
- src/agent/edge_generalizer.py (11 KB)
- src/features/regime.py (modified, signals_to_vector added)
- src/agent/strategy_memory.py (modified, signals_vector field added)
- src/agent/agent_graph.py (modified, discovery/analyze/shortlist/llm_analysis/backtest/store nodes added)
- tests/test_search_pipeline.py (14 tests, 100% coverage)
- tests/test_alpha_discovery.py (15 tests, 100% coverage)
- tests/test_integration_search.py (9 tests, 100% coverage)

**Key Metrics:**
- Token cost: ~$0.25 per day (vs $2.50 without aggregation)
- Asset search time: <3 seconds on 1500 assets
- Top 50 assets ranked by historical similarity + LLM approval
- All Phase 1 tests passing, 100% coverage maintained

---

### Phase 2: Position Management & Historical Aggregation (Weeks 5-6) ← **COMPLETED** ✅

**Status:** Complete. All 118 tests passing. Live end-to-end verified with real Gemini LLM calls.

**Files Completed:**
- src/agent/historical_aggregator.py — HistoricalAggregator, StrategyRegimeSummary dataclass
- src/agent/position_manager.py — PositionManager with 5-rule heuristic decision engine (SQLite-backed)
- src/agent/base_planner.py — Added generate_text() with token counting to all planners (Gemini, LangChain, OpenAI, Fallback)
- src/agent/agent_graph.py — llm_analysis_node (real Gemini batch call), position_management_node (LLM-filtered), backtest_node (position-decided ticker), discovery_node (100-record self-gate)
- tests/test_historical_aggregator.py — 14 tests
- tests/test_position_manager.py — 16 tests
- tests/test_position_management_node.py — 5 tests
- tests/test_phase2_e2e.py — 3 E2E lifecycle tests

**Key Metrics:**
- Token logging: prompt/completion/thinking/total for every LLM call
- LLM batch analysis: single Gemini call for top-10 assets
- Position rules: BUY(>50%), HOLD(same ticker >70%), SWITCH(>80%), EXIT(>5 days or explicit)
- Bootstrap safe: empty StrategyMemory → LLM says NO → SKIP → backtest runs anyway → grows memory

---

### Phase 3: Paper Trading Engine + Streamlit Dashboard (Weeks 7-8) ← **COMPLETED** ✅

**Status:** Fully implemented and deployed to Streamlit Community Cloud.

**Files Completed:**
- src/trading/__init__.py, src/trading/paper_trader.py — SQLite-backed virtual portfolio (₹1L capital)
- src/app/streamlit_app.py — multi-page entry point with sidebar portfolio summary
- src/app/pages/0_Status.py — keepalive page (5-min HTML meta refresh)
- src/app/pages/1_Portfolio.py — live mark-to-market via yf.Ticker.fast_info (5-min cache)
- src/app/pages/2_Agent_Runs.py — run history + "Run Agent Now" button
- src/app/pages/3_Trade_History.py — tabbed: Trade History + Transaction Ledger
- src/app/pages/4_Strategy_Research.py — backtest explorer + regime analysis
- scripts/run_scheduler.py — APScheduler daemon (Mon–Fri 04:00 UTC = 09:30 IST)
- scripts/simulate_history.py — 10-day historical simulation script
- tests/test_paper_trader.py — 12 tests

**Key Decisions:**
- PaperTrader derives transaction ledger from existing tables (no extra schema)
- Live portfolio prices use yf.Ticker.fast_info (bypasses parquet cache, 5-min TTL)
- Scheduler runs as background thread; Streamlit is PID 1 via `exec`
- Deployed via Streamlit Community Cloud (private GitHub repo, TOML secrets)

---

### Phase 3.1: Dashboard Refinements (2026-05-23) ← **CURRENT**

**Status:** In progress.

**Changes Made:**
- **Dependency fix:** Migrated from `google-generativeai` → `google-genai` (new unified SDK) to resolve conflict with `langchain-google-genai>=2.0`. Updated `GeminiPlanner` to use `genai.Client` + `client.models.generate_content`.
- **Module resolution fix:** Added `src/__init__.py`, `src/app/__init__.py`, and `sys.path` fix at top of all page files — required for Streamlit Cloud to resolve `from src.X import Y`.
- **Live price fix:** Portfolio page now calls `yf.Ticker.fast_info.last_price` (5-min Streamlit cache) instead of reading stale parquet file. Unrealized P&L is now accurate.
- **History page:** Renamed Trade History → History with two tabs:
  - **Trade History** — closed round-trips with P&L, regime, exit reason, distribution chart
  - **Transactions** — chronological BUY/SELL ledger derived from `trades` + `open_positions` tables, with cash flow chart. Green/red row colouring.

**Pending for Phase 3.1:**
- Universe expansion: upgrade `fetch_ohlcv_data` to use `yf.download(list)` batch API before expanding beyond 5 tickers. Target: 20-30 liquid ETFs (sector, bond, commodity). NSE stocks require `.NS` suffix handling and are a later milestone.
- Ingest refactor: batch download reduces cold-start from O(n×2s) sequential → O(1×8s) batch for 50 tickers.

---

**Build Instructions:**
- See: /tmp/PHASE3_BUILD_PROMPT.md — Complete Phase 3 specification (all API signatures, DB schema, tests)

**What Phase 3 Adds:**
- PaperTrader (src/trading/paper_trader.py) — SQLite-backed virtual portfolio; BUY/SELL/P&L tracking; ₹1L starting capital
- store_node integration — After each agent run, executes paper BUY/SELL based on position_decision.action
- Multi-page Streamlit dashboard (replaces existing src/app/streamlit_app.py):
  - Page 1: Portfolio — current positions, P&L, Sharpe, drawdown
  - Page 2: Agent Runs — run history, LLM decisions, "Run Agent Now" button
  - Page 3: Trade History — all closed trades with P&L breakdown
  - Page 4: Strategy Research — backtest explorer (ported from old app)

**Files to Create:**
1. src/trading/__init__.py (empty)
2. src/trading/paper_trader.py (~280 lines, PaperTrader class)
3. src/app/pages/1_Portfolio.py
4. src/app/pages/2_Agent_Runs.py
5. src/app/pages/3_Trade_History.py
6. src/app/pages/4_Strategy_Research.py
7. tests/test_paper_trader.py (12 tests)

**Files to Rewrite:**
1. src/app/streamlit_app.py (multi-page entry point)

**Files to Modify:**
1. src/agent/agent_graph.py (add paper trading to store_node)

**Success Criteria:**
- 12 new tests pass (130 total including Phase 1+2)
- store_node logs PAPER TRADE: BUY/SELL on every agent run
- Dashboard loads with 4 pages, handles empty state gracefully
- "Run Agent Now" button in Agent Runs page triggers full pipeline
- round-trip BUY→SELL correctly records P&L in trade history

**Time Estimate:** 5-6 hours

---

### Phase 3: 4-Step Search Pipeline + Alpha Discovery (Weeks 1-4) ← **FUTURE**

**Week 1: Extend StrategyMemory** (2-3 days)
- [ ] Add `signals_vector: str` field (stores RegimeSignals as JSON)
- [ ] Modify `store()` to accept and persist signals
- [ ] Add helper: `_deserialize_signals()` to reconstruct RegimeSignals
- [ ] Migration: backfill existing trades with placeholder signals_vector

**Week 1-2: Implement SearchPipeline** (5-7 days)
- [ ] `find_similar_winning_trades()` — Query StrategyMemory, compute signal distance, filter by Sharpe > 1.0
  - Use `signals_to_vector()` for Euclidean distance
  - Return: Top 10 individual trades (NOT centroid)
- [ ] `score_asset_for_strategy()` — similarity × (1 + historical_return) × win_rate
  - Score each asset for each strategy
  - Return: Per-strategy breakdown with reasoning
- [ ] `search_assets()` — Score 1500 assets in parallel, return top 50
  - ThreadPoolExecutor(max_workers=8)
  - O(1500 × 3 strategies × distance calc) = <3 seconds
- [ ] `shortlist_strategies()` — Rank by mean Sharpe per regime
  - Group StrategyMemory by (strategy, regime)
  - Return: Top 3 strategies with fitness scores
- [ ] `analyze_with_llm()` — Placeholder for risk/reward analysis
  - Currently: auto-approve all with score as confidence
  - Later: integrate with MCP for news/sentiment
- [ ] `backtest_accepted()` — Run backtest, store with signals_vector

**Week 2: Implement Alpha Discovery** (5-7 days)
- [ ] **AlphaDiscoveryAgent** (Phase 1: Discover patterns)
  - `analyze_winning_patterns()` — Feed LLM all winning trades, ask: "What patterns do you see?"
  - Parse LLM output to extract edge hypothesis
  - Store hypothesis to disk for reference
- [ ] **EdgeValidator** (Phase 2: Validate hypothesis)
  - `validate_edge_hypothesis()` — Backtest original vs restricted strategy
  - Require >100 bps Sharpe improvement to accept
  - Return: verdict (REAL edge vs SPURIOUS correlation)
- [ ] **EdgeGeneralizer** (Phase 3: Test constraints)
  - `test_across_regimes()` — Does edge work in Bull/Bear/Choppy?
  - `test_across_timeframes()` — Does edge work on different windows?
  - `test_across_asset_classes()` — Does edge work on large/mid/small caps?
  - Return: precise edge definition with constraints
- [ ] **discovery_node()** — Integrate into agent_graph
  - Runs weekly/monthly (offline research, not daily)
  - Accumulates validated edges in state

**Week 3: Update agent_graph.py** (2-3 days)
- [ ] Add `discovery_node()` (weekly discovery loop)
- [ ] Modify `analyze_node()` → add `asset_search_results` to state
- [ ] Add `shortlist_node()` → add `strategies_ranked` to state
- [ ] Add `llm_analysis_node()` → add `llm_decisions` to state
- [ ] Modify `backtest_node()` → backtest only accepted, store signals_vector with edge metadata
- [ ] Update `run_agent()` loop: 
  - Daily: analyze → shortlist → llm → backtest
  - Weekly: discovery (offline, runs in background)

**Week 3-4: Tests & Validation** (3-4 days)
- [ ] `test_search_pipeline.py` — Unit tests for each SearchPipeline method
  - Mock StrategyMemory with sample winning trades
  - Test signal distance calculation
  - Test score computation
  - Test asset scoring logic
- [ ] `test_alpha_discovery.py` — Unit tests for discovery pipeline
  - Mock LLM responses with realistic hypotheses
  - Test hypothesis parsing
  - Test edge validation logic
- [ ] `test_integration_search.py` — E2E from discover to backtest
  - Bootstrap: empty StrategyMemory → first discovery run
  - Verify: hypotheses generated and validated
  - Verify: validated edges used in search_assets
- [ ] **Validation: Run against historical 2022-2024 data**
  - Metric 1: Top 50 avg Sharpe vs baseline (expect +20-30%)
  - Metric 2: Similar trades correctly matched (precision >80%)
  - Metric 3: Discovered edges improve Sharpe by >100 bps
  - Metric 4: LLM decisions improve win rate by >5%

### Phase 2: RL Training Pipeline (Weeks 5-10)
- [ ] Environment: Gym-like trading env (state, action, reward)
- [ ] Dataset: Load OHLCV + agent signals + features
- [ ] Model: PPO policy + value network (PyTorch)
- [ ] Training: training_loop.py, PPO algorithm
- [ ] Validation: backtest_rl.py (compare vs baseline agent)
- [ ] Paper trade simulation for 4 weeks

### Phase 3: Live Execution & Monitoring (Weeks 11-14)
- [ ] Live execution: RLExecutor → broker API (Zerodha/Shoonya)
- [ ] Monitoring dashboard: Streamlit (live P&L, alerts)
- [ ] Validation: BacktestRealityValidator (gap < 20%)
- [ ] Live trading: ₹50k account for 2 weeks validation

### Phase 4: MCPs & Enhancement (Weeks 15-18)
- [ ] MCP integration: sentiment, calendar, realtime data
- [ ] Feature enhancement: sentiment score, event flags
- [ ] Retrain RL with richer features
- [ ] Scale live account if metrics validate

### Phase 5: Production Hardening (Ongoing)
- [ ] Risk limits: circuit breakers, position limits
- [ ] Strategy ensemble: combine multiple strategies
- [ ] Continuous learning: monthly retrain on new data
- [ ] Broker failover, execution monitoring

---

## Skill & Tool Reference

### When to Use Skills
Run `ls ~/.claude/skills/` to see available. Relevant for AgentQuant:

| Skill | Purpose | When to Use |
|-------|---------|------------|
| `/dspy` | Declarative programming, prompt optimization | Optimizing agent prompts, building modular RAG |
| `/crewai` | Multi-agent orchestration | If we need parallel agents (one per asset class) |
| `/llamaindex` | Data framework, RAG | Ingesting news for sentiment, building feature library |
| `/langchain` | LLM abstraction | Wrapping Gemini calls, chaining tools |
| `/ray-train` | Distributed RL training | Scaling PPO training across GPUs |
| `/deepspeed` | Distributed training optimization | Optional: accelerate RL training |
| `/modal` | Serverless execution | Deploy live_execution.py on cloud |

**Suggested for Search Module MVP:**
- None required yet (pure Python). Use langchain if adding Gemini calls.

**Suggested for RL Phase:**
- `/ray-train` — parallelize training across GPUs
- `/modal` — deploy live_execution.py

**How to Suggest Skills:** I'll mention relevant skills when planning tasks. You decide to use or skip.

---

## Code Style & Standards

**Existing Patterns (Keep Consistent):**
- All strategies return signal: {-1, 0, 1} (from `base.py`)
- Metrics via `PerformanceMetrics` class (single source of truth)
- Config via Pydantic (no ad-hoc dicts)
- Logging via `logger.info()`, no print()
- Tests in `tests/`, 100% coverage for new modules
- Docstrings: 1-liner only (no verbose blocks)

**New Module Template:**
```python
"""Module name — One line purpose."""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

class MyClass:
    """One-liner docstring."""
    
    def my_method(self) -> Dict[str, Any]:
        """Return what it does."""
        ...
```

---

## Running & Testing

**Run agent (current state):**
```bash
python -m src.agent.runner
```

**Run dashboard:**
```bash
python run_app.py
```

**Run all tests:**
```bash
pytest tests/ -v
```

**Run specific test:**
```bash
pytest tests/test_asset_search.py -v
```

**Check code quality:**
```bash
ruff check src/ tests/
```

---

## Key Decisions Made

1. **Alpha Discovery as Phase 1 foundation** — Don't start with position selection; start with edge discovery. LLM analyzes historical wins to identify real patterns, then validate them before using.
2. **LLM for discovery (offline), not decisions (online)** — Weekly: LLM analyzes 100+ winning trades, generates testable hypotheses. Daily: use validated edges as rules (not LLM opinions).
3. **3-Phase Discovery Process** — Discover patterns → Validate (>100 bps improvement required) → Generalize (test across regimes/timeframes/assets) → Encode as rules.
4. **4-Step Pipeline (Math → Risk/Reward → Backtest)** — Filter from 1500 → 50 assets (math, cheap), LLM analyzes risk/reward with historical trades + news (informed), backtest validates (safe).
5. **Keep individual trade details, not centroids** — Preserves variance, stochastic patterns, edge cases for LLM reasoning and pattern discovery.
6. **StrategyMemory stores signals_vector** — All historical results stored with full RegimeSignals for reproducibility and discovery analysis.
7. **Bootstrap with grid search fallback** — Empty StrategyMemory on day 1 is OK; tier 2 fallback handles it; discovery node grows edge library over weeks.
8. **Validate edges before deployment** — Only use edges that improve Sharpe by >100 bps in backtest, generalize across contexts, and have specific constraints.
9. **PPO for RL position sizing** — Once edges are validated, RL learns optimal allocation (not signal generation).
10. **Streamlit for monitoring** — Fast iteration, real-time plots, low ops overhead.
11. **Backtest-reality gap < 20%** — Gate for live trading (prevents blowups).
12. **MCPs in Phase 4** — Nice-to-have for news/sentiment; Phases 1-3 work without them.

---

## Questions for User (Decisions Pending)

1. Which broker? (Zerodha, Shoonya, Interactive Brokers?) → Affects API, commission
2. Real money budget? (₹50k, ₹5L, ₹50L?) → Affects strategy viability, position sizing
3. Acceptable drawdown limit? (5%, 10%, 15%?) → Hard stop in monitoring dashboard
4. Prefer 24/7 live execution or market hours only? → Infrastructure planning
5. Use leverage? (1x, 2x, 3x?) → Risk profile, margin account needed

---

## SearchPipeline Architecture (Phase 1 Deep Dive)

### Why Individual Trades, Not Centroids?

**Problem with centroids:**
- Average 1000 winning trades → single "profile"
- Loses variance: high-volatility winners vs stable winners mixed together
- Loses edge cases: extreme momentum winners vs gentle trends
- LLM can't reason about "why this works"

**Solution: Keep individual trades**
```
StrategyMemory (source of truth)
├─ Trade 1: momentum, Sharpe=1.8, Return=+15%, signals={...}
├─ Trade 2: momentum, Sharpe=1.5, Return=+12%, signals={...}
├─ Trade 3: momentum, Sharpe=1.2, Return=+8%, signals={...}
└─ ...

SearchPipeline.find_similar_winning_trades():
├─ For current asset signals
├─ Compute distance to Trade 1, Trade 2, Trade 3, ...
├─ Return top 10 most similar (NOT averaged)
└─ Pass all 10 to LLM with their details

LLM can now reason:
"Trades 1-3 used period=20-22 and worked well. Trade 4 used period=30 and failed.
Current asset matches period=20. Recommend YES with caution about sharp pullbacks."
```

### Signal Vector Definition

```python
# RegimeSignals → numeric vector (src/features/regime.py)
signals_to_vector(RegimeSignals) → np.ndarray([
    vix_percentile / 100.0,          # 0.35 → normalized 0-1
    momentum_63d,                    # 0.08 → already -1 to +1
    realized_vol_21d,                # 0.18 → ~0.1 to 0.5
    drawdown_from_52w_high,          # -0.05 → ~-1 to 0
    price_vs_200sma_pct,             # 0.03 → ~-0.2 to +0.2
    vol_regime_ordinal               # 0.33 (low=0, mid=0.33, high=0.67, crisis=1)
])
```

### Time Complexity

```
Phase 1 (Search Assets):
├─ Query StrategyMemory: 1 call → all records (~5k-10k)
├─ For each of 1500 assets:
│  └─ For each of 3 strategies:
│     └─ Compute distance to ~100 historical trades
├─ Total: 1500 × 3 × 100 = O(450k) distance calcs
├─ Time: ThreadPoolExecutor(8) → ~3 seconds
└─ Cost: $0

Phase 2 (Shortlist):
├─ Group StrategyMemory by (strategy, regime)
├─ Compute mean Sharpe per group
├─ Time: O(num_trades) ~ 5-10ms
├─ Cost: $0 (cached weekly)
└─ Reuse: Yes (same for all assets)

Phase 3 (LLM Analysis):
├─ For each of 50 accepted assets:
│  ├─ MCP: fetch news (1-2 sec)
│  ├─ SearchPipeline: find 3 similar trades (50ms)
│  └─ LLM: analyze + decide (5-10 sec)
├─ Total: 50 × 15 sec / parallel = ~30 sec
├─ Cost: ~500 tokens per asset × 50 = 25k tokens = $0.25
└─ Result: Informed decisions backed by data

Phase 4 (Backtest):
├─ Only backtest LLM-accepted (maybe 20-30 assets)
├─ Time: ~1-2 sec per asset × 20-30 = ~60 sec
├─ Cost: $0
└─ Speedup: 90% faster than backtesting all 1500
```

### Bootstrap Strategy

**Day 1 (Empty StrategyMemory):**
```
Step 1: search_assets() → finds NO trades (memory empty)
        → Returns empty or low scores
Step 2: shortlist_strategies() → finds NO trades (memory empty)
        → Returns fallback: [("momentum", 0.5), ...]
Step 3: llm_analysis_node() → auto-approves all (no history yet)
Step 4: backtest_accepted() → runs backtest, stores to memory
```

**Day 2-30 (Growing StrategyMemory):**
```
search_assets() → increasingly finds matches as memory grows
shortlist_strategies() → increasingly accurate Sharpe rankings
Result: Learning curve built in, no cold-start problem
```

---

## Alpha Discovery Module: Finding Real Edges (Phase 1 Deep Dive)

### Why Discovery First?

Most traders/quants skip this. They jump to:
- "Build a screener" (no edge, just variance)
- "Use LLM to decide positions" (LLM hallucinates, lagging indicator)
- "Backtest everything" (overfitting, not validation)

**Better approach:** Discover edges first, then build everything else around them.

### 3-Phase Discovery Process

**Phase 1: Discover Patterns (LLM Analysis)**

```
Input: 100+ winning trades from StrategyMemory
  ├─ Trade 1: momentum, Sharpe=1.8, Return=+15%, regime=LowVol-Bull, signals={...}
  ├─ Trade 2: momentum, Sharpe=1.5, Return=+12%, regime=LowVol-Bull, signals={...}
  ├─ Trade 3: momentum, Sharpe=1.2, Return=+8%, regime=MidVol-Bull, signals={...}
  └─ ...

LLM Prompt: "Analyze these winning trades. What patterns do you see? 
  - Which regimes had most winners?
  - Which parameters appear in >70% of winners?
  - What's ONE specific condition that appears in winners but NOT in losers?"

LLM Output: "Edge: Momentum strategy with fast_window=12-15 AND slow_window=45-60
  works ONLY in Bull markets, avg Sharpe 1.5 vs 0.8 in other regimes"

Output: Testable hypothesis
```

**Phase 2: Validate Hypothesis (Backtest Comparison)**

```
Hypothesis: "Momentum only in Bull regimes improves Sharpe"

Backtest 1 (Original):
  ├─ Momentum strategy, no restrictions
  ├─ Period: 2022-2024
  └─ Sharpe: 1.1

Backtest 2 (Restricted):
  ├─ Momentum strategy, Bull regime filter only
  ├─ Period: 2022-2024
  └─ Sharpe: 1.35

Improvement: +250 bps Sharpe

Verdict: ✅ REAL EDGE (>100 bps improvement)
         If <100 bps: ❌ SPURIOUS (noise in historical data)
```

**Phase 3: Generalize (Test Across Contexts)**

```
Question 1: Does edge work in ALL regimes or just Bull?
  ├─ Bull: Sharpe 1.35 ✅
  ├─ Bear: Sharpe 0.65 ⚠️
  └─ Choppy: Sharpe 0.2 ❌
  Conclusion: Only in Bull markets

Question 2: Does edge work on different timeframes?
  ├─ 10d: Sharpe 1.2
  ├─ 15d: Sharpe 1.35 ✅ (peak)
  └─ 20d: Sharpe 1.1
  Conclusion: Optimal at 15d, drops quickly

Question 3: Does edge work on all asset classes?
  ├─ Large-cap: Sharpe 0.9
  ├─ Mid-cap: Sharpe 1.35 ✅ (best)
  └─ Small-cap: Sharpe 0.4
  Conclusion: Only works on mid-cap

FINAL EDGE DEFINITION:
"Momentum strategy with fast=12, slow=55 ONLY works:
  - In Bull market regimes
  - On 15-day timeframe
  - On mid-cap Indian stocks (₹1B-₹5B)
"
```

### How This Feeds the Pipeline

```
Weekly Discovery Loop (Offline):
discover_node()
├─ AlphaDiscoveryAgent.analyze_winning_patterns()
├─ EdgeValidator.validate_edge_hypothesis()
├─ EdgeGeneralizer.test_across_contexts()
└─ state["validated_edges"] ← [{strategy, hypothesis, validation, constraints}]
                              ↓
Daily Search Loop (Online):
search_node()
├─ SearchPipeline.search_assets() uses ONLY validated edges
├─ Parameters come from discovery (not grid search)
├─ Regime filters applied from generalization testing
└─ state["asset_search_results"] ← Top 50 with validated edges
                              ↓
backtest_node()
├─ Backtest confirms edge still works today
├─ Store with edge_metadata (which hypothesis it validated)
└─ Feed to RL for position sizing
```

### Why This Gets You to 18-25%

```
Without Discovery:
├─ Signal: "similarity to past winners" (no edge, just selection bias)
├─ Expected return: 5-8% (after costs)
└─ Sharpe: 0.8-1.0

With Discovery:
├─ Signal: "momentum in Bull + mid-cap + 15d" (real, validated edge)
├─ Edge quality: +250 bps Sharpe improvement
├─ Expected return: 12-18% (10-15% baseline + 2-3% from tested edges)
├─ Sharpe: 1.3-1.8
└─ RL Position Sizing: +3-5% from dynamic allocation
   TOTAL: 15-23% annual returns
```

### Timeline to First Validated Edge

```
Week 1: Backtest 100 trades
├─ LLM identifies: "momentum in Bull markets"
├─ Validate: +50 bps improvement (borderline)
└─ Action: Keep testing, need more data

Week 2: Backtest 200 trades
├─ LLM refines: "momentum + mid-caps + fast<15"
├─ Validate: +250 bps improvement ✅
└─ Action: Real edge found

Week 3: Generalization
├─ Test across regimes: only Bull works
├─ Test across timeframes: peak at 15d
├─ Test across assets: only mid-cap
└─ Action: Deploy with constraints

Week 4+: Deploy & Iterate
├─ SearchPipeline uses validated momentum edge
├─ RL learns optimal sizing
├─ Monthly re-discovery as StrategyMemory grows
└─ Expected Sharpe: 1.3+
```

---

## How to Update This Doc

**After major discussions:**
1. I'll propose changes (show diff)
2. You approve/modify
3. I update CLAUDE.md
4. I commit & note change at top with date

**Format for updates:**
```
[DATE] [SECTION] — Brief change description
  Added: X
  Modified: Y
  Removed: Z
```
