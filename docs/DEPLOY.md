# AgentQuant — Deployment Guide (Railway)

## Architecture

```
Single Docker container (Railway Web Service)
├── Streamlit dashboard   — public HTTPS URL, mobile-accessible
└── APScheduler daemon    — fires agent at 09:30 IST (04:00 UTC) Mon–Fri

Persistent Volume (mounted at /app)
├── .cache/paper_trades.db       — PaperTrader portfolio & trade history
├── .cache/position_manager.db   — overnight position tracking
├── experiments/results.db       — StrategyMemory (backtest results)
└── data_store/*.parquet         — OHLCV cache (re-downloaded if missing)
```

## Deploy to Railway (one-time setup)

### 1. Push code to GitHub

```bash
git add .
git commit -m "feat: Phase 3 paper trading + deployment infrastructure"
git push origin main
```

### 2. Create Railway project

1. Go to [railway.app](https://railway.app) → **New Project**
2. Choose **Deploy from GitHub repo** → select `AgentQuant`
3. Railway auto-detects `Dockerfile` and `railway.toml`

### 3. Add a Persistent Volume

In Railway dashboard → your service → **Volumes** tab:
- Mount path: `/app`
- Size: 1 GB (enough for SQLite + parquet files)

This keeps paper trade history and strategy memory across deploys.

### 4. Set environment variables

In Railway dashboard → your service → **Variables** tab:

| Variable | Value | Required |
|---|---|---|
| `GEMINI_API_KEY` | your Gemini key | ✅ Yes |
| `OPENAI_API_KEY` | your OpenAI key | Optional |
| `FRED_API_KEY` | your FRED key | Optional |

Railway injects `PORT` automatically — do not set it manually.

### 5. Deploy

Railway deploys automatically on every `git push`. First deploy takes ~3 minutes to build the image.

Your dashboard URL appears in Railway dashboard (format: `https://agentquant-production.up.railway.app`).

---

## Scheduled agent runs

The scheduler (`scripts/run_scheduler.py`) starts automatically inside the container.

- **Schedule:** Monday–Friday, 04:00 UTC = **09:30 AM IST**
- **What it does:** Full agent pipeline → LLM analysis → paper trade execution → updates dashboard
- **Logs:** visible in Railway dashboard → **Logs** tab

To trigger a manual run anytime: open the dashboard → **Agent Runs** page → click **Run Agent Now**.

---

## Useful Railway commands (CLI)

```bash
# Install Railway CLI
npm install -g @railway/cli

# Link local repo to project
railway link

# View live logs
railway logs --tail

# Open dashboard URL
railway open

# Run a one-off command in the deployed container
railway run python -m src.agent.runner

# SSH into container (debugging)
railway shell
```

---

## Cost estimate

| Resource | Cost |
|---|---|
| Railway Hobby plan | $5/month |
| Railway 1 GB volume | ~$0.25/month |
| Gemini API (~20 calls/month) | ~$0.50/month |
| **Total** | **~$5.75/month** |

---

## After first deploy

1. Open your Railway URL on mobile — dashboard loads immediately
2. On first load: no trade history (fresh DB on volume)
3. Run **Run Agent Now** once to seed the first paper trade
4. From 09:30 AM IST onwards, agent runs automatically each weekday
5. Portfolio and trade history update in real time in the dashboard
