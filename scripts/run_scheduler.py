"""
run_scheduler.py — Background process: triggers daily agent run at 09:30 IST (04:00 UTC).

Run by start.sh in background. Stays alive for the lifetime of the container.
Uses APScheduler BlockingScheduler so it doesn't return until killed.
"""

import logging
import sys
from pathlib import Path

import pytz
from apscheduler.schedulers.blocking import BlockingScheduler

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logging import setup_logging

setup_logging("INFO")
logger = logging.getLogger("scheduler")


def run_agent_job() -> None:
    """Execute the full agent run. Called by APScheduler at 09:30 IST."""
    logger.info("=" * 60)
    logger.info("SCHEDULED RUN: 09:30 IST agent triggered")
    logger.info("=" * 60)
    try:
        from src.agent.runner import main
        main()
        logger.info("SCHEDULED RUN: Completed successfully")
    except Exception as e:
        logger.exception("SCHEDULED RUN: Failed — %s", e)


if __name__ == "__main__":
    ist = pytz.timezone("Asia/Kolkata")

    scheduler = BlockingScheduler(timezone=pytz.utc)

    # 09:30 IST = 04:00 UTC, Monday–Friday
    scheduler.add_job(
        run_agent_job,
        trigger="cron",
        day_of_week="mon-fri",
        hour=4,
        minute=0,
        timezone=pytz.utc,
        id="daily_agent_run",
        name="Daily Agent Run (09:30 IST)",
        misfire_grace_time=1800,   # allow up to 30 min late if container just started
        coalesce=True,             # don't stack up missed runs
    )

    logger.info("Scheduler started. Next run: 04:00 UTC (09:30 IST) Mon–Fri.")
    logger.info("Jobs: %s", scheduler.get_jobs())

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped.")
