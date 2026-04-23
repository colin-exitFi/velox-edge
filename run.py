"""Single-process launcher: bot loop + dashboard share state.

Why one process: dashboard needs live broker calls + DB access, bot writes the
DB. Sharing the same Python process means no IPC, no race conditions, no
extra deployment unit.
"""

import asyncio
import sys
import threading

import uvicorn
from loguru import logger

from velox_edge import config, dashboard, main as bot_main, state


def _setup_logging():
    logger.remove()
    logger.add(
        sys.stderr,
        level=config.LOG_LEVEL,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
    )


def _start_dashboard_in_thread():
    """Run uvicorn in a background daemon thread so the bot loop owns the main thread."""

    def _run():
        uvicorn.run(
            dashboard.app,
            host=config.DASHBOARD_HOST,
            port=config.DASHBOARD_PORT,
            log_level="warning",  # quiet uvicorn — we have our own logger
            access_log=False,
        )

    t = threading.Thread(target=_run, name="dashboard", daemon=True)
    t.start()
    logger.info(f"📊 Dashboard listening on http://{config.DASHBOARD_HOST}:{config.DASHBOARD_PORT}")


async def amain():
    state.init_db()
    _start_dashboard_in_thread()
    await bot_main.main_loop()


if __name__ == "__main__":
    _setup_logging()
    try:
        asyncio.run(amain())
    except KeyboardInterrupt:
        logger.info("Shutdown requested")
