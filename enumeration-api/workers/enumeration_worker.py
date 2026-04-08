"""Container entrypoint for staged, long-running enumeration jobs."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

# Ensure local app modules (config, database, services) resolve when run as a module.
APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from config import get_settings
from database import close_mongo_connection, get_mongo_client
from services.enumeration_staged_runner import StagedEnumerationRunner


logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("enumeration-worker")


def _parse_sku_list(value: str) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> int:
    settings = get_settings()
    run_id = os.getenv("ENUMERATION_RUN_ID", "default-enumeration-run")
    batch_size = int(os.getenv("ENUMERATION_BATCH_SIZE", "1000"))
    max_combination_size = int(os.getenv("ENUMERATION_MAX_COMBINATION_SIZE", "4"))
    sku_filter = _parse_sku_list(os.getenv("ENUMERATION_SKUS", ""))

    logger.info("Starting enumeration worker")
    logger.info("runId=%s batchSize=%s maxCombinationSize=%s", run_id, batch_size, max_combination_size)

    client = get_mongo_client()
    db = client[settings.mongodb_database]

    try:
        runner = StagedEnumerationRunner(
            database=db,
            run_id=run_id,
            batch_size=batch_size,
            max_combination_size=max_combination_size,
            sku_trade_numbers=sku_filter,
        )
        run_doc = runner.run()
        logger.info("Enumeration run completed: %s", run_doc)
        return 0
    except Exception as exc:
        logger.exception("Enumeration worker failed: %s", exc)
        return 1
    finally:
        close_mongo_connection()


if __name__ == "__main__":
    sys.exit(main())
