"""Backfill derived mix-metric fields in MongoDB.

This utility recalculates:
- unitPlan[].pctOfTotal
- totalProductProducedGrams

It updates existing documents in place so legacy records match the current API
response shape.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import get_settings
from database import get_mongo_client
from models.mix_metric import MixMetric


def normalize_metric_document(document: Dict[str, Any]) -> Dict[str, Any] | None:
    """Return a fully normalized metric document or None if invalid."""
    try:
        metric = MixMetric(**document)
    except ValidationError:
        return None
    return metric.model_dump(by_alias=True)


def backfill_mix_metrics(dry_run: bool = False, collection=None) -> Tuple[int, int, int]:
    """Backfill all mix metric documents.

    Returns a tuple of:
    - scanned documents
    - updated documents
    - skipped invalid documents
    """
    if collection is None:
        settings = get_settings()
        client = get_mongo_client()
        collection = client[settings.mongodb_database]["mix_metrics"]

    scanned = 0
    updated = 0
    skipped = 0

    for document in collection.find({}):
        scanned += 1
        normalized = normalize_metric_document(document)
        if normalized is None:
            skipped += 1
            continue

        if normalized != document:
            updated += 1
            if not dry_run:
                collection.replace_one({"_id": normalized["_id"]}, normalized)

    return scanned, updated, skipped


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill derived fields in mix_metrics.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show how many documents would be updated without writing changes.",
    )
    args = parser.parse_args()

    scanned, updated, skipped = backfill_mix_metrics(dry_run=args.dry_run)

    mode = "DRY RUN" if args.dry_run else "DONE"
    print(f"{mode}: scanned={scanned}, updated={updated}, skipped_invalid={skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
