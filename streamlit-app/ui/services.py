"""Lightweight service wrappers for Streamlit UI pages.

These wrappers call the existing `api_client` functions and provide
small convenience helpers used by the UI. Keep this thin — do not reimplement
business logic.
"""
import os
import sys
from typing import Any

# Ensure we can import api_client when loaded from the pages folder
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from api_client import search_mixes, search_mix_metrics, list_jobs  # noqa: E402


def get_recent_mixes(limit: int = 10) -> list[dict]:
    try:
        mixes = search_mixes({}) or []
        # naive: return most recent by createdAt if present
        mixes_sorted = sorted(mixes, key=lambda m: m.get("createdAt", ""), reverse=True)
        return mixes_sorted[:limit]
    except Exception:
        return []


def get_kpis_sample() -> dict[str, Any]:
    """Build a small set of KPIs using available mix and job endpoints.

    This is intentionally lightweight for the overview page.
    """
    kpis: dict[str, Any] = {
        "candidate_count": 0,
        "avg_upgrade": None,
        "recent_jobs": [],
    }

    mixes = get_recent_mixes(limit=5)
    kpis["candidate_count"] = sum(1 for _ in mixes)

    # try to derive average upgrade from mix metrics of the first mix
    if mixes:
        try:
            metrics = search_mix_metrics({"mixId": mixes[0].get("_id")}) or []
            if metrics:
                upgrades = [m.get("upgradePercentage") for m in metrics if m.get("upgradePercentage") is not None]
                if upgrades:
                    kpis["avg_upgrade"] = sum(upgrades) / len(upgrades)
        except Exception:
            kpis["avg_upgrade"] = None

    try:
        jobs = list_jobs() or []
        kpis["recent_jobs"] = sorted(jobs, key=lambda j: j.get("createdAt", ""), reverse=True)[:5]
    except Exception:
        kpis["recent_jobs"] = []

    return kpis
