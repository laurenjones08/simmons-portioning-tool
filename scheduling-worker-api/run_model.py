from __future__ import annotations

from typing import Any, Dict, Optional

from pipeline import run_pipeline


def run_for_job(
    job: Any,
    short_term_file: Optional[str],
    output_dir: str,
    tee: bool,
    plan_start_date: str,
    horizon_days: int,
    plant_id: Optional[str],
    sku_ids: Optional[list[str]],
) -> Dict[str, Any]:
    """Run the scheduling model for a submitted worker job.

    This wrapper is intentionally silent (no print statements) so job execution
    can be triggered from API job creation without console output side effects.
    """
    return run_pipeline(
        job=job,
        short_term_file=short_term_file,
        save_csv=False,
        output_dir=output_dir,
        tee=tee,
        plan_start_date=plan_start_date,
        horizon_days=horizon_days,
        plant_id=plant_id,
        sku_ids=sku_ids,
    )

