from pipeline import run_pipeline


def run_for_job(
    job=None,
    short_term_file=None,
    output_dir="outputs",
    tee=False,
    plan_start_date="2026-01-05",
    horizon_days=12,
    plant_id=None,
    sku_ids=None,
    progress_callback=None,
):
    """Run the scheduling pipeline for a job request."""
    return run_pipeline(
        job=job,
        short_term_file=short_term_file,
        save_csv=True,
        output_dir=output_dir,
        tee=tee,
        plan_start_date=plan_start_date,
        horizon_days=horizon_days,
        plant_id=plant_id,
        sku_ids=sku_ids,
        progress_callback=progress_callback,
    )


def main():
    # Put your weekly short-term demand Excel file path here.
    # If you want to test long-term demand only, leave this as None.
    short_term_file = None
    # short_term_file = "data/short_term_demand.xlsx"

    results = run_for_job(short_term_file=short_term_file, output_dir="outputs", tee=True)

    print("Model solved successfully.\n")

    print("x_long_nonzero:")
    print(results["outputs"]["x_long_nonzero"].head(20), "\n")

    print("line_schedule:")
    print(results["outputs"]["line_schedule"].head(20), "\n")

    print("pattern_mix_by_date:")
    print(results["outputs"]["pattern_mix_by_date"].head(10), "\n")

    print("line_load_by_date:")
    print(results["outputs"]["line_load_by_date"].head(20), "\n")

    print("production_vs_demand_by_date:")
    print(results["outputs"]["production_vs_demand_by_date"].head(25), "\n")

    print("monthly_contract_summary:")
    print(results["outputs"]["monthly_contract_summary"].head(20), "\n")

    print("bucket_usage_by_date:")
    print(results["outputs"]["bucket_usage_by_date"].head(20), "\n")

    print("All outputs written to: outputs")


if __name__ == "__main__":
    main()
