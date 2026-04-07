from pipeline import run_pipeline


def main():
    # Put your weekly short-term demand Excel file path here.
    # If you want to test long-term demand only, leave this as None.
    short_term_file = None
    # short_term_file = "data/short_term_demand.xlsx"

    results = run_pipeline(
        short_term_file=short_term_file,
        save_csv=True,
        output_dir="outputs",
        tee=True
    )

    print("Model solved successfully.\n")

    print("x_long_nonzero:")
    print(results["outputs"]["x_long_nonzero"].head(20), "\n")

    print("line_schedule:")
    print(results["outputs"]["line_schedule"].head(20), "\n")

    print("pattern_mix_by_shift:")
    print(results["outputs"]["pattern_mix_by_shift"].head(10), "\n")

    print("line_load_by_shift:")
    print(results["outputs"]["line_load_by_shift"].head(20), "\n")

    print("production_vs_demand_by_shift:")
    print(results["outputs"]["production_vs_demand_by_shift"].head(25), "\n")

    print("bucket_usage_by_shift:")
    print(results["outputs"]["bucket_usage_by_shift"].head(20), "\n")

    print("All outputs written to: outputs")


if __name__ == "__main__":
    main()