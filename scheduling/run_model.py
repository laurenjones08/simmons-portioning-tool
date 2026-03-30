from pipeline import run_pipeline


def main():
    results = run_pipeline(save_csv=True, output_dir="outputs", tee=True)

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

    print("nonpreferred_usage:")
    print(results["outputs"]["nonpreferred_usage"].head(20), "\n")

    print("All outputs written to: outputs")


if __name__ == "__main__":
    main()