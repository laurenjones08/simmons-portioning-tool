from pipeline import run_pipeline

def main():
    results = run_pipeline(save_csv=True, output_dir="outputs", tee=True)

    print("Model solved successfully.\n")

    print("Decision Assignments:")
    print(results["outputs"]["decision_assignments"].head(), "\n")

    print("Production vs Demand:")
    print(results["outputs"]["production_vs_demand"].head(), "\n")

    print("Line Load:")
    print(results["outputs"]["line_load"].head(), "\n")


if __name__ == "__main__":
    main()