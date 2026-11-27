import pandas as pd
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

original = pd.read_csv(ROOT_DIR / "energy_monthly_results.csv")
ga = pd.read_csv(ROOT_DIR / "ga_monthly_results.csv")
ma = pd.read_csv(ROOT_DIR / "ma_monthly_results.csv")

def calculate_reduction(original_value, optimized_value):

    if original_value == 0:
        return float("nan")

    reduction = (
        (original_value - optimized_value)
        / original_value
    ) * 100

    return reduction

def create_comparison():

    results = pd.DataFrame({
        "Original": original.iloc[0],
        "GA": ga.iloc[0],
        "MA": ma.iloc[0]
    }).T

    results["cost_savings_percent"] = results["total_cost"].apply(
        lambda cost: calculate_reduction(
            original["total_cost"].iloc[0], cost
        )
    )

    results["peak_reduction_percent"] = results["peak_grid_import_kw"].apply(
        lambda peak: calculate_reduction(
            original["peak_grid_import_kw"].iloc[0], peak
        )
    )

    return results

if __name__ == "__main__":

    comparison = create_comparison()

    print(comparison.round(2).to_string())

    comparison.to_csv(
        ROOT_DIR / "final_comparison.csv",
        index_label="Method"
    )