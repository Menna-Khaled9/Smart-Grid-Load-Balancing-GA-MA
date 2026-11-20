
import numpy as np
import pandas as pd

from src.config import NUMBER_OF_DAYS, DEFAULT_DATA_DIR

from src.data_loader import load_datasets, prepare_inputs
from src.chromosome import build_original_chromosome
from src.monthly_evaluation import evaluate_month
from src.fitness_function import create_daily_evaluators
from src.genetic_algorithm import GeneticAlgorithm


def main():

    print("Loading datasets...")

    appliances, prices, solar, base_load = load_datasets(
        DEFAULT_DATA_DIR
    )

    appliances, prices, solar, base_load = prepare_inputs(
        appliances,
        prices,
        solar,
        base_load,
    )

    original_chromosome = build_original_chromosome(appliances)

    original_month = np.tile(
        original_chromosome,
        (NUMBER_OF_DAYS, 1),
    )

    print("\nEvaluating original schedule...")

    original_daily, original_monthly, original_slots = evaluate_month(
        chromosome_by_day=original_month,
        appliances=appliances,
        prices=prices,
        solar=solar,
        base_load=base_load,
    )

    print("\nCreating daily fitness evaluators...")

    evaluators = create_daily_evaluators(DEFAULT_DATA_DIR)

    optimized_chromosomes = []
    ga_results = []

    print("\nStarting Genetic Algorithm...")

    for day, evaluator in enumerate(evaluators):

        print(f"\nOptimizing Day {day + 1}/{NUMBER_OF_DAYS}")

        ga = GeneticAlgorithm(
            evaluator=evaluator,
            population_size=40,
            generations=50,
            crossover_rate=0.8,
            mutation_rate=0.1,
            tournament_size=3,
            elitism=2,
            random_seed=42 + day,
        )

        result = ga.run(
            initial_chromosome=evaluator.original_chromosome,
            verbose=True,
        )

        optimized_chromosomes.append(
            result["best_chromosome"]
        )

        ga_results.append({
            "day": day + 1,
            **result["best_details"],
        })

    optimized_month = np.asarray(
        optimized_chromosomes,
        dtype=int,
    )

    print("\nEvaluating optimized monthly schedule...")

    optimized_daily, optimized_monthly, optimized_slots = evaluate_month(
        chromosome_by_day=optimized_month,
        appliances=appliances,
        prices=prices,
        solar=solar,
        base_load=base_load,
    )

    # Daily comparison
    daily_comparison = pd.DataFrame({
        "day": np.arange(1, NUMBER_OF_DAYS + 1),
        "original_cost": original_daily["total_cost"],
        "optimized_cost": optimized_daily["total_cost"],
        "original_peak_kw": original_daily["peak_grid_import_kw"],
        "optimized_peak_kw": optimized_daily["peak_grid_import_kw"],
        "original_violations": original_daily["constraint_violations"],
        "optimized_violations": optimized_daily["constraint_violations"],
    })

    # Monthly comparison
    monthly_comparison = pd.DataFrame([
        {
            "metric": metric,
            "original": original_monthly[metric],
            "optimized": optimized_monthly[metric],
            "improvement_percent": (
                (original_monthly[metric] - optimized_monthly[metric])
                / original_monthly[metric] * 100
                if original_monthly[metric] != 0
                else 0.0
            ),
        }
        for metric in [
            "total_cost",
            "grid_import_kwh",
            "peak_grid_import_kw",
            "constraint_violations",
        ]
    ])

    print("\n========== MONTHLY COMPARISON ==========")
    print(monthly_comparison.to_string(index=False))

    print("\n========== DAILY COMPARISON ==========")
    print(daily_comparison.to_string(index=False))

    # Save outputs
    original_daily.to_csv(
        "energy_daily_results.csv", index=False
    )

    original_slots.to_csv(
        "energy_slot_results.csv", index=False
    )

    pd.DataFrame([original_monthly]).to_csv(
        "energy_monthly_results.csv", index=False
    )

    optimized_daily.to_csv(
        "ga_daily_results.csv", index=False
    )

    optimized_slots.to_csv(
        "ga_slot_results.csv", index=False
    )

    pd.DataFrame([optimized_monthly]).to_csv(
        "ga_monthly_results.csv", index=False
    )

    daily_comparison.to_csv(
        "ga_daily_comparison.csv", index=False
    )

    monthly_comparison.to_csv(
        "ga_monthly_comparison.csv", index=False
    )

    pd.DataFrame(ga_results).to_csv(
        "ga_fitness_results.csv", index=False
    )

    print("\nAll results saved successfully.")


if __name__ == "__main__":
    main()
