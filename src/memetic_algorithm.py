import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import NUMBER_OF_DAYS, DEFAULT_DATA_DIR
from src.genetic_algorithm import GeneticAlgorithm
from src.data_loader import load_datasets, prepare_inputs
from src.chromosome import build_original_chromosome
from src.monthly_evaluation import evaluate_month
from src.fitness_function import create_daily_evaluators


class MemeticAlgorithm(GeneticAlgorithm):

    def __init__(
        self,
        evaluator,
        population_size=40,
        generations=50,
        crossover_rate=0.8,
        mutation_rate=0.1,
        tournament_size=3,
        elitism=2,
        random_seed=None,
        local_search_rate=0.3,      # chance that a child gets local search
        local_search_iterations=5,  # max improvement rounds per child
        local_search_target="children",  # "children" or "best"
        local_search_count=3,       # how many top individuals get local search ("best" mode)
    ):
        super().__init__(
            evaluator=evaluator,
            population_size=population_size,
            generations=generations,
            crossover_rate=crossover_rate,
            mutation_rate=mutation_rate,
            tournament_size=tournament_size,
            elitism=elitism,
            random_seed=random_seed,
        )

        if local_search_target not in ("children", "best"):
            raise ValueError("local_search_target must be 'children' or 'best'")

        self.local_search_rate = local_search_rate
        self.local_search_iterations = local_search_iterations
        self.local_search_target = local_search_target
        self.local_search_count = local_search_count

        # counts every call to the fitness function (used to compare GA vs MA fairly)
        self.evaluations = 0

    # -----------------------------------------------------
    # Fitness (counted)
    # -----------------------------------------------------

    def evaluate(self, chromosome):
        self.evaluations += 1
        return self.evaluator.evaluate(chromosome)

    # -----------------------------------------------------
    # Neighbors: move ONE gene by one slot (30 minutes)
    # -----------------------------------------------------

    def get_neighbors(self, chromosome):
        neighbors = []

        for i in range(self.gene_count):
            for step in (-1, 1):

                new_value = chromosome[i] + step

                # stay inside this gene's allowed time window
                if new_value < self.lower[i] or new_value > self.upper[i]:
                    continue

                neighbor = chromosome.copy()
                neighbor[i] = new_value
                neighbors.append(neighbor)

        return neighbors

    # -----------------------------------------------------
    # Local search (hill climbing)
    # -----------------------------------------------------

    def local_search(self, chromosome, fitness):
        
        current = chromosome.copy()
        current_fitness = fitness

        for _ in range(self.local_search_iterations):

            best_neighbor = None
            best_neighbor_fitness = current_fitness

            for neighbor in self.get_neighbors(current):

                neighbor_fitness = self.evaluate(neighbor)

                if neighbor_fitness > best_neighbor_fitness:
                    best_neighbor = neighbor
                    best_neighbor_fitness = neighbor_fitness

            if best_neighbor is None:
                break   # local optimum: no neighbor is better

            current = best_neighbor
            current_fitness = best_neighbor_fitness

        return current, current_fitness

    # -----------------------------------------------------
    # Main loop (same as the GA, plus local search on children)
    # -----------------------------------------------------

    def run(self, initial_chromosome=None, verbose=True):

        self.evaluations = 0

        population = self.create_population()

        if initial_chromosome is not None:
            population[0] = np.asarray(
                initial_chromosome,
                dtype=int
            ).copy()

        best_chromosome = None
        best_fitness = -np.inf
        history = []

        for generation in range(self.generations):

            fitness_values = np.asarray([
                self.evaluate(individual)
                for individual in population
            ])

            # ---- memetic step ("best" mode): improve the top individuals ----
            if self.local_search_target == "best":

                top_indices = np.argsort(fitness_values)[::-1][
                    :self.local_search_count
                ]

                for index in top_indices:
                    improved, improved_fitness = self.local_search(
                        population[index], fitness_values[index]
                    )
                    population[index] = improved
                    fitness_values[index] = improved_fitness

            generation_best_index = np.argmax(fitness_values)
            generation_best_fitness = fitness_values[generation_best_index]

            if generation_best_fitness > best_fitness:
                best_fitness = float(generation_best_fitness)
                best_chromosome = population[generation_best_index].copy()

            history.append(best_fitness)

            if verbose and (
                generation == 0
                or (generation + 1) % 10 == 0
                or generation == self.generations - 1
            ):
                print(
                    f"Generation {generation + 1}/{self.generations} "
                    f"| Best Fitness: {best_fitness:.6f} "
                    f"| Evaluations: {self.evaluations}"
                )

            ranked_indices = np.argsort(fitness_values)[::-1]

            new_population = [
                population[index].copy()
                for index in ranked_indices[:self.elitism]
            ]

            while len(new_population) < self.population_size:

                parent1 = self.select_parent(population, fitness_values)
                parent2 = self.select_parent(population, fitness_values)

                child1, child2 = self.crossover(parent1, parent2)

                child1 = self.mutate(child1)
                child2 = self.mutate(child2)

                # ---- the memetic step: local search on some children ----
                children = []

                for child in (child1, child2):

                    if (
                        self.local_search_target == "children"
                        and self.rng.random() < self.local_search_rate
                    ):
                        child_fitness = self.evaluate(child)
                        child, _ = self.local_search(child, child_fitness)

                    children.append(child)

                new_population.append(children[0])

                if len(new_population) < self.population_size:
                    new_population.append(children[1])

            population = np.asarray(new_population)

        # final check: local search may have produced the best individual
        # in the last generation, so evaluate the final population too
        final_fitness = np.asarray([
            self.evaluate(individual)
            for individual in population
        ])

        final_best_index = np.argmax(final_fitness)

        if final_fitness[final_best_index] > best_fitness:
            best_fitness = float(final_fitness[final_best_index])
            best_chromosome = population[final_best_index].copy()

        details = self.evaluator.evaluate_with_details(best_chromosome)

        return {
            "best_chromosome": best_chromosome,
            "best_fitness": best_fitness,
            "best_details": details,
            "history": history,
            "evaluations": self.evaluations,
        }


# =========================================================
# Runner: optimize all 30 days with the Memetic Algorithm
#
# Run from the project root folder with:
#     python -m src.memetic_algorithm
#     python -m src.memetic_algorithm --data-dir dataset
# =========================================================

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--population", type=int, default=40)
    parser.add_argument("--generations", type=int, default=50)
    parser.add_argument("--ls-rate", type=float, default=0.1,
                        help="chance that a child gets local search")
    parser.add_argument("--ls-iterations", type=int, default=3,
                        help="max local search rounds per child")
    parser.add_argument("--ls-target", choices=["children", "best"],
                        default="children",
                        help="apply local search to random children or to the best individuals")
    parser.add_argument("--ls-count", type=int, default=3,
                        help="how many top individuals get local search (best mode)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading datasets...")

    appliances, prices, solar, base_load = load_datasets(args.data_dir)

    appliances, prices, solar, base_load = prepare_inputs(
        appliances, prices, solar, base_load,
    )

    original_chromosome = build_original_chromosome(appliances)
    original_month = np.tile(original_chromosome, (NUMBER_OF_DAYS, 1))

    print("\nEvaluating original schedule...")

    original_daily, original_monthly, _ = evaluate_month(
        chromosome_by_day=original_month,
        appliances=appliances,
        prices=prices,
        solar=solar,
        base_load=base_load,
    )

    print("\nCreating daily fitness evaluators...")

    evaluators = create_daily_evaluators(args.data_dir)

    optimized_chromosomes = []
    ma_results = []
    history_rows = []

    print("\nStarting Memetic Algorithm...")

    for day, evaluator in enumerate(evaluators):

        print(f"\nOptimizing Day {day + 1}/{NUMBER_OF_DAYS}")

        ma = MemeticAlgorithm(
            evaluator=evaluator,
            population_size=args.population,
            generations=args.generations,
            crossover_rate=0.8,
            mutation_rate=0.1,
            tournament_size=3,
            elitism=2,
            random_seed=42 + day,
            local_search_rate=args.ls_rate,
            local_search_iterations=args.ls_iterations,
            local_search_target=args.ls_target,
            local_search_count=args.ls_count,
        )

        result = ma.run(
            initial_chromosome=evaluator.original_chromosome,
            verbose=True,
        )

        optimized_chromosomes.append(result["best_chromosome"])

        ma_results.append({
            "day": day + 1,
            "evaluations": result["evaluations"],
            **result["best_details"],
        })

        for generation, value in enumerate(result["history"]):
            history_rows.append({
                "day": day + 1,
                "generation": generation + 1,
                "best_fitness": value,
            })

    optimized_month = np.asarray(optimized_chromosomes, dtype=int)

    print("\nEvaluating optimized monthly schedule...")

    optimized_daily, optimized_monthly, optimized_slots = evaluate_month(
        chromosome_by_day=optimized_month,
        appliances=appliances,
        prices=prices,
        solar=solar,
        base_load=base_load,
    )

    daily_comparison = pd.DataFrame({
        "day": np.arange(1, NUMBER_OF_DAYS + 1),
        "original_cost": original_daily["total_cost"],
        "optimized_cost": optimized_daily["total_cost"],
        "original_peak_kw": original_daily["peak_grid_import_kw"],
        "optimized_peak_kw": optimized_daily["peak_grid_import_kw"],
        "original_violations": original_daily["constraint_violations"],
        "optimized_violations": optimized_daily["constraint_violations"],
    })

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

    print("\n========== MONTHLY COMPARISON (Original vs MA) ==========")
    print(monthly_comparison.to_string(index=False))

    optimized_daily.to_csv(output_dir / "ma_daily_results.csv", index=False)
    optimized_slots.to_csv(output_dir / "ma_slot_results.csv", index=False)
    pd.DataFrame([optimized_monthly]).to_csv(
        output_dir / "ma_monthly_results.csv", index=False)
    daily_comparison.to_csv(output_dir / "ma_daily_comparison.csv", index=False)
    monthly_comparison.to_csv(
        output_dir / "ma_monthly_comparison.csv", index=False)
    pd.DataFrame(ma_results).to_csv(
        output_dir / "ma_fitness_results.csv", index=False)
    pd.DataFrame(history_rows).to_csv(
        output_dir / "ma_history.csv", index=False)

    print("\nAll memetic results saved successfully.")


if __name__ == "__main__":
    main()