
import numpy as np


class GeneticAlgorithm:

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
    ):
        self.evaluator = evaluator
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.elitism = elitism
        self.rng = np.random.default_rng(random_seed)

        self.metadata = evaluator.metadata
        self.lower = self.metadata["gene_lower_bounds"]
        self.upper = self.metadata["gene_upper_bounds"]
        self.gene_count = len(self.lower)

        if self.gene_count == 0:
            raise ValueError("Chromosome has no genes.")

    def create_individual(self):
        return self.rng.integers(
            self.lower,
            self.upper + 1
        ).astype(int)

    def create_population(self):
        return np.asarray([
            self.create_individual()
            for _ in range(self.population_size)
        ])

    def select_parent(self, population, fitness_values):
        candidates = self.rng.choice(
            len(population),
            size=min(self.tournament_size, len(population)),
            replace=False
        )

        winner = candidates[
            np.argmax(fitness_values[candidates])
        ]

        return population[winner].copy()

    def crossover(self, parent1, parent2):
        if self.rng.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()

        mask = self.rng.random(self.gene_count) < 0.5

        child1 = np.where(mask, parent1, parent2)
        child2 = np.where(mask, parent2, parent1)

        return child1, child2

    def mutate(self, chromosome):
        child = chromosome.copy()

        for i in range(self.gene_count):
            if self.rng.random() < self.mutation_rate:
                child[i] = self.rng.integers(
                    self.lower[i],
                    self.upper[i] + 1
                )

        return child

    def run(self, initial_chromosome=None, verbose=True):

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
                self.evaluator.evaluate(individual)
                for individual in population
            ])

            generation_best_index = np.argmax(fitness_values)
            generation_best_fitness = fitness_values[
                generation_best_index
            ]

            if generation_best_fitness > best_fitness:
                best_fitness = float(generation_best_fitness)
                best_chromosome = population[
                    generation_best_index
                ].copy()

            history.append(best_fitness)

            if verbose and (
                generation == 0
                or (generation + 1) % 10 == 0
                or generation == self.generations - 1
            ):
                print(
                    f"Generation {generation + 1}/{self.generations} "
                    f"| Best Fitness: {best_fitness:.6f}"
                )

            ranked_indices = np.argsort(fitness_values)[::-1]

            new_population = [
                population[index].copy()
                for index in ranked_indices[:self.elitism]
            ]

            while len(new_population) < self.population_size:

                parent1 = self.select_parent(
                    population, fitness_values
                )
                parent2 = self.select_parent(
                    population, fitness_values
                )

                child1, child2 = self.crossover(
                    parent1, parent2
                )

                child1 = self.mutate(child1)
                child2 = self.mutate(child2)

                new_population.append(child1)

                if len(new_population) < self.population_size:
                    new_population.append(child2)

            population = np.asarray(new_population)

        details = self.evaluator.evaluate_with_details(
            best_chromosome
        )

        return {
            "best_chromosome": best_chromosome,
            "best_fitness": best_fitness,
            "best_details": details,
            "history": history,
        }
