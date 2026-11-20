
import numpy as np

from src.config import (
    COST_WEIGHT,
    PEAK_WEIGHT,
    CONSTRAINT_PENALTY,
    NUMBER_OF_DAYS,
)

from src.data_loader import load_datasets, prepare_inputs
from src.chromosome import (
    build_gene_metadata,
    build_original_chromosome,
)
from src.energy_calculations import evaluate_day


class FitnessEvaluator:

    def __init__(
        self,
        day_index,
        appliances,
        prices,
        solar,
        base_load,
    ):
        self.day_index = day_index
        self.appliances = appliances
        self.prices = prices
        self.solar = solar
        self.base_load = base_load

        self.metadata = build_gene_metadata(appliances)

        self.original_chromosome = build_original_chromosome(
            appliances
        )

        self.baseline_metrics = evaluate_day(
            day_index=day_index,
            genes=self.original_chromosome,
            appliances=appliances,
            prices=prices,
            solar=solar,
            base_load=base_load,
            metadata=self.metadata,
            return_slot_profile=False,
        )

        self.baseline_cost = self.baseline_metrics["total_cost"]
        self.baseline_peak = self.baseline_metrics["peak_grid_import_kw"]

    def evaluate_with_details(self, chromosome):

        chromosome = np.asarray(chromosome, dtype=int)

        result = evaluate_day(
            day_index=self.day_index,
            genes=chromosome,
            appliances=self.appliances,
            prices=self.prices,
            solar=self.solar,
            base_load=self.base_load,
            metadata=self.metadata,
            return_slot_profile=False,
        )

        candidate_cost = result["total_cost"]
        candidate_peak = result["peak_grid_import_kw"]
        violations = result["constraint_violations"]

        cost_ratio = (
            candidate_cost / self.baseline_cost
            if self.baseline_cost > 0
            else candidate_cost
        )

        peak_ratio = (
            candidate_peak / self.baseline_peak
            if self.baseline_peak > 0
            else candidate_peak
        )

        objective = (
            COST_WEIGHT * cost_ratio
            + PEAK_WEIGHT * peak_ratio
        )

        penalty = CONSTRAINT_PENALTY * violations

        total_objective = objective + penalty

        fitness = 1.0 / (1.0 + total_objective)

        return {
            "fitness": float(fitness),
            "objective": float(objective),
            "penalty": float(penalty),
            "total_objective": float(total_objective),
            "electricity_cost": float(candidate_cost),
            "peak_grid_import_kw": float(candidate_peak),
            "constraint_violations": int(violations),
            "cost_ratio": float(cost_ratio),
            "peak_ratio": float(peak_ratio),
        }

    def evaluate(self, chromosome):
        return self.evaluate_with_details(chromosome)["fitness"]


def create_daily_evaluators(data_dir=None):

    if data_dir is None:
        appliances, prices, solar, base_load = load_datasets()
    else:
        appliances, prices, solar, base_load = load_datasets(data_dir)

    appliances, prices, solar, base_load = prepare_inputs(
        appliances,
        prices,
        solar,
        base_load,
    )

    evaluators = []

    for day in range(NUMBER_OF_DAYS):

        evaluators.append(
            FitnessEvaluator(
                day_index=day,
                appliances=appliances,
                prices=prices,
                solar=solar,
                base_load=base_load,
            )
        )

    return evaluators
