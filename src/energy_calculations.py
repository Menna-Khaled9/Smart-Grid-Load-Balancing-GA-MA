
import numpy as np
import pandas as pd

from src.config import (
    NUMBER_OF_DAYS,
    SLOTS_PER_DAY,
    SLOT_HOURS,
    START_TIMESTAMP,
    EXPECTED_TIMESTAMPS
)

from src.chromosome import (
    build_gene_metadata,
    build_coverage_matrix
)

from src.constraint_checker import check_constraints


# ---------------------------------------------------------
# Evaluate One Day's Chromosome
# ---------------------------------------------------------

def evaluate_day(
    day_index,
    genes,
    appliances,
    prices,
    solar,
    base_load,
    metadata=None,
    return_slot_profile=True,
):

    genes = np.asarray(genes, dtype=int)

    # Validate day index
    if not 0 <= day_index < NUMBER_OF_DAYS:
        raise ValueError("Invalid day_index.")

    # Build metadata if not supplied
    if metadata is None:

        metadata = build_gene_metadata(appliances)

    # -----------------------------------------------------
    # 1. Build Coverage Matrix
    # -----------------------------------------------------

    coverage = build_coverage_matrix(
        genes,
        appliances,
        metadata
    )

    appliance_on = coverage > 0

    # -----------------------------------------------------
    # 2. Check Hard Constraints
    # -----------------------------------------------------

    violations = check_constraints(
        genes,
        appliances,
        metadata,
        coverage
    )

    # -----------------------------------------------------
    # 3. Extract Daily Data
    # -----------------------------------------------------

    base_w = (
        base_load["base_load_watts"]
        .to_numpy()
        .reshape(NUMBER_OF_DAYS, SLOTS_PER_DAY)[day_index]
    )

    solar_w = (
        solar["solar_generation_watts"]
        .to_numpy()
        .reshape(NUMBER_OF_DAYS, SLOTS_PER_DAY)[day_index]
    )

    price = (
        prices["electricity_price_per_kwh"]
        .to_numpy()
        .reshape(NUMBER_OF_DAYS, SLOTS_PER_DAY)[day_index]
    )

    # -----------------------------------------------------
    # 4. Calculate Appliance Load
    # -----------------------------------------------------

    appliance_power_w = (
        appliances["rated_power_watts"]
        .to_numpy(dtype=float)
    )

    appliance_load_w = (
        appliance_on * appliance_power_w[:, np.newaxis]
    ).sum(axis=0)

    # -----------------------------------------------------
    # 5. Calculate Total Load
    # -----------------------------------------------------

    total_load_w = base_w + appliance_load_w

    # -----------------------------------------------------
    # 6. Calculate Grid Import
    # -----------------------------------------------------

    grid_import_w = np.maximum(
        0,
        total_load_w - solar_w
    )

    grid_import_kwh = (
        grid_import_w * SLOT_HOURS / 1000
    )

    # -----------------------------------------------------
    # 7. Calculate Total Consumption
    # -----------------------------------------------------

    total_consumption_kwh = (
        total_load_w * SLOT_HOURS / 1000
    )

    # -----------------------------------------------------
    # 8. Calculate Electricity Cost
    # -----------------------------------------------------

    cost_per_slot = grid_import_kwh * price

    total_cost = float(cost_per_slot.sum())

    total_consumption = float(
        total_consumption_kwh.sum()
    )

    total_grid_import = float(
        grid_import_kwh.sum()
    )

    # -----------------------------------------------------
    # 9. Calculate Solar Metrics
    # -----------------------------------------------------

    total_solar_kwh = float(
        (solar_w * SLOT_HOURS / 1000).sum()
    )

    solar_used_kwh = float(
        (
            np.minimum(total_load_w, solar_w)
            * SLOT_HOURS / 1000
        ).sum()
    )

    solar_utilization = (
        solar_used_kwh / total_solar_kwh * 100
        if total_solar_kwh > 0
        else 0.0
    )

    # -----------------------------------------------------
    # 10. Prepare Results
    # -----------------------------------------------------

    result = {

        "day_index": day_index,

        "date": (
            START_TIMESTAMP + pd.Timedelta(days=day_index)
        ).date(),

        "total_consumption_kwh": total_consumption,

        "appliance_consumption_kwh": float(
            (appliance_load_w * SLOT_HOURS / 1000).sum()
        ),

        "grid_import_kwh": total_grid_import,

        "total_cost": total_cost,

        "peak_household_demand_kw": float(
            total_load_w.max() / 1000
        ),

        "peak_grid_import_kw": float(
            grid_import_w.max() / 1000
        ),

        "solar_generation_kwh": total_solar_kwh,

        "solar_utilization_percent": float(
            solar_utilization
        ),

        "constraint_violations": violations,
    }

    # -----------------------------------------------------
    # 11. Optional Slot Profile
    # -----------------------------------------------------

    if return_slot_profile:

        day_timestamps = EXPECTED_TIMESTAMPS[
            day_index * SLOTS_PER_DAY:
            (day_index + 1) * SLOTS_PER_DAY
        ]

        result["slot_profile"] = pd.DataFrame({

            "timestamp": day_timestamps,

            "slot_in_day": np.arange(SLOTS_PER_DAY),

            "base_load_watts": base_w,

            "appliance_load_watts": appliance_load_w,

            "total_load_watts": total_load_w,

            "solar_generation_watts": solar_w,

            "grid_import_watts": grid_import_w,

            "grid_import_kwh": grid_import_kwh,

            "electricity_price_per_kwh": price,

            "electricity_cost": cost_per_slot,
        })

    return result
