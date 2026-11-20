
import numpy as np
import pandas as pd

from src.config import NUMBER_OF_DAYS

from src.chromosome import build_gene_metadata

from src.energy_calculations import evaluate_day


# ---------------------------------------------------------
# Evaluate All 30 Days
# ---------------------------------------------------------

def evaluate_month(
    chromosome_by_day,
    appliances,
    prices,
    solar,
    base_load,
):

    chromosome_by_day = np.asarray(
        chromosome_by_day,
        dtype=int
    )

    if chromosome_by_day.shape[0] != NUMBER_OF_DAYS:
        raise ValueError(
            f"Expected {NUMBER_OF_DAYS} daily chromosomes."
        )

    metadata = build_gene_metadata(appliances)

    daily_results = []

    slot_profiles = []

    # Evaluate each day
    for day in range(NUMBER_OF_DAYS):

        result = evaluate_day(
            day_index=day,
            genes=chromosome_by_day[day],
            appliances=appliances,
            prices=prices,
            solar=solar,
            base_load=base_load,
            metadata=metadata,
            return_slot_profile=True
        )

        slot_profiles.append(
            result.pop("slot_profile")
        )

        daily_results.append(result)

    daily_metrics = pd.DataFrame(daily_results)

    slot_profile = pd.concat(
        slot_profiles,
        ignore_index=True
    )

    # -----------------------------------------------------
    # Monthly Metrics
    # -----------------------------------------------------

    total_solar = daily_metrics[
        "solar_generation_kwh"
    ].sum()

    total_solar_used = (
        daily_metrics["solar_generation_kwh"]
        * daily_metrics["solar_utilization_percent"] / 100
    ).sum()

    monthly_metrics = {

        "total_consumption_kwh": float(
            daily_metrics["total_consumption_kwh"].sum()
        ),

        "appliance_consumption_kwh": float(
            daily_metrics["appliance_consumption_kwh"].sum()
        ),

        "grid_import_kwh": float(
            daily_metrics["grid_import_kwh"].sum()
        ),

        "total_cost": float(
            daily_metrics["total_cost"].sum()
        ),

        "solar_generation_kwh": float(
            total_solar
        ),

        "solar_utilization_percent": float(
            total_solar_used / total_solar * 100
        ) if total_solar > 0 else 0.0,

        "peak_household_demand_kw": float(
            daily_metrics["peak_household_demand_kw"].max()
        ),

        "peak_grid_import_kw": float(
            daily_metrics["peak_grid_import_kw"].max()
        ),

        "average_daily_peak_grid_import_kw": float(
            daily_metrics["peak_grid_import_kw"].mean()
        ),

        "constraint_violations": int(
            daily_metrics["constraint_violations"].sum()
        ),
    }

    return daily_metrics, monthly_metrics, slot_profile
