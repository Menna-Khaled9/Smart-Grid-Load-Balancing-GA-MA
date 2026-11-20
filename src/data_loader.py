
from pathlib import Path
import pandas as pd

from src.config import DEFAULT_DATA_DIR


# ---------------------------------------------------------
# Load Datasets
# ---------------------------------------------------------

def load_datasets(data_dir=DEFAULT_DATA_DIR):

    data_dir = Path(data_dir)

    appliances = pd.read_csv(data_dir / "Appliances.csv")
    prices = pd.read_csv(data_dir / "Electricity_prices.csv")
    solar = pd.read_csv(data_dir / "Solar_Generation.csv")
    base_load = pd.read_csv(data_dir / "Base_Load.csv")

    return appliances, prices, solar, base_load


# ---------------------------------------------------------
# Convert Time to Minutes
# ---------------------------------------------------------

def _time_to_minutes(value):

    parsed = pd.to_datetime(str(value), format="%H:%M")

    return parsed.hour * 60 + parsed.minute


################ Convert Interruptibility to Boolean


def _parse_boolean_column(series, column_name):

    if series.dtype == bool:
        return series.astype(bool)

    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
    }

    converted = series.astype(str).str.strip().str.lower().map(mapping)

    if converted.isna().any():
        raise ValueError(
            f"Invalid boolean value found in column: {column_name}"
        )

    return converted.astype(bool)


# ---------------------------------------------------------
# Prepare Time Series
# ---------------------------------------------------------

def _prepare_time_series(df, name, required_columns):

    result = df.copy()

    result["timestamp"] = pd.to_datetime(result["timestamp"])

    result = (
        result.sort_values("timestamp")
        .reset_index(drop=True)
    )

    for column in required_columns:

        if column == "timestamp":
            continue

        result[column] = pd.to_numeric(
            result[column],
            errors="raise"
        )

    return result


# ---------------------------------------------------------
# Prepare All Inputs
# ---------------------------------------------------------

def prepare_inputs(appliances, prices, solar, base_load):

    appliances = appliances.copy()

    appliances["appliance_id"] = ( appliances["appliance_id"].astype(str) )

    appliances = ( appliances.sort_values("appliance_id") .reset_index(drop=True) )

    appliances["interruptible"] = _parse_boolean_column( appliances["interruptible"], "interruptible" )

    numeric_columns = [
        "rated_power_watts",
        "energy_consumption_kwh",
        "duration_minutes",
        "segment_minutes",
    ]

    for column in numeric_columns:

        appliances[column] = pd.to_numeric(
            appliances[column],
            errors="raise"
        )

    appliances["earliest_minute"] = (
        appliances["earliest_start_time"].map(_time_to_minutes)
    )

    appliances["latest_minute"] = (
        appliances["latest_start_time"].map(_time_to_minutes)
    )

    appliances["preferred_minute"] = (
        appliances["preferred_start_time"].map(_time_to_minutes)
    )

    prices = _prepare_time_series(
        prices,
        "Electricity Prices",
        ["timestamp", "electricity_price_per_kwh"]
    )

    solar = _prepare_time_series(
        solar,
        "Solar Generation",
        [
            "timestamp",
            "solar_generation_watts",
            "solar_generation_kwh"
        ]
    )

    base_load = _prepare_time_series(
        base_load,
        "Base Load",
        [
            "timestamp",
            "base_load_watts",
            "base_load_kwh"
        ]
    )

    return appliances, prices, solar, base_load
