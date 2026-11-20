
from pathlib import Path
import pandas as pd


# ---------------------------------------------------------
# Project Constants
# ---------------------------------------------------------

NUMBER_OF_DAYS = 30

SLOTS_PER_DAY = 48

SLOT_MINUTES = 30

SLOT_HOURS = SLOT_MINUTES / 60

TOTAL_SLOTS = NUMBER_OF_DAYS * SLOTS_PER_DAY


# ---------------------------------------------------------
# Dataset Configuration
# ---------------------------------------------------------

START_TIMESTAMP = pd.Timestamp("2026-06-01 00:00:00")

EXPECTED_TIMESTAMPS = pd.date_range(
    start=START_TIMESTAMP,
    periods=TOTAL_SLOTS,
    freq="30min"
)

EXPECTED_APPLIANCE_IDS = [
    "A01", "A02", "A03", "A04", "A05"
]

DEFAULT_DATA_DIR = Path("data")


# ---------------------------------------------------------
# Fitness Configuration
# ---------------------------------------------------------

COST_WEIGHT = 0.5

PEAK_WEIGHT = 0.5

# Large penalty for every constraint violation
CONSTRAINT_PENALTY = 1000.0