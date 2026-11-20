
import pandas as pd
import numpy as np


def load_datasets():

    appliances = pd.read_csv(
        "data/Appliances.csv"
    )

    prices = pd.read_csv(
        "data/Electricity_Prices.csv"
    )

    solar = pd.read_csv(
        "data/Solar_Generation.csv"
    )

    base_load = pd.read_csv(
        "data/Base_Load.csv"
    )

    return appliances, prices, solar, base_load


def inspect_datasets():

    appliances, prices, solar, base_load = load_datasets()

    datasets = {
        "Appliances": appliances,
        "Electricity Prices": prices,
        "Solar Generation": solar,
        "Base Load": base_load
    }

    for name, df in datasets.items():

        print(f"\n{'='*40}")
        print(name)
        print("="*40)

        print("\nFirst 5 Rows:")
        print(df.head())

        print("\nShape:")
        print(df.shape)

        print("\nData Types:")
        print(df.dtypes)

        print("\nMissing Values:")
        print(df.isnull().sum())

        print("\nDuplicates:")
        print(df.duplicated().sum())


if __name__ == "__main__":
    inspect_datasets()
