from pathlib import Path

import pandas as pd
import numpy as np
from matplotlib.figure import Figure

from src.comparison import create_comparison

ROOT_DIR = Path(__file__).resolve().parent.parent

COLORS = ["#EF4444", "#3B82F6", "#22C55E"]

def load_slot_results():

    original = pd.read_csv(
        ROOT_DIR / "energy_slot_results.csv",
        parse_dates=["timestamp"]
    )

    ga = pd.read_csv(
        ROOT_DIR / "ga_slot_results.csv",
        parse_dates=["timestamp"]
    )

    ma = pd.read_csv(
        ROOT_DIR / "ma_slot_results.csv",
        parse_dates=["timestamp"]
    )

    return original, ga, ma

def plot_load_profile(day=1):

    if not 1 <= day <= 30:
        raise ValueError("Day must be between 1 and 30")

    original, ga, ma = load_slot_results()

    fig = Figure(figsize=(12, 5))
    ax = fig.add_subplot(111)

    datasets = [original, ga, ma]
    labels = ["Original", "GA", "MA"]

    for data, label, color in zip(
        datasets, labels, COLORS
    ):

        daily = data[
            data["timestamp"].dt.day == day
        ].sort_values("timestamp")

        hours = daily["slot_in_day"] / 2

        load = daily["grid_import_watts"] / 1000

        ax.step(
            hours,
            load,
            where="post",
            label=label,
            color=color,
            linewidth=2
        )

    ax.set_title(f"Grid Load Profile - Day {day}")
    ax.set_xlabel("Time (Hours)")
    ax.set_ylabel("Grid Load (kW)")

    ax.set_xticks(range(0, 25, 2))
    ax.set_xlim(0, 24)

    ax.legend()
    ax.grid(alpha=0.25)

    fig.tight_layout()

    return fig

def create_bar_chart(metric, title, ylabel):

    results = create_comparison()

    values = results[metric]

    fig = Figure(figsize=(7, 5))
    ax = fig.add_subplot(111)

    bars = ax.bar(
        values.index,
        values,
        color=COLORS,
        width=0.55
    )

    ax.bar_label(
        bars,
        fmt="%.2f",
        padding=5
    )

    ax.set_title(title)
    ax.set_ylabel(ylabel)

    ax.set_ylim(0, values.max() * 1.2)
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()

    return fig

def plot_cost_comparison():

    return create_bar_chart(
        "total_cost",
        "Monthly Electricity Cost",
        "Electricity Cost"
    )
    

def plot_peak_comparison():

    return create_bar_chart(
        "peak_grid_import_kw",
        "Peak Grid Load Comparison",
        "Peak Load (kW)"
    )   

def plot_solar_utilization():

    return create_bar_chart(
        "solar_utilization_percent",
        "Solar Energy Utilization",
        "Utilization (%)"
    )

def plot_grid_import():

    return create_bar_chart(
        "grid_import_kwh",
        "Monthly Grid Energy Import",
        "Energy (kWh)"
    )

def plot_savings():

    results = create_comparison().loc[["GA", "MA"]]

    fig = Figure(figsize=(8, 5))
    ax = fig.add_subplot(111)

    x = np.arange(len(results))
    width = 0.35

    cost_bars = ax.bar(
        x - width / 2,
        results["cost_savings_percent"],
        width,
        label="Cost Savings",
        color="#3B82F6"
    )

    peak_bars = ax.bar(
        x + width / 2,
        results["peak_reduction_percent"],
        width,
        label="Peak Reduction",
        color="#22C55E"
    )

    ax.bar_label(cost_bars, fmt="%.2f%%", padding=3)
    ax.bar_label(peak_bars, fmt="%.2f%%", padding=3)

    ax.set_xticks(x)
    ax.set_xticklabels(results.index)

    ax.set_title("Optimization Efficiency")
    ax.set_ylabel("Improvement (%)")

    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()

    return fig

if __name__ == "__main__":

    output_dir = ROOT_DIR / "graphs"

    output_dir.mkdir(exist_ok=True)

    graphs = {
        "01_load_profile": plot_load_profile(day=1),
        "02_cost_comparison": plot_cost_comparison(),
        "03_peak_comparison": plot_peak_comparison(),
        "04_solar_utilization": plot_solar_utilization(),
        "05_grid_import": plot_grid_import(),
        "06_optimization_efficiency": plot_savings()
    }

    for name, fig in graphs.items():

        fig.savefig(
            output_dir / f"{name}.png",
            dpi=300,
            bbox_inches="tight"
        )

        print(f"Saved: {name}.png")

    print("\nAll graphs generated successfully!")
    

