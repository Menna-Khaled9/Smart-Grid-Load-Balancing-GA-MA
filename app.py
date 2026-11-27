from pathlib import Path
from io import BytesIO

import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.figure import Figure

from src.config import (NUMBER_OF_DAYS, SLOTS_PER_DAY, SLOT_MINUTES,
                        COST_WEIGHT, PEAK_WEIGHT, CONSTRAINT_PENALTY)

ROOT = Path(__file__).resolve().parent
METHODS = ["Original", "GA", "MA"]
PREFIXES = ["energy", "ga", "ma"]
COLORS = ["#EF4444", "#3B82F6", "#16A34A"]
DATA_FILES = {"Appliances": "Appliances.csv",
              "Electricity Prices": "Electricity_prices.csv",
              "Solar Generation": "Solar_Generation.csv",
              "Base Load": "Base_Load.csv"}
METRICS = {
    "total_cost": "Electricity Cost",
    "peak_grid_import_kw": "Peak Grid Load (kW)",
    "grid_import_kwh": "Grid Import (kWh)",
    "cost_savings_percent": "Cost Savings (%)",
    "peak_reduction_percent": "Peak Reduction (%)",
    "solar_utilization_percent": "Solar Utilization (%)",
    "total_consumption_kwh": "Total Consumption (kWh)",
    "appliance_consumption_kwh": "Appliance Consumption (kWh)",
    "solar_generation_kwh": "Solar Generation (kWh)",
    "peak_household_demand_kw": "Peak Household Demand (kW)",
    "constraint_violations": "Constraint Violations",
}
GRAPH_NAMES = ["Load Profile", "Electricity Cost", "Peak Shaving", "Solar Utilization", "Grid Import", "Efficiency"]



def dataset_paths(root=ROOT):
    folder = root / "dataset"
    if not folder.is_dir():
        folder = root / "data"
    available = {p.name.lower(): p for p in folder.iterdir() if p.is_file()}
    return {label: available[filename.lower()] for label, filename in DATA_FILES.items()}


def read_datasets(root=ROOT):
    return {name: pd.read_csv(path) for name, path in dataset_paths(root).items()}


def calculate_reduction(original, optimized):
    return (original - optimized) / original * 100 if original != 0 else float("nan")


def analysis_results(day=None, root=ROOT):
    """Read current results, without caching outdated CSV files."""
    if day is not None and not 1 <= day <= NUMBER_OF_DAYS:
        raise ValueError("Invalid simulation day.")
    rows = {}
    for method, prefix in zip(METHODS, PREFIXES):
        period = "monthly" if day is None else "daily"
        data = pd.read_csv(root / f"{prefix}_{period}_results.csv")
        selected = data if day is None else data[data["day_index"] == day - 1]
        if len(selected) != 1:
            raise ValueError(f"Expected exactly one {period} result for {method}.")
        rows[method] = selected.iloc[0].drop(["day_index", "date"], errors="ignore")
    results = pd.DataFrame(rows).T.astype(float)
    for metric, output in [("total_cost", "cost_savings_percent"),
                           ("peak_grid_import_kw", "peak_reduction_percent")]:
        baseline = results.loc["Original", metric]
        results[output] = results[metric].apply(lambda value: calculate_reduction(baseline, value))
    return results


def load_slot_results(root=ROOT):
    return {method: pd.read_csv(root / f"{prefix}_slot_results.csv", parse_dates=["timestamp"])
            .sort_values("timestamp") for method, prefix in zip(METHODS, PREFIXES)}


def comparison_table(results):
    return results[list(METRICS)].rename(columns=METRICS).T


def period_label(day):
    return f"Monthly ({NUMBER_OF_DAYS} days)" if day is None else f"Day {day} (24 hours)"



def load_profile(day=None, root=ROOT):
    fig = Figure(figsize=(10, 4.4), layout="constrained")
    ax = fig.add_subplot(111)
    for (method, data), color in zip(load_slot_results(root).items(), COLORS):
        if day is None:
            
            peaks = data.groupby(data["timestamp"].dt.normalize())["grid_import_watts"].max() / 1000
            ax.plot(range(1, len(peaks) + 1), peaks.values, marker="o", markersize=3,
                    linewidth=1.8, label=method, color=color)
        else:
            dates = data["timestamp"].dt.normalize().drop_duplicates().tolist()
            daily = data[data["timestamp"].dt.normalize() == dates[day - 1]]
            hours = (daily["slot_in_day"] * SLOT_MINUTES / 60).tolist()
            load = (daily["grid_import_watts"] / 1000).tolist()
            ax.step(hours + [24], load + [load[-1]], where="post", linewidth=1.7,
                    label=method, color=color)
    if day is None:
        ax.set_title("Monthly Load Profile — Daily Peak Grid Load")
        ax.set_xlabel("Simulation Day")
        ax.set_xticks(range(1, NUMBER_OF_DAYS + 1, 2))
        ax.set_xlim(1, NUMBER_OF_DAYS)
    else:
        ax.set_title(f"Grid Load Profile — Day {day}")
        ax.set_xlabel("Hour of Day")
        ax.set_xticks(range(0, 25, 2))
        ax.set_xlim(0, 24)
    ax.set_ylabel("Grid Load (kW)")
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.2)
    ax.legend()
    return fig


def comparison_chart(results, graph, day=None):
    fig = Figure(figsize=(7.3, 4.4), layout="constrained")
    ax = fig.add_subplot(111)
    metrics = {"Electricity Cost": ("total_cost", "Electricity Cost"),
               "Peak Shaving": ("peak_grid_import_kw", "Peak Grid Load (kW)"),
               "Solar Utilization": ("solar_utilization_percent", "Utilization (%)"),
               "Grid Import": ("grid_import_kwh", "Energy (kWh)")}
    if graph == "Efficiency":
        data = results.loc[["GA", "MA"]]
        x = np.arange(len(data))
        for offset, metric, label, color in [(-0.18, "cost_savings_percent", "Cost Savings", COLORS[1]),
                                             (0.18, "peak_reduction_percent", "Peak Reduction", COLORS[2])]:
            bars = ax.bar(x + offset, data[metric], 0.36, label=label, color=color)
            ax.bar_label(bars, fmt="%.2f%%", padding=4)
        ax.set_xticks(x, data.index)
        ax.set_ylabel("Improvement (%)")
        ax.legend()
        ax.axhline(0, color="#777777", linewidth=0.8)
    else:
        metric, ylabel = metrics[graph]
        bars = ax.bar(results.index, results[metric], color=COLORS, width=0.55)
        ax.bar_label(bars, fmt="%.2f", padding=4)
        ax.set_ylabel(ylabel)
        ax.set_ylim(bottom=0)
    ax.margins(y=0.2)
    ax.set_title(f"{graph} — {period_label(day)}")
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    return fig


def figure_bytes(fig):
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=300, bbox_inches="tight")
    return buffer.getvalue()


def select_period(key):
    mode = st.radio("Analysis Mode", ["Daily", "Monthly"], index=1, horizontal=True, key=f"{key}_mode")
    if mode == "Daily":
        return st.selectbox("Select Day", list(range(1, NUMBER_OF_DAYS + 1)), key=f"{key}_day")
    return None


def show_kpis(results):
    cols = st.columns(3)
    optimized = results.loc[["GA", "MA"]]
    for col, (label, metric) in zip(cols, [("Cost Savings", "cost_savings_percent"),
                                          ("Peak Reduction", "peak_reduction_percent"),
                                          ("Solar Utilization", "solar_utilization_percent")]):
        values = optimized[metric].dropna()
        if values.empty:
            col.metric(label, "N/A")
        else:
            best = values.max()
            winners = " & ".join(values.index[np.isclose(values, best, rtol=0, atol=1e-10)])
            col.metric(label, f"{best:.2f}%", help=f"Best method: {winners}")
            col.caption(f"Best method: {winners}")


def show_table(results):
    st.dataframe(comparison_table(results).style.format("{:.2f}", na_rep="N/A"), use_container_width=True)


def dashboard_page():
    st.title("Energy Optimization Dashboard")
    st.caption(f"Monthly simulation · {NUMBER_OF_DAYS} days · {SLOT_MINUTES}-minute time slots")
    results = analysis_results()
    show_kpis(results)
    left, right = st.columns(2)
    left.pyplot(comparison_chart(results, "Electricity Cost"), use_container_width=True)
    right.pyplot(comparison_chart(results, "Peak Shaving"), use_container_width=True)
    ga, ma = (int(results.loc[m, "constraint_violations"]) for m in ("GA", "MA"))
    if ga == 0 and ma == 0:
        st.success("Constraint Violations — GA: 0 | MA: 0")
    else:
        st.warning(f"Constraint Violations — GA: {ga} | MA: {ma}")
    


def dataset_page():
    st.title("Dataset & Constraints")
    datasets = read_datasets()
    for col, label, value in zip(st.columns(4), ["Appliances", "Simulation Days", "Slot Duration", "Total Slots"],
                                 [len(datasets["Appliances"]), NUMBER_OF_DAYS, f"{SLOT_MINUTES} min", NUMBER_OF_DAYS * SLOTS_PER_DAY]):
        col.metric(label, value)
    tabs = st.tabs(list(datasets) + ["Constraints"])
    with tabs[0]:
        st.subheader("Appliance profiles and allowed start times")
        st.dataframe(datasets["Appliances"], hide_index=True, use_container_width=True)
    columns = {"Electricity Prices": "electricity_price_per_kwh", "Solar Generation": "solar_generation_watts", "Base Load": "base_load_watts"}
    for tab, name in zip(tabs[1:4], columns):
        with tab:
            data = datasets[name].copy()
            data["timestamp"] = pd.to_datetime(data["timestamp"])
            dates = sorted(data["timestamp"].dt.date.unique())
            date = st.selectbox("Select Date", dates, key=f"dataset_{name}")
            daily = data[data["timestamp"].dt.date == date]
            st.line_chart(daily.set_index("timestamp")[columns[name]])
            st.dataframe(daily, hide_index=True, use_container_width=True)
    with tabs[4]:
        st.subheader("Hard constraints")
        for title, description in [
            ("Allowed start times", "Every gene must remain within its permitted start-time bounds."),
            ("Daily time boundary", f"Each segment must fit inside the {SLOTS_PER_DAY} slots of its day."),
            ("No overlapping segments", "Segments of the same appliance cannot overlap."),
            ("Required operating duration", "Each appliance must run for its required number of slots."),
            ("Non-interruptible appliances", "These appliances must run in one continuous block.")]:
            st.markdown(f"**{title}** — {description}")
        st.subheader("Fitness Function")
        st.latex(r"F=\frac{1}{1+w_c(C/C_0)+w_p(P/P_0)+\lambda V}")
        st.caption("C and P are candidate cost and peak grid load; C₀ and P₀ are the original values. V is the number of constraint violations. Higher fitness is better. Formula assumes positive original cost and peak.")
        a, b, c = st.columns(3)
        a.metric("Cost Weight", COST_WEIGHT)
        b.metric("Peak Weight", PEAK_WEIGHT)
        c.metric("Penalty per Violation", CONSTRAINT_PENALTY)


def fitness_results(day, root=ROOT):
    
    daily = analysis_results(day, root)
    original = daily.loc["Original"]
    cost = original["total_cost"]
    peak = original["peak_grid_import_kw"]
    cost_ratio = 1.0 if cost > 0 else cost
    peak_ratio = 1.0 if peak > 0 else peak
    objective = COST_WEIGHT * cost_ratio + PEAK_WEIGHT * peak_ratio
    penalty = CONSTRAINT_PENALTY * original["constraint_violations"]
    rows = {"Original": {"electricity_cost": cost, "peak_grid_import_kw": peak,
                         "cost_ratio": cost_ratio, "peak_ratio": peak_ratio,
                         "objective": objective, "constraint_violations": original["constraint_violations"],
                         "penalty": penalty, "total_objective": objective + penalty,
                         "fitness": 1 / (1 + objective + penalty)}}
    for method, prefix in [("GA", "ga"), ("MA", "ma")]:
        saved = pd.read_csv(root / f"{prefix}_fitness_results.csv")
        selected = saved.loc[saved["day"] == day]
        if len(selected) != 1:
            raise ValueError(f"Expected one fitness result for {method}, day {day}.")
        rows[method] = selected.iloc[0][list(rows["Original"])].to_dict()
    return pd.DataFrame(rows).T.astype(float)


def fitness_chart(results, day):
    fig = Figure(figsize=(7.3, 4), layout="constrained")
    ax = fig.add_subplot(111)
    bars = ax.bar(results.index, results["fitness"], color=COLORS, width=0.55)
    ax.bar_label(bars, fmt="%.4f", padding=4)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Fitness Score")
    ax.set_title(f"Fitness Score Comparison — Day {day}")
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    return fig


def fitness_page():
    st.title("Fitness & Evaluation")
    st.subheader("Fitness Configuration")
    a, b, c = st.columns(3)
    a.metric("Cost Weight", COST_WEIGHT)
    b.metric("Peak Weight", PEAK_WEIGHT)
    c.metric("Penalty per Violation", CONSTRAINT_PENALTY)
    st.subheader("Fitness Function")
    st.latex(r"C_{ratio}=C/C_0,\quad P_{ratio}=P/P_0")
    st.latex(r"Objective=w_c C_{ratio}+w_p P_{ratio}")
    st.latex(r"Penalty=\lambda\times Violations")
    st.latex(r"Fitness=\frac{1}{1+Objective+Penalty}")
    st.caption("Ratios use original daily cost and peak; if either baseline is non-positive, the evaluator uses the candidate value instead. Higher fitness is better.")
    st.subheader("Evaluation Results")
    day = st.selectbox("Select Day", list(range(1, NUMBER_OF_DAYS + 1)), key="fitness_day")
    results = fitness_results(day)
    labels = {"electricity_cost": "Electricity Cost", "peak_grid_import_kw": "Peak Grid Load (kW)",
              "cost_ratio": "Cost Ratio", "peak_ratio": "Peak Ratio", "objective": "Objective",
              "constraint_violations": "Constraint Violations", "penalty": "Penalty",
              "total_objective": "Total Objective", "fitness": "Fitness Score"}
    table = results.rename(columns=labels).T
    st.dataframe(table.style.format("{:.4f}"), use_container_width=True)
    st.pyplot(fitness_chart(results, day), use_container_width=True)
    if (results["constraint_violations"] == 0).all():
        st.success("All schedules satisfy the constraints for the selected day.")
    else:
        st.warning("One or more schedules have constraint violations.")



def algorithm_fitness_history(method, root=ROOT):
    return pd.read_csv(root / f"{method.lower()}_fitness_results.csv").sort_values("day")


def algorithms_page():
    st.title("Algorithms Overview")
    ga_tab, ma_tab = st.tabs(["Genetic Algorithm (GA)", "Memetic Algorithm (MA)"])
    results = analysis_results()
    for tab, method in [(ga_tab, "GA"), (ma_tab, "MA")]:
        with tab:
            is_ma = method == "MA"
            st.subheader("Evolutionary Optimization" + (" + Local Search" if is_ma else ""))
            row = results.loc[method]
            for col, label, value in zip(st.columns(4),
                ["Cost Savings", "Peak Reduction", "Monthly Cost", "Violations"],
                [f"{row['cost_savings_percent']:.2f}%", f"{row['peak_reduction_percent']:.2f}%",
                 f"{row['total_cost']:.2f}", int(row['constraint_violations'])]):
                col.metric(label, value)
            st.subheader("Algorithm Configuration")
            params = {"Population Size": 40, "Generations": 50, "Crossover Rate": 0.8,
                      "Mutation Rate": 0.1, "Tournament Size": 3, "Elitism": 2,
                      "Random Seed": "42 + zero-based day index"}
            if is_ma:
                params.update({"Local Search Rate": 0.1, "Local Search Iterations": 3,
                               "Local Search Target": "children"})
            st.table(pd.DataFrame({"Parameter": list(params), "Runner setting": [str(v) for v in params.values()]}))
            st.subheader("Chromosome Representation")
            st.write(f"Each integer gene represents the starting time slot of an appliance operating segment. Each slot is {SLOT_MINUTES} minutes; genes are initialized within their allowed bounds.")
            st.subheader("Algorithm Workflow")
            steps = ["Initialize population (include original schedule)", "Evaluate fitness", "Retain elites and select parents by tournament", "Uniform crossover", "Random mutation within gene bounds"]
            if is_ma:
                steps.append("Local search on selected children (default mode)")
            steps += ["Create next generation and repeat", "Return best chromosome and fitness"]
            st.markdown(" → ".join(steps))
            if is_ma:
                st.info("Local search tries moving one gene forward or backward by one slot, within its bounds. It accepts the best strictly improving neighbor and stops when no improvement exists or the iteration limit is reached. The implementation also evaluates the final population.")
            st.subheader("Saved Daily Final Fitness")
            saved = algorithm_fitness_history(method)
            fig = Figure(figsize=(9, 3.8), layout="constrained")
            ax = fig.add_subplot(111)
            ax.plot(saved["day"], saved["fitness"], marker="o", markersize=3, color=COLORS[2 if is_ma else 1])
            ax.set(xlabel="Simulation Day", ylabel="Final Fitness Score", title=f"{method} — Final Fitness per Day")
            ax.grid(alpha=0.2)
            st.pyplot(fig, use_container_width=True)
            st.dataframe(saved, hide_index=True, use_container_width=True)
            if is_ma:
                st.subheader("Fitness Convergence by Generation")
                history_path = ROOT / "ma_history.csv"
                if history_path.exists():
                    history = pd.read_csv(history_path)
                    day = st.selectbox("Select Day", sorted(history["day"].unique()), key="ma_history_day")
                    daily = history.loc[history["day"] == day].sort_values("generation")
                    fig = Figure(figsize=(9, 3.8), layout="constrained")
                    ax = fig.add_subplot(111)
                    ax.plot(daily["generation"], daily["best_fitness"], color=COLORS[2], linewidth=2)
                    ax.set(xlabel="Generation", ylabel="Best Fitness", title=f"MA Convergence — Day {day}")
                    ax.grid(alpha=0.2)
                    st.pyplot(fig, use_container_width=True)
                    st.caption("Saved generation history; the final population check may improve the final score beyond the last recorded generation.")
                else:
                    st.warning("MA generation history file is missing.")
            
            st.subheader("Monthly Optimization Results")
            st.dataframe(comparison_table(results.loc[["Original", method]]).style.format("{:.2f}", na_rep="N/A"), use_container_width=True)


def visualization_page():
    st.title("Visualization")
    day = select_period("visualization")
    graph = st.selectbox("Select Graph", GRAPH_NAMES)
    results = analysis_results(day)
    fig = load_profile(day) if graph == "Load Profile" else comparison_chart(results, graph, day)
    st.pyplot(fig, use_container_width=True)
    if graph == "Load Profile" and day is None:
        st.caption("The monthly chart displays each day's maximum grid load, not all half-hour samples. Select Daily for the full 24-hour profile.")
    name = f"{graph.lower().replace(' ', '_')}_{'monthly' if day is None else f'day_{day}'}"
    st.download_button("Export Graph (PNG)", figure_bytes(fig), f"{name}.png", mime="image/png")


def comparison_page():
    st.title("Comparison & Results")
    day = select_period("comparison")
    st.caption(period_label(day))
    results = analysis_results(day)
    show_kpis(results)
    show_table(results)
    st.pyplot(comparison_chart(results, "Efficiency", day), use_container_width=True)
    st.latex(r"\mathrm{Reduction}(\%)=\frac{\mathrm{Original}-\mathrm{Optimized}}{\mathrm{Original}}\times100")
    st.caption("The same formula is applied to electricity cost and peak grid load. Negative savings indicate an increase. N/A means the original value was zero.")
    filename = "final_comparison_monthly.csv" if day is None else f"final_comparison_day_{day}.csv"
    st.download_button("Export Results (CSV)", results.to_csv(index_label="Method"), filename, mime="text/csv")


def main():
    st.set_page_config(page_title="Smart Grid Optimizer", layout="wide")
    st.sidebar.title("SMART GRID")
    st.sidebar.caption("GA & MA Energy Optimization")
    pages = {"Dashboard": dashboard_page, "Dataset & Constraints": dataset_page,
             "Fitness & Evaluation": fitness_page,
             "Algorithms Overview": algorithms_page,
             "Visualization": visualization_page, "Comparison & Results": comparison_page}
    page = st.sidebar.radio("Navigation", list(pages))
    st.sidebar.divider()
    
    try:
        pages[page]()
    except (FileNotFoundError, KeyError, ValueError, IndexError, OSError) as exc:
        st.error(f"Unable to load project data: {exc}")
        st.info("Place app.py next to main.py and keep the original dataset and result CSV files in the project folder.")


if __name__ == "__main__":
    main()
