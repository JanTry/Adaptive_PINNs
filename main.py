import os

import plotly.express as px
import plotly.graph_objects as go

from src.adaptations.adaptations_1D import (
    DEAdaptation1D,
    DensitySamplingAdaptation1D,
    NoAdaptation1D,
    R3Adaptation1D,
)
from src.adaptations.adaptations_2D import (
    DensitySamplingAdaptation2D,
    NoAdaptation2D,
    R3Adaptation2D,
    DEAdaptation2D,
)
from src.enums.problems import Problems1D, Problems2D
from src.helpers.analysis import extract_df_from_results_1D, extract_df_from_results_2D

from src.plots.plots_1D.plot_specific_run import plot_specific_run_1D
from src.plots.plots_2D.plot_specific_run import plot_specific_run_2D

from src.runners.adaptive_PINN_1D import train_PINN_1D
from src.runners.adaptive_PINN_2D import train_PINN_2D

PROBLEM_TYPES_1D = [
    Problems1D.DIFFUSION,
    Problems1D.TAN_03,
    Problems1D.P07_01,
]

ADAPTATIONS_1D = [
    DEAdaptation1D(),
    NoAdaptation1D(),
    R3Adaptation1D(),
    DensitySamplingAdaptation1D(),
]

PROBLEM_TYPES_2D = [
    Problems2D.CONVECTION,
    Problems2D.ALLEN_CAHN,
]

ADAPTATIONS_2D = [
    R3Adaptation2D(),
    DEAdaptation2D(),
    NoAdaptation2D(),
    DensitySamplingAdaptation2D(),
]

NUM_RUNS_1D = 30
NUM_RUNS_2D = 5

for problem_type in PROBLEM_TYPES_2D:
    for adaptation in ADAPTATIONS_2D:
        for i in range(NUM_RUNS_2D):
            train_PINN_2D(i, problem_type=problem_type, adaptation=adaptation)
            plot_specific_run_2D(
                run_id=i,
                problem_type=problem_type,
                adaptation=adaptation,
                plot_training_points=True,
            )

for problem_type in PROBLEM_TYPES_1D:
    for adaptation in ADAPTATIONS_1D:
        for i in range(NUM_RUNS_1D):
            train_PINN_1D(i, problem_type=problem_type, adaptation=adaptation)
            plot_specific_run_1D(
                run_id=i,
                problem_type=problem_type,
                adaptation=adaptation,
                plot_training_points=True,
            )


# =============================================================================
# Results Analysis and Visualization
# =============================================================================

RESULTS_DIR = "results_analysis"
os.makedirs(RESULTS_DIR, exist_ok=True)


def create_boxplots(df, metric, title_suffix, filename_suffix, results_dir):
    """Create boxplots for a given metric across all problems."""
    if df.empty:
        print(f"No data available for {title_suffix}")
        return

    problems = df["problem"].unique()

    for problem in problems:
        problem_df = df[df["problem"] == problem]

        fig = px.box(
            problem_df,
            x="adaptation",
            y=metric,
            color="adaptation",
            title=f"{metric.capitalize()} Comparison - {problem}",
        )
        fig.update_layout(
            width=1200,
            height=800,
            xaxis_title="Adaptation Method",
            yaxis_title=metric.capitalize(),
        )
        fig.write_html(os.path.join(results_dir, f"{problem}_{filename_suffix}.html"))
        fig.write_image(os.path.join(results_dir, f"{problem}_{filename_suffix}.png"))

    # Combined plot for all problems
    fig = px.box(
        df,
        x="adaptation",
        y=metric,
        color="adaptation",
        facet_col="problem",
        title=f"{metric.capitalize()} Comparison - All Problems {title_suffix}",
    )
    fig.update_layout(
        width=1600,
        height=800,
    )
    fig.write_html(os.path.join(results_dir, f"all_problems_{filename_suffix}.html"))
    fig.write_image(os.path.join(results_dir, f"all_problems_{filename_suffix}.png"))


def create_summary_statistics(df, results_dir, suffix):
    """Create and save summary statistics table."""
    if df.empty:
        print(f"No data available for summary statistics {suffix}")
        return

    summary = df.groupby(["problem", "adaptation"]).agg(
        iterations_mean=("iterations", "mean"),
        iterations_std=("iterations", "std"),
        iterations_median=("iterations", "median"),
        iterations_min=("iterations", "min"),
        iterations_max=("iterations", "max"),
        time_mean=("time", "mean"),
        time_std=("time", "std"),
        time_median=("time", "median"),
        num_runs=("run_id", "count"),
    ).round(2)

    summary.to_csv(os.path.join(results_dir, f"summary_statistics_{suffix}.csv"))

    # Create a styled HTML table
    summary_reset = summary.reset_index()
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=list(summary_reset.columns),
            fill_color="paleturquoise",
            align="left",
        ),
        cells=dict(
            values=[summary_reset[col] for col in summary_reset.columns],
            fill_color="lavender",
            align="left",
        ),
    )])
    fig.update_layout(
        title=f"Summary Statistics {suffix}",
        width=1400,
        height=600,
    )
    fig.write_html(os.path.join(results_dir, f"summary_table_{suffix}.html"))

    return summary


def create_violin_plots(df, metric, title_suffix, filename_suffix, results_dir):
    """Create violin plots for better distribution visualization."""
    if df.empty:
        return

    fig = px.violin(
        df,
        x="adaptation",
        y=metric,
        color="adaptation",
        facet_col="problem",
        box=True,
        points="all",
        title=f"{metric.capitalize()} Distribution - {title_suffix}",
    )
    fig.update_layout(
        width=1600,
        height=800,
    )
    fig.write_html(os.path.join(results_dir, f"violin_{filename_suffix}.html"))
    fig.write_image(os.path.join(results_dir, f"violin_{filename_suffix}.png"))


def create_scatter_time_vs_iterations(df, title_suffix, filename_suffix, results_dir):
    """Create scatter plot of time vs iterations."""
    if df.empty:
        return

    fig = px.scatter(
        df,
        x="iterations",
        y="time",
        color="adaptation",
        facet_col="problem",
        title=f"Time vs Iterations - {title_suffix}",
        hover_data=["run_id"],
    )
    fig.update_layout(
        width=1600,
        height=800,
    )
    fig.write_html(os.path.join(results_dir, f"scatter_time_vs_iter_{filename_suffix}.html"))
    fig.write_image(os.path.join(results_dir, f"scatter_time_vs_iter_{filename_suffix}.png"))


def create_bar_chart_means(df, results_dir, suffix):
    """Create bar chart comparing mean iterations and time."""
    if df.empty:
        return

    summary = df.groupby(["problem", "adaptation"]).agg(
        iterations_mean=("iterations", "mean"),
        iterations_std=("iterations", "std"),
        time_mean=("time", "mean"),
        time_std=("time", "std"),
    ).reset_index()

    # Iterations bar chart
    fig = px.bar(
        summary,
        x="problem",
        y="iterations_mean",
        color="adaptation",
        barmode="group",
        error_y="iterations_std",
        title=f"Mean Iterations by Problem and Adaptation - {suffix}",
    )
    fig.update_layout(
        width=1200,
        height=800,
        xaxis_title="Problem",
        yaxis_title="Mean Iterations",
    )
    fig.write_html(os.path.join(results_dir, f"bar_iterations_{suffix}.html"))
    fig.write_image(os.path.join(results_dir, f"bar_iterations_{suffix}.png"))

    # Time bar chart
    fig = px.bar(
        summary,
        x="problem",
        y="time_mean",
        color="adaptation",
        barmode="group",
        error_y="time_std",
        title=f"Mean Execution Time by Problem and Adaptation - {suffix}",
    )
    fig.update_layout(
        width=1200,
        height=800,
        xaxis_title="Problem",
        yaxis_title="Mean Time (s)",
    )
    fig.write_html(os.path.join(results_dir, f"bar_time_{suffix}.html"))
    fig.write_image(os.path.join(results_dir, f"bar_time_{suffix}.png"))


# Extract and analyze 2D results
print("=" * 60)
print("Extracting and analyzing 2D results...")
print("=" * 60)

results_2d_df = extract_df_from_results_2D(
    adaptations=ADAPTATIONS_2D,
    problem_types=PROBLEM_TYPES_2D,
)

if not results_2d_df.empty:
    results_2d_dir = os.path.join(RESULTS_DIR, "2D")
    os.makedirs(results_2d_dir, exist_ok=True)

    # Save raw data
    results_2d_df.to_csv(os.path.join(results_2d_dir, "raw_results_2D.csv"), index=False)

    # Create visualizations
    create_boxplots(results_2d_df, "iterations", "(2D)", "iterations_2D", results_2d_dir)
    create_boxplots(results_2d_df, "time", "(2D)", "time_2D", results_2d_dir)
    create_violin_plots(results_2d_df, "iterations", "2D Problems", "iterations_2D", results_2d_dir)
    create_violin_plots(results_2d_df, "time", "2D Problems", "time_2D", results_2d_dir)
    create_scatter_time_vs_iterations(results_2d_df, "2D Problems", "2D", results_2d_dir)
    create_bar_chart_means(results_2d_df, results_2d_dir, "2D")
    create_summary_statistics(results_2d_df, results_2d_dir, "2D")

    print(f"2D results saved to {results_2d_dir}")
    print(f"Number of 2D runs analyzed: {len(results_2d_df)}")
else:
    print("No 2D results found.")

# Extract and analyze 1D results
print("=" * 60)
print("Extracting and analyzing 1D results...")
print("=" * 60)

results_1d_df = extract_df_from_results_1D(
    adaptations=ADAPTATIONS_1D,
    problem_types=PROBLEM_TYPES_1D,
)

if not results_1d_df.empty:
    results_1d_dir = os.path.join(RESULTS_DIR, "1D")
    os.makedirs(results_1d_dir, exist_ok=True)

    # Save raw data
    results_1d_df.to_csv(os.path.join(results_1d_dir, "raw_results_1D.csv"), index=False)

    # Create visualizations
    create_boxplots(results_1d_df, "iterations", "(1D)", "iterations_1D", results_1d_dir)
    create_boxplots(results_1d_df, "time", "(1D)", "time_1D", results_1d_dir)
    create_violin_plots(results_1d_df, "iterations", "1D Problems", "iterations_1D", results_1d_dir)
    create_violin_plots(results_1d_df, "time", "1D Problems", "time_1D", results_1d_dir)
    create_scatter_time_vs_iterations(results_1d_df, "1D Problems", "1D", results_1d_dir)
    create_bar_chart_means(results_1d_df, results_1d_dir, "1D")
    create_summary_statistics(results_1d_df, results_1d_dir, "1D")

    print(f"1D results saved to {results_1d_dir}")
    print(f"Number of 1D runs analyzed: {len(results_1d_df)}")
else:
    print("No 1D results found.")

print("=" * 60)
print("Analysis complete!")
print("=" * 60)
