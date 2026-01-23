import os

import pandas as pd
import src.params.params_1D as params_1D
import src.params.params_2D as params_2D
import torch
from src.adaptations.adaptations_1D import (
    DEAdaptation1D,
    DensitySamplingAdaptation1D,
    # HMSAdaptation1D,
    MiddlePointAdaptation1D,
    NoAdaptation1D,
    R3Adaptation1D,
    RandomSearchWithSelection,
    SelectionMethod,
)
from src.adaptations.adaptations_1D.adaptation_interface import AdaptationInterface1D
from src.adaptations.adaptations_2D import (
    DEAdaptation2D,
    DensitySamplingAdaptation2D,
    NoAdaptation2D,
    R3Adaptation2D,
)
from src.adaptations.adaptations_2D.adaptation_interface import AdaptationInterface2D
from src.enums.problems import Problems1D, Problems2D
from src.plots.plots_1D.plot_specific_run import N_ITERS_FILE, TIME_FILE

ALL_ADAPTATIONS = [
    NoAdaptation1D(),
    MiddlePointAdaptation1D(),
    DensitySamplingAdaptation1D(),
    R3Adaptation1D(),
    # HMSAdaptation1D(),
    DEAdaptation1D(),
    RandomSearchWithSelection(selection_method=SelectionMethod.ROULETTE),
    RandomSearchWithSelection(selection_method=SelectionMethod.TOURNAMENT),
]


def get_path_1D(problem_type: Problems1D, adaptation: AdaptationInterface1D) -> str:
    return os.path.join(
        "results_1D",
        problem_type.value,
        str(adaptation),
        f"L{params_1D.LAYERS}_N{params_1D.NEURONS}_"
        f"P{params_1D.NUM_MAX_POINTS}_E{params_1D.NUMBER_EPOCHS}",
        f"LR{params_1D.LEARNING_RATE}_TOL{params_1D.TOLERANCE}",
    )


def get_path_2D(problem_type: Problems2D, adaptation: AdaptationInterface2D) -> str:
    return os.path.join(
        "results_2D",
        problem_type.value,
        str(adaptation),
        f"L{params_2D.LAYERS}_N{params_2D.NEURONS}_"
        f"P{params_2D.MAX_POINTS_NUMBER}_E{params_2D.NUMBER_EPOCHS}",
        f"LR{params_2D.LEARNING_RATE}_TOL{params_2D.TOLERANCE}",
    )


ALL_ADAPTATIONS_2D = [
    NoAdaptation2D(),
    DensitySamplingAdaptation2D(),
    R3Adaptation2D(),
    DEAdaptation2D(),
]


def extract_df_from_results_1D(
    adaptations: list[AdaptationInterface1D] = ALL_ADAPTATIONS,
    problem_types: list[Problems1D] | None = None,
) -> pd.DataFrame:
    if problem_types is None:
        problem_types = list(Problems1D)

    all_rows = []

    for problem_type in problem_types:
        for adaptation in adaptations:
            try:
                path = get_path_1D(problem_type, adaptation)
                runs = sorted(
                    [dir for dir in os.listdir(path) if str.isdigit(dir)],
                    key=lambda x: int(x),
                )
                iterations = [
                    torch.load(os.path.join(path, run, N_ITERS_FILE), weights_only=False)
                    for run in runs
                ]
                times = [
                    torch.load(os.path.join(path, run, TIME_FILE), weights_only=False)
                    for run in runs
                ]
                all_rows.extend(
                    [
                        {
                            "run_id": run,
                            "iterations": iter,
                            "time": time,
                            "problem": problem_type.value,
                            "adaptation": str(adaptation),
                        }
                        for run, iter, time in zip(runs, iterations, times)
                    ]
                )
            except FileNotFoundError:
                pass
    return pd.DataFrame(all_rows)


def extract_df_from_results_2D(
    adaptations: list[AdaptationInterface2D] = ALL_ADAPTATIONS_2D,
    problem_types: list[Problems2D] | None = None,
) -> pd.DataFrame:
    if problem_types is None:
        problem_types = list(Problems2D)

    all_rows = []

    for problem_type in problem_types:
        for adaptation in adaptations:
            try:
                path = get_path_2D(problem_type, adaptation)
                runs = sorted(
                    [dir for dir in os.listdir(path) if str.isdigit(dir)],
                    key=lambda x: int(x),
                )
                iterations = [
                    torch.load(os.path.join(path, run, N_ITERS_FILE), weights_only=False)
                    for run in runs
                ]
                times = [
                    torch.load(os.path.join(path, run, TIME_FILE), weights_only=False)
                    for run in runs
                ]
                all_rows.extend(
                    [
                        {
                            "run_id": run,
                            "iterations": iter,
                            "time": time,
                            "problem": problem_type.value,
                            "adaptation": str(adaptation),
                        }
                        for run, iter, time in zip(runs, iterations, times)
                    ]
                )
            except FileNotFoundError:
                pass
    return pd.DataFrame(all_rows)
