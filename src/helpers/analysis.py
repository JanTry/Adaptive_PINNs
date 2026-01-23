import os

import pandas as pd
import src.params.params_1D as params
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
from src.enums.problems import Problems1D
from src.plots.plots_1D import N_ITERS_FILE, TIME_FILE

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


def get_path(problem_type: Problems1D, adaptation: AdaptationInterface1D) -> str:
    return os.path.join(
        "results_1D",
        problem_type.value,
        str(adaptation),
        f"L{params.LAYERS}_N{params.NEURONS}_" f"P{params.NUM_MAX_POINTS}_E{params.NUMBER_EPOCHS}",
        f"LR{params.LEARNING_RATE}_TOL{params.TOLERANCE}",
    )


def extract_df_from_results(
    adaptations: list[AdaptationInterface1D] = ALL_ADAPTATIONS,
) -> pd.DataFrame:
    all_rows = []

    for problem_type in Problems1D:
        for adaptation in adaptations:
            try:
                path = get_path(problem_type, adaptation)
                runs = sorted(
                    [dir for dir in os.listdir(path) if str.isdigit(dir)],
                    key=lambda x: int(x),
                )
                iterations = [torch.load(os.path.join(path, run, N_ITERS_FILE)) for run in runs]
                times = [torch.load(os.path.join(path, run, TIME_FILE)) for run in runs]
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
