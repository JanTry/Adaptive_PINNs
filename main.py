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
    DensitySamplingAdaptation1D(),
    NoAdaptation1D(),
    R3Adaptation1D(),
]

PROBLEM_TYPES_2D = [
    Problems2D.CONVECTION,
    Problems2D.ALLEN_CAHN,
]

ADAPTATIONS_2D = [
    R3Adaptation2D(),
    DensitySamplingAdaptation2D(),
    DEAdaptation2D(),
    NoAdaptation2D(),
]

NUM_RUNS_1D = 30
NUM_RUNS_2D = 5

types_to_time = {}

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
