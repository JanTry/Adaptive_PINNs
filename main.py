# from src.adaptations.adaptations_1D import (
#     DEAdaptation1D,
#     DensitySamplingAdaptation1D,
#     HMSAdaptation1D,
#     MiddlePointAdaptation1D,
#     NoAdaptation1D,
#     R3Adaptation1D,
#     RandomSearchWithSelection,
#     SelectionMethod,
# )
from src.adaptations.adaptations_2D import (
    DensitySamplingAdaptation2D,
    NoAdaptation2D,
    R3Adaptation2D,
    DEAdaptation2D,
)
from src.enums.problems import Problems1D, Problems2D

# from src.plots.plots_1D.plot_specific_run import plot_specific_run_1D
# from src.runners.adaptive_PINN_1D import train_PINN_1D
from src.runners.adaptive_PINN_2D import train_PINN_2D

PROBLEM_TYPES = [
    # Problems1D.DIFFUSION,
    # Problems1D.TAN_03,
    # Problems1D.P07_01,
    Problems2D.CONVECTION,
]

ADAPTATIONS = [
    # DEAdaptation1D(),
    # MiddlePointAdaptation1D(),
    # DensitySamplingAdaptation1D(),
    # RandomSearchWithSelection(selection_method=SelectionMethod.TOURNAMENT),
    # HMSAdaptation1D(),
    # NoAdaptation1D(),
    # R3Adaptation1D(),
    # RandomSearchWithSelection(selection_method=SelectionMethod.ROULETTE),
    # RandomSearchWithSelection(selection_method=SelectionMethod.TOURNAMENT),
    R3Adaptation2D(),
    NoAdaptation2D(),
    DensitySamplingAdaptation2D(),
    DEAdaptation2D(),
]
types_to_time = {}
NUM_RUNS = 1

for problem_type in PROBLEM_TYPES:
    for adaptation in ADAPTATIONS:
        for i in range(NUM_RUNS):
            # train_PINN_1D(i, problem_type=problem_type, adaptation=adaptation)
            # plot_specific_run_1D(
            #     run_id=i,
            #     problem_type=problem_type,
            #     adaptation=adaptation,
            #     plot_training_points=True,
            # )
            train_PINN_2D(i, problem_type=problem_type, adaptation=adaptation)
