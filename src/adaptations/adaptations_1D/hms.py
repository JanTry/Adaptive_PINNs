# from typing import Callable

# import numpy as np
# import src.params.params_1D as params
# import torch
# from pyhms import (
#     SEA,
#     DontStop,
#     EALevelConfig,
#     EvalCutoffProblem,
#     FunctionProblem,
#     MetaepochLimit,
#     SingularProblemEvalLimitReached,
#     hms,
# )
# from pyhms.core.population import Population
# from pyhms.demes.single_pop_eas.multiwinner import CCGreedyPolicy, MultiwinnerSelection, UtilityFunction
# from pyhms.sprout import get_simple_sprout
# from src.adaptations.adaptations_1D.adaptation_interface import AdaptationInterface1D

# DEFAULT_EVAL_CUTOFF = 5000
# DEFAULT_N_DEMES = 10
# RANDOM_SEED = 42


# def run_hms(
#     x_range: tuple[float, float],
#     loss_function: Callable,
#     number_of_points: int,
#     eval_cutoff: int,
# ) -> np.ndarray:
#     def objective_function(x: np.ndarray) -> float:
#         x_tensor = torch.tensor(x).float().reshape(-1, 1).requires_grad_(True).to(params.DEVICE)
#         return loss_function(x_tensor).abs().detach().to("cpu").numpy().reshape(-1).item()

#     function_problem = FunctionProblem(objective_function, maximize=True, bounds=np.array([x_range]), use_cache=True)
#     problem_with_cutoff = EvalCutoffProblem(function_problem, eval_cutoff)
#     config = [
#         EALevelConfig(
#             ea_class=SEA,
#             generations=1,
#             problem=problem_with_cutoff,
#             pop_size=number_of_points // 2,
#             mutation_std=0.5,
#             lsc=DontStop(),
#         ),
#         EALevelConfig(
#             ea_class=SEA,
#             generations=1,
#             problem=problem_with_cutoff,
#             pop_size=number_of_points // 4,
#             mutation_std=0.25,
#             lsc=MetaepochLimit(5),
#         ),
#     ]
#     global_stop_condition = SingularProblemEvalLimitReached(eval_cutoff)
#     sprout_condition = get_simple_sprout(far_enough=(x_range[1] - x_range[0] / DEFAULT_N_DEMES))
#     hms_tree = hms(
#         config,
#         global_stop_condition,
#         sprout_condition,
#         {"random_seed": RANDOM_SEED},
#     )
#     last_population_individuals = []
#     for idx, deme in hms_tree.all_demes:
#         last_population_individuals.extend(deme.current_population)
#     selector = MultiwinnerSelection(
#         utility_function=UtilityFunction(
#             distance=lambda x, y: np.sum(np.abs(x - y)),
#             gamma=lambda x: x**6,
#             delta=lambda x: 1 / x,
#         ),
#         voting_scheme=CCGreedyPolicy(),
#         k=number_of_points,
#     )
#     selected_population = selector(Population.from_individuals(last_population_individuals))
#     return selected_population.genomes


# class HMSAdaptation1D(AdaptationInterface1D):
#     def __init__(self, eval_cutoff: int = DEFAULT_EVAL_CUTOFF) -> None:
#         self.eval_cutoff = eval_cutoff

#     def refine(self, loss_function: Callable, old_x: torch.Tensor):
#         self.validate_problem_details()
#         x_np = run_hms(
#             x_range=self.x_range,
#             loss_function=loss_function,
#             number_of_points=self.max_number_of_points,
#             eval_cutoff=self.eval_cutoff,
#         )
#         x = torch.tensor(x_np, device=old_x.device, dtype=old_x.dtype)
#         refined_x = torch.cat([old_x[0:1], x, old_x[-1:]]).sort()[0]
#         return refined_x.detach().clone().requires_grad_(True).to(old_x.device)

#     def __str__(self) -> str:
#         return "hms" if self.eval_cutoff == DEFAULT_EVAL_CUTOFF else f"hms_{self.eval_cutoff}"
