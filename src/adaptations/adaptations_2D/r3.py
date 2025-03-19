from typing import Callable, Tuple

import torch
from src.adaptations.adaptations_2D.adaptation_interface import AdaptationInterface2D
from src.helpers.separate_boundary_points_2D import separate_boundary_points_2D

DEFAULT_R3_MAX_ITERATIONS = 1


class R3Adaptation2D(AdaptationInterface2D):
    """
    2D implementation of Retain-Resample-Release (R3) adaptation technique.

    Daw, Arka, et al. "Mitigating propagation failures in physics-informed neural
    networks using retain-resample-release (r3) sampling."
    """

    def __init__(
        self,
        max_iterations: int = DEFAULT_R3_MAX_ITERATIONS,
    ) -> None:
        super().__init__()
        self.max_iterations = max_iterations
        self.boundary_x = None
        self.boundary_y = None

    def set_problem_details(
        self,
        x_range: torch.Tensor,
        y_range: torch.Tensor,
        base_points_x: torch.Tensor,
        base_points_y: torch.Tensor,
        max_number_of_points: int,
    ):
        self.device = base_points_x.device
        self.x_range = x_range
        self.y_range = y_range

        inner_x, inner_y, self.boundary_x, self.boundary_y = (
            separate_boundary_points_2D(base_points_x, base_points_y, x_range, y_range)
        )

        self.max_number_of_points = max_number_of_points
        self.max_number_of_interior_points = (
            max_number_of_points - list(self.boundary_x.shape)[0]
        )
        self.base_points_x = base_points_x
        self.base_points_y = base_points_y

    def refine(
        self, loss_function: Callable, old_x: torch.Tensor, old_y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self.validate_problem_details()

        inner_x, inner_y, _, _ = separate_boundary_points_2D(
            old_x, old_y, self.x_range, self.y_range
        )
        boundary_points = torch.cat((self.boundary_x, self.boundary_y), dim=1)

        points = torch.cat((inner_x, inner_y), dim=1)
        for _ in range(self.max_iterations):
            residual_function_values = (
                loss_function(points[:, 0].reshape(-1, 1), points[:, 1].reshape(-1, 1))
                .abs()
                .reshape(-1)
            )
            threshold = residual_function_values.mean()
            retained_points = points[residual_function_values > threshold]
            num_points_to_sample = points.shape[0] - retained_points.shape[0]

            if num_points_to_sample > 0:
                # Sample new points within the domain
                random_points = torch.empty(num_points_to_sample, 2).to(points.device)
                random_points[:, 0] = (
                    torch.empty(num_points_to_sample)
                    .uniform_(*self.x_range)
                    .to(points.device)
                )
                random_points[:, 1] = (
                    torch.empty(num_points_to_sample)
                    .uniform_(*self.y_range)
                    .to(points.device)
                )
                points = torch.cat([retained_points, random_points])
            else:
                points = retained_points

        refined_points = torch.cat([boundary_points, points])
        refined_points = refined_points.detach().clone().requires_grad_(True)
        return refined_points[:, 0].reshape(-1, 1), refined_points[:, 1].reshape(-1, 1)

    def __str__(self) -> str:
        return (
            "r3"
            if self.max_iterations == DEFAULT_R3_MAX_ITERATIONS
            else f"r3_{self.max_iterations}"
        )
