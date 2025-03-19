from typing import Callable, Tuple

import torch
from src.adaptations.adaptations_2D.adaptation_interface import AdaptationInterface2D
from src.helpers.separate_boundary_points_2D import separate_boundary_points_2D

DEFAULT_DE_MAX_ITERATIONS = 1
DEFAULT_DE_F = 0.8
DEFAULT_DE_CR = 0.9


def mirror_bounds(x: torch.Tensor, lower: float, upper: float) -> torch.Tensor:
    """
    Handles boundary constraints using mirror method.
    When a value exceeds the boundary, it is reflected back into the valid range.
    """
    range_size = upper - lower
    x_shifted = x - lower
    quotient = torch.floor(x_shifted / range_size)
    remainder = x_shifted % range_size
    mirrored = torch.where(quotient % 2 == 1, range_size - remainder, remainder)
    return mirrored + lower


class DEAdaptation2D(AdaptationInterface2D):
    """
    2D implementation of Differential Evolution (DE) adaptation technique.
    """

    def __init__(
        self,
        max_iterations: int = DEFAULT_DE_MAX_ITERATIONS,
        f: float = DEFAULT_DE_F,
        cr: float = DEFAULT_DE_CR,
    ) -> None:
        super().__init__()
        self.max_iterations = max_iterations
        self.f = f
        self.cr = cr
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

        # Separate boundary and interior points
        inner_x, inner_y, self.boundary_x, self.boundary_y = (
            separate_boundary_points_2D(old_x, old_y, self.x_range, self.y_range)
        )
        boundary_points = torch.cat((self.boundary_x, self.boundary_y), dim=1)

        # Combine x and y into a single tensor for interior points
        interior_points = torch.cat((inner_x, inner_y), dim=1)
        interior_points = interior_points.detach().clone().requires_grad_(True)

        population_size, number_of_dimensions = interior_points.shape

        for _ in range(self.max_iterations):
            # Generate random indices for DE mutation
            r1, r2, r3 = self.__generate_indices(
                population_size, interior_points.device
            )

            # DE mutation
            v = interior_points[r1] + self.f * (
                interior_points[r2] - interior_points[r3]
            )

            # Apply mirror bounds separately for x and y
            v[:, 0] = mirror_bounds(v[:, 0], self.x_range[0], self.x_range[1])
            v[:, 1] = mirror_bounds(v[:, 1], self.y_range[0], self.y_range[1])

            # DE crossover
            rand = torch.rand(
                population_size, number_of_dimensions, device=interior_points.device
            )
            mask = rand < self.cr
            u = torch.where(mask, v, interior_points)

            # Evaluate and select
            f_u = (
                loss_function(u[:, 0].reshape(-1, 1), u[:, 1].reshape(-1, 1))
                .abs()
                .view(-1)
            )
            f_x = (
                loss_function(
                    interior_points[:, 0].reshape(-1, 1),
                    interior_points[:, 1].reshape(-1, 1),
                )
                .abs()
                .view(-1)
            )
            improved = f_u >= f_x
            interior_points = torch.where(improved.unsqueeze(1), u, interior_points)

        # Combine boundary and interior points
        refined_points = torch.cat([boundary_points, interior_points])
        refined_points = refined_points.detach().clone().requires_grad_(True)
        return refined_points[:, 0].reshape(-1, 1), refined_points[:, 1].reshape(-1, 1)

    def __generate_indices(self, population_size: int, device: torch.device):
        """
        Generate random indices for DE mutation while ensuring they're all different.
        Memory efficient implementation that avoids creating large intermediate tensors.
        """
        r1 = torch.zeros(population_size, dtype=torch.long, device=device)
        r2 = torch.zeros(population_size, dtype=torch.long, device=device)
        r3 = torch.zeros(population_size, dtype=torch.long, device=device)

        for i in range(population_size):
            # Create a permutation of indices not including i
            perm = torch.randperm(population_size - 1, device=device)
            # Convert permutation to actual indices, skipping i
            perm = perm + (perm >= i).long()

            # Assign the first 3 indices
            r1[i] = perm[0]
            r2[i] = perm[1]
            r3[i] = perm[2]

        return r1, r2, r3

    def __str__(self) -> str:
        return (
            "de"
            if self.max_iterations == DEFAULT_DE_MAX_ITERATIONS
            else f"de_{self.max_iterations}"
        )
