import math
import torch
from src.base.pinn_2D_core import PINN_2D, dfdx, dfdy, f
from src.helpers.problem_interface import ProblemInterface2D
from src.helpers.separate_boundary_points_2D import separate_boundary_points_2D
from src.params.params_2D import DEVICE


class ConvectionProblem2D(ProblemInterface2D):
    def __init__(
        self,
        beta: float = 1.0,
        lambda_ic: float = 1.0,
        lambda_per: float = 1.0,
    ):
        self.beta = torch.tensor(float(beta), device=DEVICE)
        self.x_range = torch.tensor([0.0, 2.0 * math.pi], device=DEVICE)
        self.y_range = torch.tensor([0.0, 1.0], device=DEVICE)
        self.lambda_ic = lambda_ic
        self.lambda_per = lambda_per

    def get_range(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x_range, self.y_range

    def exact_solution(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.sin(x - self.beta * y)

    def f_inner_loss(
        self, x: torch.Tensor, y: torch.Tensor, pinn: PINN_2D
    ) -> torch.Tensor:
        u_t = dfdy(pinn, x, y)
        u_x = dfdx(pinn, x, y)
        return u_t + self.beta * u_x

    def __initial_condition_loss(
        self, x: torch.Tensor, y: torch.Tensor, pinn: PINN_2D
    ) -> torch.Tensor:
        # enforce u(x,0) = sin(x)
        t0_mask = torch.isin(y, self.y_range[0])
        if not t0_mask.any():
            return y.new_zeros(())

        x0 = torch.masked_select(x, t0_mask).reshape(-1, 1)
        y0 = torch.masked_select(y, t0_mask).reshape(-1, 1)
        target = torch.sin(x0)
        return (f(pinn, x0, y0) - target).pow(2).mean()

    def __periodic_x_loss(
        self, x: torch.Tensor, y: torch.Tensor, pinn: PINN_2D
    ) -> torch.Tensor:
        # Pair left (x=0) and right (x=2π) boundary points by matching/sorting in t.
        left_mask = torch.isin(x, self.x_range[0])
        right_mask = torch.isin(x, self.x_range[1])
        if not (left_mask.any() and right_mask.any()):
            return x.new_zeros(())

        xl = torch.masked_select(x, left_mask).reshape(-1, 1)
        tl = torch.masked_select(y, left_mask).reshape(-1, 1)
        xr = torch.masked_select(x, right_mask).reshape(-1, 1)
        tr = torch.masked_select(y, right_mask).reshape(-1, 1)

        # sort by t and align counts
        il = torch.argsort(tl[:, 0])
        ir = torch.argsort(tr[:, 0])
        m = min(il.numel(), ir.numel())
        if m == 0:
            return x.new_zeros(())
        xl, tl = xl[il[:m]], tl[il[:m]]
        xr, tr = xr[ir[:m]], tr[ir[:m]]

        # value periodicity: u(0,t) = u(2π,t)
        u_left = f(pinn, xl, tl)
        u_right = f(pinn, xr, tr)
        loss_u = (u_left - u_right).pow(2).mean()

        return loss_u

    def __f_boundary_loss_mean(
        self, x: torch.Tensor, y: torch.Tensor, pinn: PINN_2D
    ) -> torch.Tensor:
        return self.lambda_ic * self.__initial_condition_loss(
            x, y, pinn
        ) + self.lambda_per * self.__periodic_x_loss(x, y, pinn)

    def compute_loss(
        self, x: torch.Tensor, y: torch.Tensor, pinn: PINN_2D
    ) -> torch.Tensor:
        inner_x, inner_y, boundary_x, boundary_y = separate_boundary_points_2D(
            x, y, self.x_range, self.y_range
        )
        interior = self.f_inner_loss(inner_x, inner_y, pinn).pow(2).mean()
        boundary = self.__f_boundary_loss_mean(boundary_x, boundary_y, pinn)
        return interior + boundary
