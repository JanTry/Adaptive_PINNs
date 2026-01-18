from typing import Callable

import torch
import logging


def exit_criterion_1D(base_x: torch.Tensor, loss_fun: Callable, tol: float):
    x = base_x.detach().clone().requires_grad_(True)

    for x1, x2 in zip(x[:-1], x[1:]):
        int_x = torch.linspace(x1.item(), x2.item(), 20).requires_grad_(True).reshape(-1, 1).to(x.device)
        int_y = loss_fun(x=int_x) ** 2
        el_loss = torch.trapezoid(int_y, int_x, dim=0) / (x2 - x1)
        if el_loss > tol:
            return False

    return True


def exit_criterion_2D(base_x: torch.Tensor, base_y: torch.Tensor, loss_fun: Callable, tol: float):
    steps = 10
    x = base_x.detach().clone().requires_grad_(True)
    y = base_y.detach().clone().requires_grad_(True)

    for x1, x2 in zip(x[:-1], x[1:]):
        int_x = torch.linspace(x1.item(), x2.item(), steps).requires_grad_(True).to(x.device)
        for y1, y2 in zip(y[:-1], y[1:]):
            int_y = torch.linspace(y1.item(), y2.item(), steps).requires_grad_(True).to(y.device)
            grid_x, grid_y = torch.meshgrid(int_x, int_y, indexing="ij")
            linear_grid_x = torch.reshape(grid_x, [-1]).reshape(-1, 1).to(x.device)
            linear_grid_y = torch.reshape(grid_y, [-1]).reshape(-1, 1).to(x.device)

            linear_grid_z = loss_fun(x=linear_grid_x, y=linear_grid_y) ** 2
            grid_z = torch.reshape(linear_grid_z, shape=(steps, steps))

            el_loss = torch.trapezoid(torch.trapezoid(grid_z, int_y, dim=0), int_x, dim=0) / ((x2 - x1) * (y2 - y1))

            if el_loss > tol:
                return False

    return True


def exit_criterion_2D_new(
    base_x: torch.Tensor,
    base_y: torch.Tensor,
    loss_fun: Callable,
    tol: float,
    steps: int = 10,
    bx: int = 64,
    by: int = 64,
) -> bool:
    """
    Vectorized/batched version of exit_criterion_2D.

    Computes the mean squared loss over each cell using 2D trapezoidal integration.
    Returns True if all cells have mean loss <= tol, False otherwise.

    Args:
        base_x: 1D tensor of x grid boundaries
        base_y: 1D tensor of y grid boundaries
        loss_fun: Function(x, y) -> loss values
        tol: Tolerance threshold for mean loss per cell
        steps: Number of integration points per dimension
        bx: Batch size for x cells
        by: Batch size for y cells
    """
    with torch.no_grad():
        device, dtype = base_x.device, base_x.dtype

        # Handle both [N, 1] and [N] shapes
        x = base_x.flatten()
        y = base_y.flatten()

        # Cell boundaries
        x1, x2 = x[:-1], x[1:]
        y1, y2 = y[:-1], y[1:]
        Nx, Ny = x1.numel(), y1.numel()

        logging.warning(f"Exit criterion 2D (new) on {Nx + 1}x{Ny + 1} grid")

        t = torch.linspace(0, 1, steps, device=device, dtype=dtype)
        w = torch.ones(steps, device=device, dtype=dtype)
        w[0] = w[-1] = 0.5
        w /= steps - 1
        wX = w.view(1, 1, -1, 1)
        wY = w.view(1, 1, 1, -1)

        # Process in batches
        for ix in range(0, Nx, bx):
            ix_end = min(ix + bx, Nx)
            nbx = ix_end - ix

            # Extract batch of x cells
            xi1 = x1[ix:ix_end].view(-1, 1)  # [nbx, 1]
            xi2 = x2[ix:ix_end].view(-1, 1)  # [nbx, 1]
            xs = xi1 + (xi2 - xi1) * t.view(1, -1)  # [nbx, steps]
            xs = xs[:, None, :, None]  # [nbx, 1, steps, 1]

            for iy in range(0, Ny, by):
                iy_end = min(iy + by, Ny)
                nby = iy_end - iy

                # Extract batch of y cells
                yj1 = y1[iy:iy_end].view(1, -1, 1)  # [1, nby, 1]
                yj2 = y2[iy:iy_end].view(1, -1, 1)  # [1, nby, 1]
                ys = yj1 + (yj2 - yj1) * t.view(1, 1, -1)  # [1, nby, steps]
                ys = ys[:, :, None, :]  # [1, nby, 1, steps]

                # Broadcast to full grid: [nbx, nby, steps, steps]
                X = xs.expand(-1, nby, -1, steps)
                Y = ys.expand(nbx, -1, steps, -1)

                # Flatten for loss function call
                X_flat = X.reshape(-1, 1).requires_grad_(True)
                Y_flat = Y.reshape(-1, 1).requires_grad_(True)

                # Single batched call to loss function (needs gradients for dfdx/dfdy)
                with torch.enable_grad():
                    Z_flat = loss_fun(x=X_flat, y=Y_flat)
                Z = (Z_flat.detach() ** 2).view(nbx, nby, steps, steps)

                # Apply trapezoidal weights and sum
                cell_means = (Z * wX * wY).sum(dim=(-1, -2))  # [nbx, nby]
                if (cell_means > tol).any():
                    return False

        return True
