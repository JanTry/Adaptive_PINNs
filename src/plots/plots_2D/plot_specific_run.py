import os

import matplotlib
import matplotlib.pyplot as plt
import src.params.params_2D as params
import torch
from matplotlib import rc
from src.adaptations.adaptations_2D.adaptation_interface import AdaptationInterface2D
from src.base.pinn_2D_core import f
from src.enums.problems import Problems2D
from src.helpers.factories import problem_factory_2D
from src.helpers.mesh_2D import get_mesh_2D

N_ITERS_FILE = "n_iters.pt"
TIME_FILE = "exec_time.pt"
CONVERGENCE_FILE = "convergence.pt"
PINN_FILE = "pinn.pt"
POINT_DATA_FILE = "point_data.pt"


def plot_specific_run_2D(
    run_id: int,
    problem_type: Problems2D,
    adaptation: AdaptationInterface2D,
    tolerance: float = params.TOLERANCE,
    learning_rate: float = params.LEARNING_RATE,
    layers: int = params.LAYERS,
    neurons: int = params.NEURONS,
    epochs: int = params.NUMBER_EPOCHS,
    max_points: int = params.MAX_POINTS_NUMBER,
    plot_training_points: bool = False,
):
    device = "cpu"
    matplotlib.rcParams.update({"font.size": 15})

    path = os.path.join(
        "results_2D",
        problem_type.value,
        str(adaptation),
        f"L{layers}_N{neurons}_" f"P{max_points}_E{epochs}",
        f"LR{learning_rate}_TOL{tolerance}",
        str(run_id),
    )

    problem = problem_factory_2D(problem_type)
    x_range, y_range = problem.get_range()
    # Get the problem's device for exact solution computation
    problem_device = x_range.device

    plt.rcParams["figure.dpi"] = 150
    rc("animation", html="html5")

    nn_approximator = torch.load(os.path.join(path, PINN_FILE), weights_only=False)
    convergence_data = torch.load(os.path.join(path, CONVERGENCE_FILE), weights_only=False)
    n_iters = torch.load(os.path.join(path, N_ITERS_FILE), weights_only=False)
    exec_time = torch.load(os.path.join(path, TIME_FILE), weights_only=False)

    os.makedirs(os.path.join(path, "plots", "iterations"), exist_ok=True)

    # Plot points for each iteration
    if plot_training_points:
        point_data = torch.load(os.path.join(path, POINT_DATA_FILE), weights_only=False)
        for i, p in enumerate(point_data):
            fig, ax = plt.subplots()
            data = p.cpu().detach().numpy()
            ax.scatter(data[:, 0], data[:, 1], s=1)
            ax.set_title(f"Points distribution iteration {i}")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            fig.savefig(os.path.join(path, "plots", "iterations", f"iteration_{i}"))
            plt.close(fig)

    # Create dense mesh for plotting (use CPU ranges for mesh creation)
    mesh_size = 100
    n_points = mesh_size * mesh_size
    x_range_cpu = x_range.cpu()
    y_range_cpu = y_range.cpu()
    n_x, n_y = get_mesh_2D(x_range_cpu, y_range_cpu, n_points, device=device, requires_grad=True)

    # Compute PINN solution
    z_pinn = f(nn_approximator, n_x, n_y)

    # Reshape for plotting
    x_plot = n_x.cpu().detach().numpy().reshape(mesh_size, mesh_size)
    y_plot = n_y.cpu().detach().numpy().reshape(mesh_size, mesh_size)
    z_pinn_plot = z_pinn.cpu().detach().numpy().reshape(mesh_size, mesh_size)

    # Plot the PINN solution as surface
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(x_plot, y_plot, z_pinn_plot, cmap="viridis", edgecolor="none")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("u")
    ax.set_title(f"PINN solution,\n time = {exec_time:.2f}s, iterations = {n_iters}")
    fig.savefig(os.path.join(path, "plots", "pinn_solution_3d"))
    plt.close(fig)

    # Plot PINN solution as contour
    fig, ax = plt.subplots(figsize=(8, 6))
    contour = ax.contourf(x_plot, y_plot, z_pinn_plot, levels=50, cmap="viridis")
    plt.colorbar(contour, ax=ax, label="u")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"PINN solution (contour),\n time = {exec_time:.2f}s, iterations = {n_iters}")
    fig.savefig(os.path.join(path, "plots", "pinn_solution_contour"))
    plt.close(fig)

    # Try to compute exact solution (some problems don't have analytical solutions)
    # Move tensors to problem's device for exact solution computation
    has_exact_solution = True
    try:
        n_x_problem = n_x.to(problem_device)
        n_y_problem = n_y.to(problem_device)
        exact_z = problem.exact_solution(n_x_problem, n_y_problem)
        z_exact_plot = exact_z.cpu().detach().numpy().reshape(mesh_size, mesh_size)
    except NotImplementedError:
        has_exact_solution = False

    if has_exact_solution:
        # Plot exact solution as surface
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(x_plot, y_plot, z_exact_plot, cmap="viridis", edgecolor="none")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("u")
        ax.set_title(f"Exact solution,\n time = {exec_time:.2f}s, iterations = {n_iters}")
        fig.savefig(os.path.join(path, "plots", "exact_solution_3d"))
        plt.close(fig)

        # Plot exact solution as contour
        fig, ax = plt.subplots(figsize=(8, 6))
        contour = ax.contourf(x_plot, y_plot, z_exact_plot, levels=50, cmap="viridis")
        plt.colorbar(contour, ax=ax, label="u")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"Exact solution (contour),\n time = {exec_time:.2f}s, iterations = {n_iters}")
        fig.savefig(os.path.join(path, "plots", "exact_solution_contour"))
        plt.close(fig)

        # PINN and exact solutions comparison (side by side surface plots)
        fig = plt.figure(figsize=(16, 6))
        ax1 = fig.add_subplot(121, projection="3d")
        ax1.plot_surface(x_plot, y_plot, z_exact_plot, cmap="viridis", edgecolor="none")
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")
        ax1.set_zlabel("u")
        ax1.set_title("Exact solution")

        ax2 = fig.add_subplot(122, projection="3d")
        ax2.plot_surface(x_plot, y_plot, z_pinn_plot, cmap="viridis", edgecolor="none")
        ax2.set_xlabel("x")
        ax2.set_ylabel("y")
        ax2.set_zlabel("u")
        ax2.set_title("PINN solution")

        fig.suptitle(f"Solutions comparison,\n time = {exec_time:.2f}s, iterations = {n_iters}")
        fig.savefig(os.path.join(path, "plots", "solutions_comparison_3d"))
        plt.close(fig)

        # Plot error
        error = z_pinn - exact_z.cpu()
        error_plot = error.cpu().detach().numpy().reshape(mesh_size, mesh_size)

        # Error as surface plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(x_plot, y_plot, error_plot, cmap="coolwarm", edgecolor="none")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("Error")
        ax.set_title(f"Error: NN_u - exact_solution,\n time = {exec_time:.2f}s, iterations = {n_iters}")
        fig.savefig(os.path.join(path, "plots", "error_3d"))
        plt.close(fig)

        # Error as contour plot
        fig, ax = plt.subplots(figsize=(8, 6))
        contour = ax.contourf(x_plot, y_plot, error_plot, levels=50, cmap="coolwarm")
        plt.colorbar(contour, ax=ax, label="Error")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"Error: NN_u - exact_solution (contour),\n time = {exec_time:.2f}s, iterations = {n_iters}")
        fig.savefig(os.path.join(path, "plots", "error_contour"))
        plt.close(fig)

    # Plot convergence
    rc("ytick", labelsize=15)
    fig, ax = plt.subplots()
    ax.semilogy(convergence_data.cpu().detach().numpy())
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"Convergence,\n time = {exec_time:.2f}s, iterations = {n_iters}")
    fig.savefig(os.path.join(path, "plots", "convergence"))
    plt.close(fig)
