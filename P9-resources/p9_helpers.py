"""
P9 Helper Data — Gradient Descent (1D)

FIT1061 Introduction to Artificial Intelligence

This module provides loss functions and utility functions for P9.
You do NOT need to modify this file. Import it in your notebook:

    from p9_helpers import (f_quadratic, df_quadratic,
                            f_harder, df_harder,
                            f_harder_2d, grad_harder_2d,
                            HAND_TRACE_W0, HAND_TRACE_ALPHA,
                            HARDER_2D_GLOBAL_MIN, HARDER_2D_LOCAL_MIN,
                            plot_function, plot_descent,
                            plot_loss_curve, plot_lr_comparison,
                            plot_contour_2d, plot_contour_with_path_2d)

1D functions operate on scalars; 2D functions take a (2,) NumPy array.
"""

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Loss functions and their derivatives
# ---------------------------------------------------------------------------

def f_quadratic(w):
    """f(w) = w^2. Minimum at w = 0."""
    return w ** 2


def df_quadratic(w):
    """f'(w) = 2w. Derivative of f(w) = w^2."""
    return 2 * w


def f_harder(w):
    """f(w) = (w - 3)^2 + 0.5*sin(4w). A shifted quadratic with ripples.

    Minimum is near w = 3, but the sine term creates a bumpy landscape.
    This is harder than f_quadratic because the gradient is noisier.
    """
    return (w - 3) ** 2 + 0.5 * np.sin(4 * w)


def df_harder(w):
    """f'(w) = 2(w - 3) + 2*cos(4w). Derivative of f_harder."""
    return 2 * (w - 3) + 2 * np.cos(4 * w)


# ---------------------------------------------------------------------------
# Hand-Trace Parameters (for P9.1)
# ---------------------------------------------------------------------------
# f(w) = w^2, f'(w) = 2w
# Start at w = 4, learning rate alpha = 0.1
# Students compute 5 steps by hand.

HAND_TRACE_W0 = 4.0
HAND_TRACE_ALPHA = 0.1
HAND_TRACE_STEPS = 5

# Expected hand trace:
# Step 0: w=4.0000, f(w)=16.0000, f'(w)=8.0000, update=0.8000, new_w=3.2000
# Step 1: w=3.2000, f(w)=10.2400, f'(w)=6.4000, update=0.6400, new_w=2.5600
# Step 2: w=2.5600, f(w)= 6.5536, f'(w)=5.1200, update=0.5120, new_w=2.0480
# Step 3: w=2.0480, f(w)= 4.1943, f'(w)=4.0960, update=0.4096, new_w=1.6384
# Step 4: w=1.6384, f(w)= 2.6844, f'(w)=3.2768, update=0.3277, new_w=1.3107
# After 5 updates (history has 6 entries, indices 0-5):
# history[-1] = (1.3107, 1.7180)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_function(f, w_range=(-5, 5), title="f(w)", n_points=200):
    """Plot a 1D function over a range.

    Args:
        f: Function to plot (takes scalar or array, returns same).
        w_range: Tuple (min, max) for the w axis.
        title: Plot title.
        n_points: Number of points to plot.
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    w_vals = np.linspace(w_range[0], w_range[1], n_points)
    ax.plot(w_vals, f(w_vals), 'k-', linewidth=2)
    ax.set_xlabel('w', fontsize=12)
    ax.set_ylabel('f(w)', fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='gray', linewidth=0.5)
    ax.axvline(x=0, color='gray', linewidth=0.5)
    plt.tight_layout()
    plt.show()


def plot_descent(f, history, w_range=(-1, 5), title="Gradient Descent"):
    """Plot the function with gradient descent steps overlaid.

    Args:
        f: The loss function.
        history: List of (w, f(w)) tuples from gradient_descent.
        w_range: Tuple (min, max) for the w axis.
        title: Plot title.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: function with descent path
    w_vals = np.linspace(w_range[0], w_range[1], 200)
    ax1.plot(w_vals, f(w_vals), 'k-', linewidth=2, label='f(w)')

    ws = [h[0] for h in history]
    fs = [h[1] for h in history]
    ax1.plot(ws, fs, 'o-', color='#e74c3c', markersize=6,
             linewidth=1.5, label='GD steps', zorder=3)
    ax1.plot(ws[0], fs[0], 's', color='#2ecc71', markersize=10,
             zorder=4, label='Start')
    ax1.plot(ws[-1], fs[-1], '*', color='#3498db', markersize=14,
             zorder=4, label='End')
    ax1.set_xlabel('w', fontsize=12)
    ax1.set_ylabel('f(w)', fontsize=12)
    ax1.set_title(title, fontsize=13)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Right: loss over iterations
    ax2.plot(range(len(fs)), fs, 'o-', color='#e74c3c',
             markersize=4, linewidth=1.5)
    ax2.set_xlabel('Step', fontsize=12)
    ax2.set_ylabel('f(w)', fontsize=12)
    ax2.set_title('Loss Over Iterations', fontsize=13)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Print summary
    print(f"Start: w = {ws[0]:.4f}, f(w) = {fs[0]:.4f}")
    print(f"End:   w = {ws[-1]:.4f}, f(w) = {fs[-1]:.4f}")
    print(f"Steps: {len(history) - 1}")


def plot_loss_curve(history, alpha, title=None, ax=None):
    """Plot loss over iterations for one run.

    Args:
        history: List of (w, f(w)) tuples.
        alpha: Learning rate (for labelling).
        title: Optional title override.
        ax: Optional matplotlib Axes.
    """
    fs = [h[1] for h in history]
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(range(len(fs)), fs, 'o-', markersize=3, linewidth=1.5,
            label=f'alpha = {alpha}')
    ax.set_xlabel('Step', fontsize=12)
    ax.set_ylabel('f(w)', fontsize=12)
    if title:
        ax.set_title(title, fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)


def plot_lr_comparison(f, histories, alphas, title="Learning Rate Comparison"):
    """Plot loss curves for multiple learning rates side by side.

    Args:
        f: The loss function (unused, kept for API consistency).
        histories: List of trajectory lists, one per learning rate.
        alphas: List of learning rate values.
        title: Plot title.
    """
    colours = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6', '#f39c12']

    fig, ax = plt.subplots(1, 1, figsize=(9, 5))
    for hist, alpha, colour in zip(histories, alphas, colours):
        fs = [h[1] for h in hist]
        ax.plot(range(len(fs)), fs, 'o-', markersize=3, linewidth=2,
                color=colour, label=f'alpha = {alpha}')

    ax.set_xlabel('Step', fontsize=12)
    ax.set_ylabel('f(w)', fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def print_trace(history, max_rows=10):
    """Print a table of gradient descent steps.

    Args:
        history: List of (w, f(w)) tuples.
        max_rows: Maximum rows to print.
    """
    print(f"{'Step':>4}  {'w':>10}  {'f(w)':>10}")
    print("-" * 30)
    for i, (w, fw) in enumerate(history):
        if i >= max_rows and i < len(history) - 1:
            if i == max_rows:
                print(f"  ... ({len(history) - max_rows - 1} more steps)")
            continue
        print(f"{i:4d}  {w:10.4f}  {fw:10.4f}")


# =============================================================
# 2D extension (P9.3) — absorbed from old C4 in C-lite-revised 2026-05-21
# =============================================================
# These helpers extend P9 from 1D to 2D for the P9.3 sub-task.
# Each 2D function takes a NumPy array of shape (2,) and returns a scalar
# (or, for gradient functions, a (2,) array).

# --- Simple bowl: f(x1, x2) = (x1 - 2)^2 + (x2 - 1)^2 ---
# Minimum at (2, 1), f = 0. Symmetric — equally steep in both directions.

def f_bowl_2d(x):
    """f(x1, x2) = (x1-2)^2 + (x2-1)^2. Minimum at (2, 1)."""
    return (x[0] - 2) ** 2 + (x[1] - 1) ** 2


def grad_bowl_2d(x):
    """Gradient of f_bowl_2d: [2(x1-2), 2(x2-1)]."""
    return np.array([2 * (x[0] - 2), 2 * (x[1] - 1)])


# --- Elongated bowl: same minimum, 10x steeper in x1 direction.
# Causes zigzag — the "easy" 2D pathology to observe.

def f_elongated_2d(x):
    """f(x1, x2) = 10*(x1-2)^2 + (x2-1)^2. Minimum at (2, 1)."""
    return 10 * (x[0] - 2) ** 2 + (x[1] - 1) ** 2


def grad_elongated_2d(x):
    """Gradient of f_elongated_2d: [20(x1-2), 2(x2-1)]."""
    return np.array([20 * (x[0] - 2), 2 * (x[1] - 1)])


BOWL_MINIMUM_2D = np.array([2.0, 1.0])
START_POINT_2D = np.array([-3.0, -2.0])

# Hand-trace data for P9.3 partial-derivative step
# f(x1, x2) = (x1 - 2)^2 + (x2 - 1)^2  at  x = (-3, -2)
#   df/dx1 = 2(x1 - 2) = 2(-3 - 2) = -10
#   df/dx2 = 2(x2 - 1) = 2(-2 - 1) = -6
# With alpha = 0.1:
#   x_new = [-3 - 0.1*(-10), -2 - 0.1*(-6)] = [-2.0, -1.4]
HAND_TRACE_2D_START = np.array([-3.0, -2.0])
HAND_TRACE_2D_ALPHA = 0.1
HAND_TRACE_2D_EXPECTED_GRAD = np.array([-10.0, -6.0])
HAND_TRACE_2D_EXPECTED_NEXT = np.array([-2.0, -1.4])


# --- 2D plotting helpers ---

def plot_contour_2d(f_2d, title="Loss Surface", xlim=(-5, 6), ylim=(-4, 5),
                    minimum=None, zlim=None):
    """Plot colour-filled contours of a 2D function."""
    fig, ax = plt.subplots(1, 1, figsize=(9, 7))
    x1 = np.linspace(xlim[0], xlim[1], 200)
    x2 = np.linspace(ylim[0], ylim[1], 200)
    X1, X2 = np.meshgrid(x1, x2)
    Z = np.array([[f_2d(np.array([a, b])) for a in x1] for b in x2])
    zmin, zmax = zlim if zlim is not None else (Z.min(), Z.max())
    levels = np.linspace(zmin, zmax, 20)
    ax.contour(X1, X2, Z, levels=levels, colors='gray', alpha=0.4,
               linewidths=0.8)
    filled = ax.contourf(X1, X2, Z, levels=levels, cmap='viridis',
                         extend='both')
    fig.colorbar(filled, ax=ax, label='f(x1, x2)')
    if minimum is not None:
        ax.plot(minimum[0], minimum[1], '*', color='#f39c12',
                markersize=18, zorder=5, label='Minimum')
        ax.legend(fontsize=10)
    ax.set_xlabel('x1', fontsize=12)
    ax.set_ylabel('x2', fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.show()


def plot_contour_with_path_2d(f_2d, paths, labels, title,
                              xlim=(-5, 6), ylim=(-4, 5),
                              minimum=None, zlim=None):
    """Plot colour-filled 2D contours with optimisation paths."""
    colours = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f39c12']

    fig, ax = plt.subplots(1, 1, figsize=(9, 7))
    x1 = np.linspace(xlim[0], xlim[1], 200)
    x2 = np.linspace(ylim[0], ylim[1], 200)
    X1, X2 = np.meshgrid(x1, x2)
    Z = np.array([[f_2d(np.array([a, b])) for a in x1] for b in x2])
    zmin, zmax = zlim if zlim is not None else (Z.min(), Z.max())
    levels = np.linspace(zmin, zmax, 20)
    ax.contour(X1, X2, Z, levels=levels, colors='gray', alpha=0.4,
               linewidths=0.8)
    filled = ax.contourf(X1, X2, Z, levels=levels, cmap='viridis',
                         extend='both')
    fig.colorbar(filled, ax=ax, label='f(x1, x2)')

    if minimum is not None:
        ax.plot(minimum[0], minimum[1], '*', color='#f39c12',
                markersize=18, zorder=5, label='Minimum')

    for path, label, colour in zip(paths, labels, colours):
        xs = [p[0] for p in path]
        ys = [p[1] for p in path]
        ax.plot(xs, ys, 'o-', color=colour, markersize=4, linewidth=1.5,
                label=label, zorder=3)
        ax.plot(xs[0], ys[0], 's', color=colour, markersize=10, zorder=4)
        ax.plot(xs[-1], ys[-1], 'D', color=colour, markersize=8, zorder=4)

    ax.set_xlabel('x1', fontsize=12)
    ax.set_ylabel('x2', fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.show()


def _gradient_descent_2d_ref(grad_f, x0, alpha, steps):
    """Reference 2D gradient descent loop. TUTOR-ONLY (not imported by student notebook).

    From 2026-05-29: P9.3 promotes students to implement their own
    `gradient_descent_2d`. This reference remains available for tutor
    solution notebooks and for grading.

    Args:
        grad_f: Gradient function (takes (2,) array, returns (2,) array).
        x0: Starting point as a (2,) array.
        alpha: Learning rate.
        steps: Number of iterations.

    Returns:
        list of (2,) arrays — the trajectory including the starting point.
    """
    path = [np.array(x0, dtype=float)]
    x = np.array(x0, dtype=float)
    for _ in range(steps):
        x = x - alpha * grad_f(x)
        path.append(x.copy())
    return path


# --- Non-convex 2D landscape (2026-05-29): two Gaussian wells on a shallow bowl ---
# Two basins of attraction:
#   Right basin (global): minimum near (2, 1), deeper well (-3)
#   Left basin (local):   minimum near (-2, -1), shallower well (-1.5)
# A shallow quadratic bowl prevents the function from being flat far away,
# so gradient descent always moves toward one of the two basins.

def f_harder_2d(x):
    """Two Gaussian wells on a shallow bowl.

    f(x1, x2) = 0.1 (x1^2 + x2^2)
              - 3.0 * exp(-((x1 - 2)^2 + (x2 - 1)^2))
              - 1.5 * exp(-((x1 + 2)^2 + (x2 + 1)^2))

    Approximate global minimum near (2, 1) (deeper well).
    Approximate local minimum near (-2, -1) (shallower well).
    """
    g1 = np.exp(-((x[0] - 2.0) ** 2 + (x[1] - 1.0) ** 2))
    g2 = np.exp(-((x[0] + 2.0) ** 2 + (x[1] + 1.0) ** 2))
    return 0.1 * (x[0] ** 2 + x[1] ** 2) - 3.0 * g1 - 1.5 * g2


def grad_harder_2d(x):
    """Analytical gradient of f_harder_2d.

    Returns a (2,) NumPy array.
    """
    g1 = np.exp(-((x[0] - 2.0) ** 2 + (x[1] - 1.0) ** 2))
    g2 = np.exp(-((x[0] + 2.0) ** 2 + (x[1] + 1.0) ** 2))
    dx1 = 0.2 * x[0] + 6.0 * (x[0] - 2.0) * g1 + 3.0 * (x[0] + 2.0) * g2
    dx2 = 0.2 * x[1] + 6.0 * (x[1] - 1.0) * g1 + 3.0 * (x[1] + 1.0) * g2
    return np.array([dx1, dx2])


HARDER_2D_GLOBAL_MIN = np.array([2.0, 1.0])    # approximate; deeper well
HARDER_2D_LOCAL_MIN = np.array([-2.0, -1.0])  # approximate; shallower well
HARDER_2D_SADDLE = np.array([0.0, 0.0])       # approximate saddle between basins


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("P9 Helpers")
    print(f"  f_quadratic(2) = {f_quadratic(2)}")
    print(f"  df_quadratic(2) = {df_quadratic(2)}")
    print(f"  f_harder(3) = {f_harder(3):.4f}")
    print(f"  Hand trace 1D: w0={HAND_TRACE_W0}, alpha={HAND_TRACE_ALPHA}")
    print(f"  f_bowl_2d([-3, -2]) = {f_bowl_2d(np.array([-3.0, -2.0]))}")
    print(f"  grad_bowl_2d([-3, -2]) = {grad_bowl_2d(np.array([-3.0, -2.0]))}")
    print(f"  Hand trace 2D: x0={HAND_TRACE_2D_START}, "
          f"expected next={HAND_TRACE_2D_EXPECTED_NEXT}")
