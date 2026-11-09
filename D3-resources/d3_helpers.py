"""
D3 Helper Data — Perceptron Training (XOR + Visualisation)

FIT1061 Introduction to Artificial Intelligence
Distinction Task D3: Perceptron Training

This module provides datasets, visualisation helpers, and a small
animation utility for D3. You do NOT need to modify this file.

    from d3_helpers import (
        # Datasets (NumPy arrays, labels in {-1, +1} matching P8 convention)
        SEPARABLE_POINTS, SEPARABLE_LABELS,
        XOR_POINTS, XOR_LABELS,
        # Plotting
        plot_data, plot_boundary, plot_boundary_evolution,
        plot_learning_curve,
        # Convenience
        sign, accuracy,
    )

Datasets:
    SEPARABLE_*  — 40 points, two Gaussian clusters, linearly separable
    XOR_*        — the four XOR corners (cannot be separated by any line)

All labels are in {-1, +1} (same convention as P8).
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# =============================================================
# Brand palette (Monash 2026) — also used downstream by plots
# =============================================================
MONASH_BLUE = "#006DAE"
BLUEBERRY = "#121256"
UTIL_ORANGE = "#F86700"
FOREST = "#0B6554"
UTIL_RED = "#EA001F"
GREY_1 = "#5A5A5A"
GREY_3 = "#E6E6E6"


# =============================================================
# Datasets
# =============================================================

# A linearly separable dataset. Two Gaussian clusters in 2D, centred
# at (3, 3) for class +1 and (0, 0) for class -1. 40 points total.
# Pre-generated and frozen so every student sees the same numbers.
SEPARABLE_POINTS = np.array([
    [3.40, 2.89], [3.52, 4.22], [2.81, 2.81], [4.26, 3.61],
    [2.62, 3.43], [2.63, 2.63], [3.19, 1.47], [1.62, 2.55],
    [2.19, 3.25], [2.27, 1.87], [4.17, 2.82], [3.05, 1.86],
    [2.56, 3.09], [2.08, 3.30], [2.52, 2.77], [3.83, 2.45],
    [3.05, 3.12], [2.44, 2.74], [3.71, 3.95], [2.86, 3.30],
    [0.59, 0.14], [-0.09, -0.24], [0.41, -0.39], [-0.45, 0.97],
    [-0.62, 0.30], [0.07, -0.21], [1.02, -0.81], [-0.20, -0.43],
    [-0.12, 0.78], [-0.78, -0.05], [0.31, 1.13], [0.65, -1.21],
    [-1.07, -0.06], [-0.41, -0.78], [0.93, -0.12], [-0.43, 0.74],
    [-0.18, -0.66], [0.51, 0.30], [-0.66, -0.94], [1.21, 0.40],
])
SEPARABLE_LABELS = np.array([1] * 20 + [-1] * 20)


# XOR. Four corners of the unit square. The perceptron cannot learn
# this with a single linear boundary.
XOR_POINTS = np.array([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0],
])
XOR_LABELS = np.array([-1, 1, 1, -1])


# =============================================================
# Convenience numerics
# =============================================================

def sign(x: float) -> int:
    """Return +1 if x >= 0 else -1.

    The "tie at zero" convention varies across textbooks; we send 0 to +1
    to keep the sign function single-valued and to match P8.
    """
    return 1 if x >= 0 else -1


def accuracy(predictions: np.ndarray, labels: np.ndarray) -> float:
    """Fraction of predictions matching labels (in [0, 1])."""
    return float(np.mean(predictions == labels))


# =============================================================
# Plotting
# =============================================================

def plot_data(
    points: np.ndarray,
    labels: np.ndarray,
    title: str = "Data",
    ax: plt.Axes | None = None,
) -> None:
    """Scatter plot of a 2D dataset with class colour-coding."""
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(5, 5))

    pos = labels == 1
    neg = labels == -1
    ax.scatter(points[pos, 0], points[pos, 1], c=MONASH_BLUE,
               edgecolors="white", linewidths=1.2, s=80, label="+1")
    ax.scatter(points[neg, 0], points[neg, 1], c=UTIL_ORANGE,
               edgecolors="white", linewidths=1.2, s=80, label="-1")
    ax.set_xlabel("x_1", color=MONASH_BLUE)
    ax.set_ylabel("x_2", color=MONASH_BLUE)
    ax.set_title(title, color=MONASH_BLUE)
    ax.legend(loc="best", frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)

    if own_fig:
        plt.tight_layout()
        plt.show()


def _line_endpoints(weights: np.ndarray, bias: float,
                    xlim: tuple[float, float]):
    """Return two endpoints of the decision boundary line w·x + b = 0.

    For 2D, the line is w0*x + w1*y + b = 0. If w1 != 0 we solve for y:
        y = -(w0*x + b) / w1
    If w1 == 0 the line is vertical: x = -b / w0.
    """
    w0, w1 = float(weights[0]), float(weights[1])
    if abs(w1) < 1e-9:
        if abs(w0) < 1e-9:
            return None  # degenerate (no boundary)
        x = -bias / w0
        return (x, xlim[0]), (x, xlim[1])
    xs = np.array(xlim)
    ys = -(w0 * xs + bias) / w1
    return (xs[0], ys[0]), (xs[1], ys[1])


def plot_boundary(
    points: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    bias: float,
    title: str = "Decision boundary",
    ax: plt.Axes | None = None,
) -> None:
    """Scatter plot plus the linear decision boundary w·x + b = 0."""
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(5, 5))

    # Padded axis limits from the data
    x_min, x_max = points[:, 0].min() - 1, points[:, 0].max() + 1
    y_min, y_max = points[:, 1].min() - 1, points[:, 1].max() + 1

    # Coloured half-plane (light wash to show the prediction region)
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(y_min, y_max, 200),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    activations = grid @ weights + bias
    preds = np.where(activations >= 0, 1, -1).reshape(xx.shape)
    ax.contourf(xx, yy, preds, levels=[-1.5, 0, 1.5],
                colors=[UTIL_ORANGE + "33", MONASH_BLUE + "33"])

    # Boundary line
    endpoints = _line_endpoints(weights, bias, (x_min, x_max))
    if endpoints is not None:
        (x0, y0), (x1, y1) = endpoints
        ax.plot([x0, x1], [y0, y1], color=GREY_1, linewidth=2,
                label="w·x + b = 0")

    plot_data(points, labels, title=title, ax=ax)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    if own_fig:
        plt.tight_layout()
        plt.show()


def plot_boundary_evolution(
    points: np.ndarray,
    labels: np.ndarray,
    snapshots: list[tuple[int, np.ndarray, float]],
    title: str = "Decision boundary across epochs",
) -> None:
    """Plot up to 6 boundary snapshots side-by-side.

    Args:
        snapshots: list of (epoch, weights, bias) tuples produced by the
            training loop. Function picks 4–6 evenly spaced snapshots.
    """
    n = min(len(snapshots), 6)
    if n == 0:
        print("plot_boundary_evolution: no snapshots to plot.")
        return
    if len(snapshots) <= n:
        chosen = snapshots
    else:
        idxs = [int(round(i * (len(snapshots) - 1) / (n - 1)))
                for i in range(n)]
        chosen = [snapshots[i] for i in idxs]

    fig, axes = plt.subplots(1, len(chosen), figsize=(3.2 * len(chosen), 3.2))
    if len(chosen) == 1:
        axes = [axes]
    for ax, (epoch, w, b) in zip(axes, chosen):
        plot_boundary(points, labels, w, b,
                      title=f"epoch {epoch}", ax=ax)
    fig.suptitle(title, color=MONASH_BLUE, fontsize=12)
    plt.tight_layout()
    plt.show()


def plot_learning_curve(
    epoch_accuracies: list[float],
    title: str = "Learning curve",
    ax: plt.Axes | None = None,
) -> None:
    """Plot training accuracy vs epoch."""
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(7, 3.5))

    epochs = list(range(len(epoch_accuracies)))
    ax.plot(epochs, epoch_accuracies, color=MONASH_BLUE, linewidth=2)
    ax.scatter(epochs, epoch_accuracies, color=MONASH_BLUE, s=15, zorder=3)
    ax.set_xlabel("epoch", color=MONASH_BLUE)
    ax.set_ylabel("training accuracy", color=MONASH_BLUE)
    ax.set_title(title, color=MONASH_BLUE)
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(1.0, color=FOREST, linestyle="--", alpha=0.5,
               label="perfect classification")
    ax.legend(loc="lower right", frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.2)

    if own_fig:
        plt.tight_layout()
        plt.show()


# =============================================================
# Self-test (manual)
# =============================================================

if __name__ == "__main__":
    print(f"SEPARABLE_POINTS shape: {SEPARABLE_POINTS.shape}")
    print(f"XOR_POINTS:\n{XOR_POINTS}")
    print(f"XOR_LABELS: {XOR_LABELS}")
    assert SEPARABLE_LABELS.shape[0] == SEPARABLE_POINTS.shape[0]
    assert XOR_LABELS.shape[0] == 4
    print("OK: shapes consistent")
    print(f"sign(0.0) = {sign(0.0)} (expected +1)")
    print(f"sign(-0.1) = {sign(-0.1)} (expected -1)")
