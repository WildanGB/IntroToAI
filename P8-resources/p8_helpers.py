"""
P8 Helper Data — Weighted Sums & Prediction

FIT1061 Introduction to Artificial Intelligence

This module provides datasets and utility functions for P8.
You do NOT need to modify this file. Import it in your notebook:

    from p8_helpers import (HAND_TRACE_POINTS, HAND_TRACE_LABELS,
                            HAND_TRACE_WEIGHTS, HAND_TRACE_BIAS,
                            TRAIN_POINTS, TRAIN_LABELS,
                            TEST_POINTS, TEST_LABELS,
                            XOR_POINTS, XOR_LABELS,
                            plot_data, plot_boundary,
                            print_dataset_summary, print_confusion_matrix)

The dataset uses NumPy arrays. Each point is a 2D coordinate and
each label is +1 or -1.
"""

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Hand-Trace Dataset (for P8.1)
# ---------------------------------------------------------------------------
# 6 points in 2D, small enough to compute w·x + b by hand.
# With w = [2, -1] and b = -1, all 6 classify correctly.

HAND_TRACE_POINTS = np.array([
    [3, 1],
    [1, 3],
    [4, 0],
    [1, 4],
    [5, 2],
    [0, 2],
])

HAND_TRACE_LABELS = np.array([1, -1, 1, -1, 1, -1])

HAND_TRACE_WEIGHTS = np.array([2, -1])
HAND_TRACE_BIAS = -1

# Expected hand trace:
# Point (3,1): 2*3 + (-1)*1 + (-1) =  4  >= 0 → +1  (actual +1) ✓
# Point (1,3): 2*1 + (-1)*3 + (-1) = -2  <  0 → -1  (actual -1) ✓
# Point (4,0): 2*4 + (-1)*0 + (-1) =  7  >= 0 → +1  (actual +1) ✓
# Point (1,4): 2*1 + (-1)*4 + (-1) = -3  <  0 → -1  (actual -1) ✓
# Point (5,2): 2*5 + (-1)*2 + (-1) =  7  >= 0 → +1  (actual +1) ✓
# Point (0,2): 2*0 + (-1)*2 + (-1) = -3  <  0 → -1  (actual -1) ✓
# Accuracy: 6/6 = 100%


# ---------------------------------------------------------------------------
# Main Dataset (for P8.1 and P8.2)
# ---------------------------------------------------------------------------
# 40 points in 2D: two Gaussian clusters.
# Class +1 centred around (3, 3), class -1 centred around (0, 0).
# Pre-split: 30 training, 10 test. Deterministic — every student
# gets the same results.

TRAIN_POINTS = np.array([
    [ 3.40,  2.89],
    [ 3.52,  4.22],
    [ 2.81,  2.81],
    [ 4.26,  3.61],
    [ 2.62,  3.43],
    [ 2.63,  2.63],
    [ 3.19,  1.47],
    [ 1.62,  2.55],
    [ 2.19,  3.25],
    [ 2.27,  1.87],
    [ 4.17,  2.82],
    [ 3.05,  1.86],
    [ 2.56,  3.09],
    [ 2.08,  3.30],
    [ 2.52,  2.77],
    [ 0.59,  0.14],
    [-0.09, -0.24],
    [-1.18, -0.58],
    [-0.37,  0.85],
    [ 0.27, -1.41],
    [ 0.26, -0.31],
    [-0.54,  0.49],
    [ 0.82,  0.75],
    [-0.67, -0.25],
    [ 0.27,  0.78],
    [-0.38, -0.15],
    [-0.89, -0.96],
    [ 0.65,  1.08],
    [-0.06,  0.80],
    [ 0.29, -0.52],
])

TRAIN_LABELS = np.array([
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
])

TEST_POINTS = np.array([
    [ 2.52,  4.48],
    [ 2.99,  2.15],
    [ 3.66,  2.02],
    [ 3.17,  1.43],
    [ 1.94,  3.16],
    [ 0.29,  1.23],
    [-0.03,  1.25],
    [-2.10,  0.66],
    [ 0.07, -0.24],
    [ 0.07, -1.59],
])

TEST_LABELS = np.array([1, 1, 1, 1, 1, -1, -1, -1, -1, -1])


# ---------------------------------------------------------------------------
# XOR Dataset (for Part 5)
# ---------------------------------------------------------------------------
# The problem that killed the perceptron. No single line can
# separate +1 from -1.

XOR_POINTS = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
])

XOR_LABELS = np.array([-1, 1, 1, -1])


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_data(X, y, title="Dataset", ax=None):
    """Scatter plot of 2D labelled data.

    Args:
        X: Array of shape (n, 2) — data points.
        y: Array of shape (n,) — labels (+1 or -1).
        title: Plot title.
        ax: Optional matplotlib Axes. Creates a new figure if None.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(7, 6))

    pos = y == 1
    neg = y == -1
    ax.scatter(X[pos, 0], X[pos, 1], c='#3498db', marker='o', s=60,
               edgecolors='black', linewidths=0.5, label='+1', zorder=3)
    ax.scatter(X[neg, 0], X[neg, 1], c='#e74c3c', marker='s', s=60,
               edgecolors='black', linewidths=0.5, label='-1', zorder=3)
    ax.set_xlabel('$x_1$', fontsize=12)
    ax.set_ylabel('$x_2$', fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()


def plot_boundary(X, y, w, b, title="Decision Boundary"):
    """Plot data points and the decision boundary w·x + b = 0.

    The boundary is the line where w[0]*x1 + w[1]*x2 + b = 0.
    Also prints the accuracy of this weight vector on the data.

    Args:
        X: Array of shape (n, 2) — data points.
        y: Array of shape (n,) — labels (+1 or -1).
        w: Array of shape (2,) — weight vector.
        b: float — bias.
        title: Plot title.
    """
    fig, ax = plt.subplots(1, 1, figsize=(7, 6))

    # Plot data
    pos = y == 1
    neg = y == -1
    ax.scatter(X[pos, 0], X[pos, 1], c='#3498db', marker='o', s=60,
               edgecolors='black', linewidths=0.5, label='+1', zorder=3)
    ax.scatter(X[neg, 0], X[neg, 1], c='#e74c3c', marker='s', s=60,
               edgecolors='black', linewidths=0.5, label='-1', zorder=3)

    # Plot decision boundary: w[0]*x1 + w[1]*x2 + b = 0
    xlim = ax.get_xlim()
    x1_range = np.linspace(xlim[0] - 1, xlim[1] + 1, 200)

    if abs(w[1]) > 1e-10:
        # x2 = -(w[0]*x1 + b) / w[1]
        x2_boundary = -(w[0] * x1_range + b) / w[1]
        ax.plot(x1_range, x2_boundary, 'k-', linewidth=2,
                label=f'boundary: {w[0]}$x_1$ + {w[1]}$x_2$ + {b} = 0')
    elif abs(w[0]) > 1e-10:
        # Vertical line: x1 = -b / w[0]
        x1_val = -b / w[0]
        ax.axvline(x=x1_val, color='k', linewidth=2,
                   label=f'boundary: $x_1$ = {x1_val:.2f}')
    else:
        ax.set_title(title + " (WARNING: zero weights!)", fontsize=13)

    # Shade regions
    if abs(w[1]) > 1e-10:
        x2_boundary = -(w[0] * x1_range + b) / w[1]
        ax.fill_between(x1_range, x2_boundary, ax.get_ylim()[1] + 5,
                        alpha=0.08, color='#3498db')
        ax.fill_between(x1_range, ax.get_ylim()[0] - 5, x2_boundary,
                        alpha=0.08, color='#e74c3c')

    ax.set_xlabel('$x_1$', fontsize=12)
    ax.set_ylabel('$x_2$', fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Set sensible axis limits
    margin = 1.0
    ax.set_xlim(X[:, 0].min() - margin, X[:, 0].max() + margin)
    ax.set_ylim(X[:, 1].min() - margin, X[:, 1].max() + margin)

    plt.tight_layout()
    plt.show()

    # Print accuracy
    predictions = np.where(X @ np.array(w) + b >= 0, 1, -1)
    accuracy = np.mean(predictions == y)
    print(f"Accuracy: {int(accuracy * len(y))}/{len(y)} = {accuracy:.0%}")


def print_dataset_summary(X, y, label="Dataset"):
    """Print a summary of the dataset."""
    pos = int(np.sum(y == 1))
    neg = int(np.sum(y == -1))
    print(f"{label}: {len(y)} points ({pos} positive, {neg} negative)")
    print(f"  Features: {X.shape[1]}D")
    print(f"  x1 range: [{X[:, 0].min():.2f}, {X[:, 0].max():.2f}]")
    print(f"  x2 range: [{X[:, 1].min():.2f}, {X[:, 1].max():.2f}]")


def print_confusion_matrix(tp, fp, fn, tn, positive_label="+1", negative_label="-1"):
    """Pretty-print a 2x2 confusion matrix for binary classification.

    Convention: positive class is +1; negative class is -1.

    Args:
        tp: True positives.
        fp: False positives.
        fn: False negatives.
        tn: True negatives.
        positive_label: Label for the positive class (default "+1").
        negative_label: Label for the negative class (default "-1").
    """
    print(f"                       Predicted")
    print(f"                  {positive_label:>5s}    {negative_label:>5s}")
    print(f"  Actual {positive_label:>3s}  |  {tp:3d}  |  {fn:3d}  |")
    print(f"  Actual {negative_label:>3s}  |  {fp:3d}  |  {tn:3d}  |")
    print()
    print(f"  True Positives  (TP):  {tp}")
    print(f"  False Positives (FP):  {fp}")
    print(f"  False Negatives (FN):  {fn}")
    print(f"  True Negatives  (TN):  {tn}")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print_dataset_summary(TRAIN_POINTS, TRAIN_LABELS, "Training set")
    print_dataset_summary(TEST_POINTS, TEST_LABELS, "Test set")
    print(f"\nHand-trace: {len(HAND_TRACE_POINTS)} points")
    print(f"  Weights: {HAND_TRACE_WEIGHTS}, bias: {HAND_TRACE_BIAS}")
    print(f"\nXOR: {len(XOR_POINTS)} points")
