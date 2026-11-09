"""
P4 Helper Functions — Hill Climbing on 8-Queens

FIT1061 Introduction to Artificial Intelligence

This module provides the tools you need for P4.
You do NOT need to modify this file. Import it in your notebook:

    from p4_helpers import (random_board, count_violations, get_neighbours,
                            visualise_board, plot_violations)

Functions:
    random_board()        — Generate a random 8-queens board
    count_violations(b)   — Count pairs of queens attacking each other
    get_neighbours(b)     — Generate all boards reachable by moving one queen
    visualise_board(b)    — Draw the board with queens
    plot_violations(log)  — Plot violations per iteration
"""

import random
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# ---------------------------------------------------------------------------
# Board representation
# ---------------------------------------------------------------------------
#
# A board is a list of 8 integers: board[row] = column of the queen in
# that row. For example:
#
#   board = [0, 4, 7, 5, 2, 6, 1, 3]
#
# means: row 0 has a queen in column 0, row 1 has a queen in column 4, etc.
# Since there's exactly one queen per row, we only need to worry about
# column and diagonal conflicts.


def random_board(n=8):
    """Generate a random board with one queen per row.

    Each queen is placed in a random column (0 to n-1).
    Columns CAN repeat — that's a conflict the algorithm must fix.

    Args:
        n: Board size (default 8).

    Returns:
        A list of n integers, each in range [0, n-1].
    """
    return [random.randint(0, n - 1) for _ in range(n)]


def count_violations(board):
    """Count the number of pairs of queens attacking each other.

    Two queens attack each other if they share a column or a diagonal.
    (They can't share a row because our representation puts one queen
    per row.)

    Args:
        board: A list of column positions, one per row.

    Returns:
        The number of attacking pairs (0 means solved).
    """
    n = len(board)
    violations = 0
    for i in range(n):
        for j in range(i + 1, n):
            # Same column
            if board[i] == board[j]:
                violations += 1
            # Same diagonal (difference in rows == difference in columns)
            if abs(board[i] - board[j]) == abs(i - j):
                violations += 1
    return violations


def get_neighbours(board):
    """Generate all boards reachable by moving one queen within its row.

    For each row, try every other column. This gives n * (n-1) neighbours
    for an n-queens board (8 * 7 = 56 for 8-queens).

    Args:
        board: A list of column positions, one per row.

    Returns:
        A list of (new_board, row, new_col) tuples. Each new_board is a
        copy of the original with one queen moved.
    """
    n = len(board)
    neighbours = []
    for row in range(n):
        for col in range(n):
            if col != board[row]:
                new_board = list(board)
                new_board[row] = col
                neighbours.append((new_board, row, col))
    return neighbours


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def visualise_board(board, title="8-Queens Board", highlight_conflicts=True):
    """Draw the chessboard with queens.

    Args:
        board: A list of column positions, one per row.
        title: Plot title.
        highlight_conflicts: If True, draw red lines between attacking pairs.
    """
    n = len(board)
    fig, ax = plt.subplots(figsize=(6, 6))

    # Draw checkerboard
    for row in range(n):
        for col in range(n):
            color = '#F0D9B5' if (row + col) % 2 == 0 else '#B58863'
            ax.add_patch(plt.Rectangle((col, n - 1 - row), 1, 1,
                                       facecolor=color, edgecolor='none'))

    # Draw queens
    for row in range(n):
        col = board[row]
        ax.text(col + 0.5, n - 1 - row + 0.5, '♛',
                fontsize=28, ha='center', va='center',
                color='#1a1a1a')

    # Highlight conflicts
    if highlight_conflicts:
        for i in range(n):
            for j in range(i + 1, n):
                conflict = False
                if board[i] == board[j]:
                    conflict = True
                if abs(board[i] - board[j]) == abs(i - j):
                    conflict = True
                if conflict:
                    ax.plot([board[i] + 0.5, board[j] + 0.5],
                            [n - 1 - i + 0.5, n - 1 - j + 0.5],
                            'r-', linewidth=2, alpha=0.5)

    violations = count_violations(board)
    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(range(n))
    ax.set_yticklabels(range(n - 1, -1, -1))
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.set_title(f"{title}\nViolations: {violations}")
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.show()


def plot_violations(violations_log, title="Hill Climbing Progress"):
    """Plot violations per iteration.

    Args:
        violations_log: A list of violation counts, one per iteration.
        title: Plot title.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(violations_log, 'b-o', markersize=4, linewidth=1.5)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Violations (attacking pairs)")
    ax.set_title(title)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)

    # Mark final value
    final = violations_log[-1]
    if final == 0:
        ax.axhline(y=0, color='green', linestyle='--', alpha=0.5)
        ax.text(len(violations_log) - 1, 0.3, 'SOLVED!',
                color='green', fontweight='bold', ha='right')
    else:
        ax.text(len(violations_log) - 1, final + 0.3,
                f'Stuck at {final}', color='red', fontweight='bold',
                ha='right')

    plt.tight_layout()
    plt.show()


def plot_multiple_runs(all_logs, title="Hill Climbing — Multiple Runs"):
    """Plot violations per iteration for multiple runs overlaid.

    Args:
        all_logs: A list of violations_log lists.
        title: Plot title.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, log in enumerate(all_logs):
        color = 'green' if log[-1] == 0 else 'red'
        alpha = 0.7 if log[-1] == 0 else 0.3
        label = f"Run {i+1} ({'solved' if log[-1] == 0 else 'stuck'})"
        ax.plot(log, '-', color=color, alpha=alpha, linewidth=1.5,
                label=label)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Violations")
    ax.set_title(title)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc='upper right', ncol=2)

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Demo
    board = random_board()
    print(f"Random board: {board}")
    print(f"Violations: {count_violations(board)}")
    print(f"Number of neighbours: {len(get_neighbours(board))}")
    visualise_board(board)
