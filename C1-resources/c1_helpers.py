"""
C1 Helper Data — Search Extensions

FIT1061 Introduction to Artificial Intelligence

This module provides data and utilities for C1. It builds on your
P2, P3, and P4 work — you'll reuse your search and hill climbing code.

    from c1_helpers import (SPARSE_GRAPH, DENSE_GRAPH, MISLEADING_GRAPH,
                            GRAPH_CONFIGS, visualise_graph,
                            path_cost, plot_restarts_curve,
                            random_board, count_violations,
                            get_neighbours, visualise_board)
"""

import random

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# C1.2: Multi-Graph Comparison Data
# ---------------------------------------------------------------------------
# Three graphs with different structures to test how BFS, DFS, and greedy
# best-first search behave under different conditions.

# Graph 1: SPARSE — tree-like, few edges, one clear path.
# BFS, DFS, and greedy should behave similarly here.
SPARSE_GRAPH = {
    'S': {'A': 3, 'B': 5},
    'A': {'S': 3, 'C': 4},
    'B': {'S': 5, 'D': 2},
    'C': {'A': 4, 'G': 6},
    'D': {'B': 2, 'G': 3},
    'G': {'C': 6, 'D': 3},
}

SPARSE_HEURISTIC = {
    'S': 8.0, 'A': 6.0, 'B': 4.0,
    'C': 5.0, 'D': 3.0, 'G': 0.0,
}

SPARSE_POSITIONS = {
    'S': (0, 1), 'A': (2, 2), 'B': (2, 0),
    'C': (4, 2), 'D': (4, 0), 'G': (6, 1),
}

# Graph 2: DENSE — many connections, multiple paths of varying cost.
# BFS finds fewest-edges path; greedy may find a different (possibly
# shorter cost) path. Good for showing the algorithms' different priorities.
DENSE_GRAPH = {
    'S': {'A': 1, 'B': 4, 'C': 7},
    'A': {'S': 1, 'B': 2, 'D': 5, 'E': 8},
    'B': {'S': 4, 'A': 2, 'C': 1, 'D': 3, 'E': 6},
    'C': {'S': 7, 'B': 1, 'E': 2, 'G': 9},
    'D': {'A': 5, 'B': 3, 'E': 1, 'G': 4},
    'E': {'A': 8, 'B': 6, 'C': 2, 'D': 1, 'G': 3},
    'G': {'C': 9, 'D': 4, 'E': 3},
}

DENSE_HEURISTIC = {
    'S': 7.0, 'A': 6.0, 'B': 4.0,
    'C': 3.0, 'D': 3.5, 'E': 2.0, 'G': 0.0,
}

DENSE_POSITIONS = {
    'S': (0, 2), 'A': (2, 3), 'B': (2, 1),
    'C': (3, 0), 'D': (4, 3), 'E': (4, 1), 'G': (6, 2),
}

# Graph 3: MISLEADING — heuristic leads greedy astray.
# The heuristic says F looks close to G, but the actual path through F
# costs much more. Greedy follows the heuristic and gets a worse path;
# BFS ignores the heuristic and finds the shorter path.
#
# Good path:  S → A → B → G  (cost 3+2+3 = 8, but h(A)=6.0 looks far)
# Trap path:  S → F → H → I → G  (cost 1+5+5+4 = 15, but h(F)=2.0 looks close)
MISLEADING_GRAPH = {
    'S': {'A': 3, 'F': 1},
    'A': {'S': 3, 'B': 2},
    'B': {'A': 2, 'G': 3},
    'F': {'S': 1, 'H': 5},
    'H': {'F': 5, 'I': 5},
    'I': {'H': 5, 'G': 4},
    'G': {'B': 3, 'I': 4},
}

MISLEADING_HEURISTIC = {
    'S': 6.0, 'A': 6.0, 'B': 3.0,
    'F': 2.0, 'H': 1.5, 'I': 3.0, 'G': 0.0,
}

MISLEADING_POSITIONS = {
    'S': (0, 2), 'A': (2, 3), 'B': (4, 3),
    'F': (1, 0.5), 'H': (3, 0), 'I': (5, 0.5),
    'G': (6, 2),
}

# Convenience: all three graphs packaged together for easy iteration.
GRAPH_CONFIGS = {
    'sparse': {
        'graph': SPARSE_GRAPH,
        'heuristic': SPARSE_HEURISTIC,
        'positions': SPARSE_POSITIONS,
        'start': 'S',
        'goal': 'G',
        'description': 'Sparse (tree-like, few paths)',
    },
    'dense': {
        'graph': DENSE_GRAPH,
        'heuristic': DENSE_HEURISTIC,
        'positions': DENSE_POSITIONS,
        'start': 'S',
        'goal': 'G',
        'description': 'Dense (many connections, many paths)',
    },
    'misleading': {
        'graph': MISLEADING_GRAPH,
        'heuristic': MISLEADING_HEURISTIC,
        'positions': MISLEADING_POSITIONS,
        'start': 'S',
        'goal': 'G',
        'description': 'Misleading (heuristic leads greedy astray)',
    },
}


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def visualise_graph(graph, positions, heuristic=None, start='S', goal='G',
                    path=None, title="Graph"):
    """Visualise a weighted graph with optional heuristic values and path.

    Args:
        graph: Dict of dicts {node: {neighbour: weight}}.
        positions: Dict mapping each node to (x, y).
        heuristic: Optional dict mapping each node to h(n).
        start: Start node name.
        goal: Goal node name.
        path: Optional list of node names to highlight.
        title: Plot title.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Draw edges with weights
    drawn = set()
    for v, edges in graph.items():
        for nb, weight in edges.items():
            edge = tuple(sorted([v, nb]))
            if edge not in drawn:
                x = [positions[v][0], positions[nb][0]]
                y = [positions[v][1], positions[nb][1]]
                ax.plot(x, y, 'k-', linewidth=1, alpha=0.4)
                mx, my = (x[0]+x[1])/2, (y[0]+y[1])/2
                ax.text(mx, my, str(weight), fontsize=8, ha='center',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                                  edgecolor='none', alpha=0.8))
                drawn.add(edge)

    # Highlight path
    if path and len(path) > 1:
        for i in range(len(path) - 1):
            x = [positions[path[i]][0], positions[path[i+1]][0]]
            y = [positions[path[i]][1], positions[path[i+1]][1]]
            ax.plot(x, y, '-', color='#e74c3c', linewidth=3, zorder=2)

    # Draw nodes
    for v, (x, y) in positions.items():
        colour = '#2ecc71' if v == start else '#e74c3c' if v == goal else '#3498db'
        ax.scatter(x, y, s=600, c=colour, edgecolors='black',
                   linewidths=1.5, zorder=3)
        label = v
        if heuristic and v in heuristic:
            label = f"{v}\nh={heuristic[v]}"
        ax.text(x, y, label, ha='center', va='center', fontsize=8,
                fontweight='bold', zorder=4)

    ax.set_title(title, fontsize=13)
    ax.axis('off')
    plt.tight_layout()
    plt.show()


def plot_restarts_curve(restart_counts, success_rates, title="Success Rate vs Restarts"):
    """Plot success rate as a function of number of restarts.

    Args:
        restart_counts: List of restart counts tested.
        success_rates: List of success rates (0-1) for each restart count.
        title: Plot title.
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(restart_counts, [s * 100 for s in success_rates], 'o-',
            color='#3498db', linewidth=2, markersize=8)
    ax.axhline(y=95, color='#e74c3c', linestyle='--', alpha=0.7, label='95% target')
    ax.set_xlabel('Number of Restarts', fontsize=12)
    ax.set_ylabel('Success Rate (%)', fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def path_cost(graph, path):
    """Compute total edge cost of a path.

    Args:
        graph: Dict of dicts {node: {neighbour: weight}}.
        path: List of node names.

    Returns:
        Total cost (sum of edge weights along path).
    """
    cost = 0
    for i in range(len(path) - 1):
        cost += graph[path[i]][path[i+1]]
    return cost


# ---------------------------------------------------------------------------
# C1.1: 8-queens board primitives
# ---------------------------------------------------------------------------
# Mirrored from p4_helpers.py so C1 runs from its own download alone (you do
# not need the P4 folder present). These are scaffolding — the hill-climbing
# algorithm is still yours to adapt from P4 in the notebook.

def random_board(n=8):
    """Generate a random board with one queen per row.

    Each queen is placed in a random column (0 to n-1). Columns CAN repeat —
    that's a conflict the algorithm must fix.

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


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("C1 Helpers")
    for name, cfg in GRAPH_CONFIGS.items():
        g = cfg['graph']
        print(f"  {name}: {len(g)} nodes, {cfg['description']}")
