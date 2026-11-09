"""
D1 Helper Classes — A* Search on the 8-Puzzle

FIT1061 Introduction to Artificial Intelligence
Distinction Task D1: A* with Admissible Heuristic Design

This module provides the data structures and reference search algorithms
you need for D1. You do NOT need to modify this file. Import it in your notebook:

    from d1_helpers import (
        Puzzle8, PriorityQueue, Queue,
        is_solvable, random_puzzle, parse_state,
        visualise_puzzle, plot_search_comparison,
        bfs_search, greedy_best_first_search,
    )

Classes:
    Puzzle8       — Immutable 3x3 sliding-tile puzzle state
    PriorityQueue — Min-priority queue (same shape as P3)
    Queue         — FIFO queue (same as P2)

Functions:
    is_solvable(state)       — True if an 8-puzzle state is reachable from the goal
    random_puzzle(...)       — Generate a random solvable puzzle by k random shuffles
    parse_state(string)      — Build a Puzzle8 from a 9-char string ("123405678" etc.)
    visualise_puzzle(p)      — Show one Puzzle8 state as a 3x3 grid figure
    plot_search_comparison() — Bar/box plots for empirical comparison
    bfs_search(start)        — Reference BFS (for comparison only)
    greedy_best_first_search — Reference greedy best-first (for comparison only)

You implement A* in the notebook. The reference BFS and greedy
implementations are provided so the empirical-comparison sub-task does
not require re-implementing them.
"""

from __future__ import annotations

import heapq
import random
from dataclasses import dataclass
from typing import Callable, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# =============================================================
# Brand palette (Monash 2026)
# =============================================================
MONASH_BLUE = "#006DAE"
BLUEBERRY = "#121256"
UTIL_ORANGE = "#F86700"
FOREST = "#0B6554"
GREY_1 = "#5A5A5A"
GREY_3 = "#E6E6E6"


# =============================================================
# Puzzle8 — immutable 3x3 sliding-tile puzzle
# =============================================================

class Puzzle8:
    """An 8-puzzle state.

    State is a 9-tuple of ints. The blank is represented by 0.
    The goal state is ``(1, 2, 3, 4, 5, 6, 7, 8, 0)`` — blank in the
    bottom-right corner.

    Two Puzzle8 instances compare equal iff their tuples match, so
    you can put them in sets and dict keys.

    Example:
        >>> p = Puzzle8((1, 2, 3, 4, 0, 6, 7, 5, 8))
        >>> p.is_goal()
        False
        >>> [n.state for n in p.neighbours()][:2]
        [(1, 0, 3, 4, 2, 6, 7, 5, 8), (1, 2, 3, 4, 5, 6, 7, 0, 8)]
    """

    GOAL: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8, 0)
    _MOVES: tuple[tuple[int, int], ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))

    __slots__ = ("state",)

    def __init__(self, state: tuple[int, ...]):
        if tuple(sorted(state)) != tuple(range(9)):
            raise ValueError(
                f"Puzzle8 state must be a permutation of 0..8, got {state}"
            )
        self.state = tuple(state)

    def neighbours(self) -> list["Puzzle8"]:
        """Return the list of Puzzle8 states reachable by sliding the blank.

        At a corner there are 2 neighbours; at an edge there are 3;
        at the centre there are 4.
        """
        blank = self.state.index(0)
        br, bc = divmod(blank, 3)
        result = []
        for dr, dc in Puzzle8._MOVES:
            nr, nc = br + dr, bc + dc
            if 0 <= nr < 3 and 0 <= nc < 3:
                target = nr * 3 + nc
                lst = list(self.state)
                lst[blank], lst[target] = lst[target], lst[blank]
                result.append(Puzzle8(tuple(lst)))
        return result

    def is_goal(self) -> bool:
        return self.state == Puzzle8.GOAL

    def __hash__(self) -> int:
        return hash(self.state)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Puzzle8) and self.state == other.state

    def __repr__(self) -> str:
        return f"Puzzle8{self.state}"


def parse_state(s: str) -> Puzzle8:
    """Build a Puzzle8 from a 9-character string of digits 0..8.

    Useful for hand-constructed test cases. Use ``0`` for the blank.

    Example:
        >>> parse_state("123405678")
        Puzzle8((1, 2, 3, 4, 0, 5, 6, 7, 8))
    """
    if len(s) != 9 or not all(c.isdigit() for c in s):
        raise ValueError(f"parse_state expects 9 digits, got {s!r}")
    return Puzzle8(tuple(int(c) for c in s))


def is_solvable(state: tuple[int, ...] | Puzzle8) -> bool:
    """Return True iff this 8-puzzle state is reachable from the goal.

    Half of all 9-tile permutations are reachable; the rest form a
    second disconnected orbit. The reachability criterion for the
    3x3 puzzle is: number of inversions (over the non-blank tiles) is
    even.

    Reference: Russell & Norvig, AIMA 4th ed., §3.2 (8-puzzle).
    """
    tup = state.state if isinstance(state, Puzzle8) else tuple(state)
    flat = [x for x in tup if x != 0]
    inv = 0
    for i in range(len(flat)):
        for j in range(i + 1, len(flat)):
            if flat[i] > flat[j]:
                inv += 1
    return inv % 2 == 0


def random_puzzle(n_shuffles: int = 25, seed: Optional[int] = None) -> Puzzle8:
    """Generate a random *solvable* 8-puzzle by k random moves from the goal.

    Starting from the goal and applying legal moves guarantees solvability
    (the puzzle is reversible). The optimal solution length is bounded by
    n_shuffles but in practice is much smaller because moves backtrack.

    Args:
        n_shuffles: Number of random sliding moves from the goal.
        seed: Optional RNG seed for reproducibility.

    Returns:
        A solvable Puzzle8.
    """
    rng = random.Random(seed)
    p = Puzzle8(Puzzle8.GOAL)
    for _ in range(n_shuffles):
        p = rng.choice(p.neighbours())
    return p


# =============================================================
# PriorityQueue and Queue (same shape as P3 and P2)
# =============================================================

class PriorityQueue:
    """Min-priority queue. Lower priority = popped first.

    Tie-breaking is FIFO via an internal counter (stable across pushes).
    """

    def __init__(self) -> None:
        self._items: list[tuple[float, int, object]] = []
        self._counter: int = 0

    def push(self, item: object, priority: float) -> None:
        heapq.heappush(self._items, (priority, self._counter, item))
        self._counter += 1

    def pop(self) -> object:
        if self.is_empty():
            raise IndexError("pop from an empty priority queue")
        _priority, _counter, item = heapq.heappop(self._items)
        return item

    def is_empty(self) -> bool:
        return len(self._items) == 0

    def size(self) -> int:
        return len(self._items)


class Queue:
    """FIFO queue (same as P2)."""

    def __init__(self) -> None:
        self._items: list[object] = []

    def enqueue(self, item: object) -> None:
        self._items.append(item)

    def dequeue(self) -> object:
        if self.is_empty():
            raise IndexError("dequeue from an empty queue")
        return self._items.pop(0)

    def is_empty(self) -> bool:
        return len(self._items) == 0

    def size(self) -> int:
        return len(self._items)


# =============================================================
# Reference search algorithms (BFS, greedy) — for comparison only
# =============================================================

@dataclass
class SearchResult:
    """Container for search outputs.

    Attributes:
        path: List of Puzzle8 states from start to goal, or None if no path.
        nodes_expanded: Number of states popped from the frontier.
        max_frontier: Maximum size of the frontier at any point.
    """

    path: Optional[list[Puzzle8]]
    nodes_expanded: int
    max_frontier: int


def bfs_search(start: Puzzle8) -> SearchResult:
    """Breadth-first search on the 8-puzzle (reference implementation).

    Provided so students do not re-implement BFS for the empirical
    comparison sub-task (they already did that for P2). BFS finds the
    optimal solution but expands many nodes.
    """
    q = Queue()
    q.enqueue(start)
    came_from: dict[Puzzle8, Optional[Puzzle8]] = {start: None}
    nodes_expanded = 0
    max_frontier = 1
    while not q.is_empty():
        max_frontier = max(max_frontier, q.size())
        current = q.dequeue()
        if current.is_goal():
            return SearchResult(_reconstruct(came_from, current),
                                nodes_expanded, max_frontier)
        nodes_expanded += 1
        for nbr in current.neighbours():
            if nbr not in came_from:
                came_from[nbr] = current
                q.enqueue(nbr)
    return SearchResult(None, nodes_expanded, max_frontier)


def greedy_best_first_search(
    start: Puzzle8,
    heuristic: Callable[[Puzzle8], float],
) -> SearchResult:
    """Greedy best-first search on the 8-puzzle (reference implementation).

    Expands the node with the lowest h(n) regardless of path cost. Often
    fast, but not optimal — and on the 8-puzzle the path it returns may
    be substantially longer than BFS's shortest path.
    """
    pq = PriorityQueue()
    pq.push(start, priority=heuristic(start))
    came_from: dict[Puzzle8, Optional[Puzzle8]] = {start: None}
    visited: set[Puzzle8] = {start}
    nodes_expanded = 0
    max_frontier = 1
    while not pq.is_empty():
        max_frontier = max(max_frontier, pq.size())
        current = pq.pop()  # type: ignore[assignment]
        if current.is_goal():  # type: ignore[union-attr]
            return SearchResult(_reconstruct(came_from, current),
                                nodes_expanded, max_frontier)
        nodes_expanded += 1
        for nbr in current.neighbours():  # type: ignore[union-attr]
            if nbr not in visited:
                visited.add(nbr)
                came_from[nbr] = current  # type: ignore[assignment]
                pq.push(nbr, priority=heuristic(nbr))
    return SearchResult(None, nodes_expanded, max_frontier)


def _reconstruct(came_from: dict, goal: Puzzle8) -> list[Puzzle8]:
    """Walk the came_from chain to reconstruct the path from start to goal."""
    path: list[Puzzle8] = []
    node: Optional[Puzzle8] = goal
    while node is not None:
        path.append(node)
        node = came_from[node]
    path.reverse()
    return path


# =============================================================
# Visualisation
# =============================================================

def visualise_puzzle(p: Puzzle8, title: str = "8-Puzzle State",
                     ax: Optional[plt.Axes] = None) -> None:
    """Render an 8-puzzle state as a 3x3 grid figure.

    Args:
        p: Puzzle8 to render.
        title: Figure title.
        ax: Optional matplotlib Axes to draw on. If None, a new figure
            is created and shown.
    """
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(3, 3))

    for i, v in enumerate(p.state):
        r, c = divmod(i, 3)
        # Note: row 0 at the TOP of the figure → invert vertical.
        y = 2 - r
        x = c
        if v == 0:
            colour = GREY_3
            text = ""
        else:
            colour = MONASH_BLUE
            text = str(v)
        rect = plt.Rectangle((x - 0.45, y - 0.45), 0.9, 0.9,
                             facecolor=colour, edgecolor="white", linewidth=2)
        ax.add_patch(rect)
        ax.text(x, y, text, ha="center", va="center",
                fontsize=18, fontweight="bold", color="white")

    ax.set_xlim(-0.6, 2.6)
    ax.set_ylim(-0.6, 2.6)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=10, color=MONASH_BLUE)

    if own_fig:
        plt.tight_layout()
        plt.show()


def visualise_path(path: list[Puzzle8], max_steps: int = 8,
                   title: str = "Solution Path") -> None:
    """Render up to ``max_steps`` states from a solution path side-by-side."""
    n = min(len(path), max_steps)
    fig, axes = plt.subplots(1, n, figsize=(2.0 * n, 2.4))
    if n == 1:
        axes = [axes]
    step_indices = (
        list(range(n)) if len(path) <= max_steps
        else [int(round(i * (len(path) - 1) / (n - 1))) for i in range(n)]
    )
    for ax, idx in zip(axes, step_indices):
        visualise_puzzle(path[idx], title=f"Step {idx}", ax=ax)
    fig.suptitle(title + f"  (length: {len(path) - 1} moves)",
                 fontsize=11, color=MONASH_BLUE)
    plt.tight_layout()
    plt.show()


def plot_search_comparison(rows: list[dict], metric: str = "nodes_expanded",
                            title: Optional[str] = None) -> None:
    """Plot one metric across algorithms as a horizontal bar chart.

    Args:
        rows: List of dicts produced by your empirical-comparison loop.
            Each dict must have keys: ``algorithm`` (str) and ``metric``
            (whatever ``metric`` argument is set to — e.g. ``nodes_expanded``).
        metric: Which numeric field to plot.
        title: Optional plot title.
    """
    algos = [r["algorithm"] for r in rows]
    values = [r[metric] for r in rows]
    colours = [MONASH_BLUE, UTIL_ORANGE, FOREST, BLUEBERRY, GREY_1][: len(rows)]
    fig, ax = plt.subplots(figsize=(7, 0.5 * len(rows) + 1.5))
    ax.barh(algos, values, color=colours)
    ax.set_xlabel(metric.replace("_", " "), color=MONASH_BLUE)
    if title:
        ax.set_title(title, color=MONASH_BLUE)
    ax.invert_yaxis()
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    plt.tight_layout()
    plt.show()


def plot_distribution_comparison(
    by_algorithm: dict[str, list[int]],
    metric_name: str = "nodes expanded",
    title: Optional[str] = None,
) -> None:
    """Box plots of one metric across algorithms over many instances.

    Args:
        by_algorithm: Mapping from algorithm name to a list of metric values
            (one entry per puzzle instance).
        metric_name: Y-axis label (just a human-readable string).
        title: Optional plot title.
    """
    names = list(by_algorithm.keys())
    data = [by_algorithm[n] for n in names]
    fig, ax = plt.subplots(figsize=(2 + 1.4 * len(names), 4.5))
    bp = ax.boxplot(data, tick_labels=names, patch_artist=True,
                    medianprops={"color": "white", "linewidth": 1.8})
    colours = [MONASH_BLUE, UTIL_ORANGE, FOREST, BLUEBERRY, GREY_1]
    for patch, colour in zip(bp["boxes"], colours):
        patch.set_facecolor(colour)
        patch.set_edgecolor(colour)
    ax.set_ylabel(metric_name, color=MONASH_BLUE)
    if title:
        ax.set_title(title, color=MONASH_BLUE)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    plt.tight_layout()
    plt.show()


# =============================================================
# Self-test (manual)
# =============================================================

if __name__ == "__main__":
    # Small smoke test
    p = parse_state("123450678")
    assert p.neighbours()
    assert is_solvable(p)
    print(f"OK: parse_state, neighbours, is_solvable on {p}")

    # Solvability check: a known unsolvable permutation has odd inversions
    # State (1,2,3,4,5,6,8,7,0) has one inversion (8,7) -> not solvable.
    unsolv = Puzzle8((1, 2, 3, 4, 5, 6, 8, 7, 0))
    assert not is_solvable(unsolv), "expected unsolvable"
    print(f"OK: is_solvable False on {unsolv}")

    # BFS on a 4-shuffle puzzle should find a short path quickly
    start = random_puzzle(n_shuffles=4, seed=0)
    result = bfs_search(start)
    print(f"BFS on {start}: path length {len(result.path) - 1}, "
          f"nodes {result.nodes_expanded}, max frontier {result.max_frontier}")
