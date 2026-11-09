"""Grid-related generator helpers used by multiple search-tier P-tasks."""

from __future__ import annotations

import random
from typing import List, Set, Tuple

Cell = Tuple[int, int]


def is_grid_reachable(rows: int, cols: int, start: Cell, goal: Cell,
                      obstacles: Set[Cell]) -> bool:
    """BFS connectivity check on a 4-connected grid with obstacles."""
    if start in obstacles or goal in obstacles:
        return False
    if start == goal:
        return True
    seen = {start}
    frontier: List[Cell] = [start]
    while frontier:
        r, c = frontier.pop(0)
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            nb = (nr, nc)
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            if nb in obstacles or nb in seen:
                continue
            if nb == goal:
                return True
            seen.add(nb)
            frontier.append(nb)
    return False


def sample_reachable_grid(rng: random.Random, rows: int, cols: int,
                          n_obstacles: int, start: Cell, goal: Cell,
                          max_attempts: int = 200) -> List[Cell]:
    """Sample obstacle positions, rejecting layouts that disconnect start/goal.

    Returns a sorted list of (row, col) obstacles. Raises RuntimeError if
    no valid layout is found within `max_attempts` (which usually means
    `n_obstacles` is too high for the grid size).
    """
    candidates = [(r, c) for r in range(rows) for c in range(cols)
                  if (r, c) != start and (r, c) != goal]
    for _ in range(max_attempts):
        rng.shuffle(candidates)
        obstacles = candidates[:n_obstacles]
        if is_grid_reachable(rows, cols, start, goal, set(obstacles)):
            return sorted(obstacles)
    raise RuntimeError(
        f"No reachable layout found in {max_attempts} attempts "
        f"(rows={rows}, cols={cols}, n_obstacles={n_obstacles}). "
        f"Lower the obstacle count or enlarge the grid."
    )
