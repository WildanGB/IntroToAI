"""
P5 Helper Functions — Humanitarian Routing (Search Consolidation)

FIT1061 Introduction to Artificial Intelligence

This module provides the scenario and search algorithms for P5.
You do NOT need to modify this file. Import it in your notebook:

    from p5_helpers import (HumanitarianGrid, create_disaster_zone,
                            bfs_search, dfs_search, greedy_best_first_search,
                            manhattan_distance, euclidean_distance,
                            reconstruct_path,
                            TARGETS_4, visualise_targets, plot_pareto)

The humanitarian routing scenario: after a disaster, a relief vehicle
must navigate from a supply depot to a target area through a grid of
city blocks. Some roads are blocked by rubble (obstacles).
"""

import math
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from collections import deque
import heapq


# ---------------------------------------------------------------------------
# Humanitarian Routing Grid
# ---------------------------------------------------------------------------

class HumanitarianGrid:
    """A 2D grid representing a disaster zone.

    Each cell (row, col) is a city block. Obstacles represent locations
    blocked by rubble or marked unsafe. The relief vehicle can move up,
    down, left, or right.
    """

    def __init__(self, rows, cols, obstacles=None, depot=None, target=None):
        self.rows = rows
        self.cols = cols
        self.obstacles = obstacles if obstacles is not None else set()
        self.depot = depot or (0, 0)
        self.target = target or (rows - 1, cols - 1)
        self._graph = self._build_graph()

    def _build_graph(self):
        graph = {}
        for r in range(self.rows):
            for c in range(self.cols):
                if (r, c) in self.obstacles:
                    continue
                neighbours = []
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if (0 <= nr < self.rows and 0 <= nc < self.cols
                            and (nr, nc) not in self.obstacles):
                        neighbours.append((nr, nc))
                graph[(r, c)] = neighbours
        return graph

    def get_neighbours(self, row, col):
        return self._graph.get((row, col), [])

    def is_walkable(self, row, col):
        return (row, col) in self._graph

    def visualise(self, path=None, explored=None, title="Disaster Zone",
                  show_legend=True):
        """Draw the grid with optional path and explored cells."""
        grid = np.ones((self.rows, self.cols, 3))

        # Obstacles (rubble)
        for (r, c) in self.obstacles:
            grid[r, c] = [0.3, 0.3, 0.3]

        # Explored cells
        if explored:
            for (r, c) in explored:
                if (r, c) not in self.obstacles:
                    grid[r, c] = [0.78, 0.89, 1.0]

        # Path
        if path:
            for (r, c) in path:
                grid[r, c] = [1.0, 0.65, 0.0]

        # Depot and target
        grid[self.depot[0], self.depot[1]] = [0.2, 0.8, 0.2]
        grid[self.target[0], self.target[1]] = [0.9, 0.2, 0.2]

        fig, ax = plt.subplots(1, 1, figsize=(max(self.cols * 0.6, 5),
                                               max(self.rows * 0.6, 5)))
        ax.imshow(grid, interpolation='nearest')

        for x in range(self.cols + 1):
            ax.axvline(x - 0.5, color='grey', linewidth=0.5)
        for y in range(self.rows + 1):
            ax.axhline(y - 0.5, color='grey', linewidth=0.5)

        if path:
            for idx, (r, c) in enumerate(path):
                ax.text(c, r, str(idx), ha='center', va='center',
                        fontsize=7, fontweight='bold')

        # Labels
        dr, dc = self.depot
        ax.text(dc, dr, 'D', ha='center', va='center', fontsize=14,
                fontweight='bold', color='white')
        tr, tc = self.target
        ax.text(tc, tr, 'T', ha='center', va='center', fontsize=14,
                fontweight='bold', color='white')

        ax.set_title(title)
        ax.set_xticks(range(self.cols))
        ax.set_yticks(range(self.rows))
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")

        if show_legend:
            legend_items = [
                mpatches.Patch(color=[0.2, 0.8, 0.2], label='D = Supply Depot'),
                mpatches.Patch(color=[0.9, 0.2, 0.2], label='T = Target Area'),
            ]
            if path:
                legend_items.append(
                    mpatches.Patch(color=[1.0, 0.65, 0.0], label='Route'))
            if explored:
                legend_items.append(
                    mpatches.Patch(color=[0.78, 0.89, 1.0], label='Explored'))
            legend_items.append(
                mpatches.Patch(color=[0.3, 0.3, 0.3], label='Blocked or Unsafe'))
            ax.legend(handles=legend_items, loc='upper left',
                      bbox_to_anchor=(1.02, 1), fontsize=8)

        plt.tight_layout()
        plt.show()

    def __repr__(self):
        return (f"HumanitarianGrid(rows={self.rows}, cols={self.cols}, "
                f"obstacles={len(self.obstacles)})")


# ---------------------------------------------------------------------------
# The Scenario
# ---------------------------------------------------------------------------

def create_disaster_zone():
    """Create the humanitarian routing scenario.

    Returns a 16x16 grid representing a city district after an earthquake.
    Damage appears in connected clusters and along closed road segments,
    rather than as isolated obstacles. Limited crossings into the eastern
    district force detours. This is designed so that greedy best-first search
    follows an initially promising approach and finds a longer route than BFS.
    """
    obstacles = set()

    def block_rectangle(top, left, bottom, right):
        """Block a rectangular collapse zone, inclusive of all boundaries."""
        for row in range(top, bottom + 1):
            for col in range(left, right + 1):
                obstacles.add((row, col))

    def block_horizontal(row, left, right):
        """Block a contiguous east-west road segment."""
        for col in range(left, right + 1):
            obstacles.add((row, col))

    def block_vertical(col, top, bottom):
        """Block a contiguous north-south road segment."""
        for row in range(top, bottom + 1):
            obstacles.add((row, col))

    # Collapsed residential and commercial blocks.
    block_rectangle(2, 2, 3, 4)
    block_rectangle(5, 1, 6, 2)
    block_rectangle(8, 4, 9, 5)
    block_rectangle(11, 2, 12, 4)

    # Rubble across local roads around the damaged central district.
    block_horizontal(4, 7, 10)
    block_horizontal(5, 7, 12)
    block_horizontal(7, 7, 11)
    block_horizontal(10, 7, 10)
    block_horizontal(13, 5, 11)
    block_horizontal(6, 9, 12)

    block_horizontal(9, 0, 2)
    block_horizontal(9, 10, 13)
    block_vertical(7, 11, 12)




    # A signed diversion through the northern district avoids unsafe roads.
    # Its staggered closures mean the shortest route is not a straight line.
    block_vertical(5, 0, 3)
    block_vertical(9, 3, 3)
    block_vertical(12, 0, 1)

    # The eastern access road is unsafe. Its limited crossings make a
    # direct-looking approach to Hillside misleading.
    block_vertical(14, 0, 2)
    block_vertical(14, 5, 8)
    block_vertical(14, 11, 15)

    return HumanitarianGrid(
        rows=16, cols=16,
        obstacles=obstacles,
        depot=(0, 0),
        target=(15, 15)
    )


# ---------------------------------------------------------------------------
# Four-Target Scenario (for ethical analysis — Part 6)
# ---------------------------------------------------------------------------
#
# 4 areas across the trade space (path cost vs population served). Each has a
# position (relative to the depot) and a population it would serve. Computing
# each target's path cost, and working out which targets are dominated, is the
# task — see Part 6.
#
#   Riverside Clinic   — close to the depot, small population.
#   Suburb Shelter     — middle distance, mid-sized population.
#   Hillside Shelter   — far from the depot, large population.
#   Eastside School    — middle distance, smallest population.

TARGETS_4 = {
    "Riverside Clinic": {
        "position": (3, 7),
        "population": 50,
        "description": "Small clinic, close to depot",
    },
    "Suburb Shelter": {
        "position": (9, 7),
        "population": 300,
        "description": "Medium shelter, middle distance",
    },
    "Hillside Shelter": {
        "position": (15, 15),
        "population": 800,
        "description": "Large shelter, far from depot",
    },
    "Eastside School": {
        "position": (7, 13),
        "population": 30,
        "description": "Tiny school, middle distance",
    },
}


def visualise_targets(grid, targets=TARGETS_4, title="Four Areas Need Help"):
    """Draw the grid with the four target areas marked.

    Args:
        grid: HumanitarianGrid instance.
        targets: dict like TARGETS_4 — {name: {"position": (r, c), "population": int, ...}}
        title: plot title.
    """
    fig, ax = plt.subplots(figsize=(max(grid.cols * 0.6, 5),
                                    max(grid.rows * 0.6, 5)))

    grid_img = np.ones((grid.rows, grid.cols, 3))
    for (r, c) in grid.obstacles:
        grid_img[r, c] = [0.3, 0.3, 0.3]

    grid_img[grid.depot[0], grid.depot[1]] = [0.2, 0.8, 0.2]
    for name, info in targets.items():
        r, c = info["position"]
        grid_img[r, c] = [0.9, 0.2, 0.2]

    ax.imshow(grid_img, interpolation='nearest')

    for x in range(grid.cols + 1):
        ax.axvline(x - 0.5, color='grey', linewidth=0.5)
    for y in range(grid.rows + 1):
        ax.axhline(y - 0.5, color='grey', linewidth=0.5)

    dr, dc = grid.depot
    ax.text(dc, dr, 'D', ha='center', va='center', fontsize=14,
            fontweight='bold', color='white')

    for name, info in targets.items():
        r, c = info["position"]
        pop = info["population"]
        short_name = name.split()[0]
        ax.text(c, r, f'{short_name}\n({pop})', ha='center', va='center',
                fontsize=6.5, fontweight='bold', color='white')

    ax.set_title(title)
    ax.set_xticks(range(grid.cols))
    ax.set_yticks(range(grid.rows))
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")

    legend_items = [
        mpatches.Patch(color=[0.2, 0.8, 0.2], label='D = Supply Depot'),
        mpatches.Patch(color=[0.9, 0.2, 0.2], label='Target Areas'),
        mpatches.Patch(color=[0.3, 0.3, 0.3], label='Blocked or Unsafe'),
    ]
    ax.legend(handles=legend_items, loc='upper left',
              bbox_to_anchor=(1.02, 1), fontsize=8)
    plt.tight_layout()
    plt.show()


def plot_pareto(target_names, costs, populations, frontier_indices,
                title="Path cost vs population served"):
    """Scatter plot of (path_cost, population) per target with Pareto frontier highlighted.

    The frontier is drawn as a step line connecting the Pareto-optimal points
    in cost-ascending order. Dominated points are greyed out.

    Args:
        target_names: list of target name strings (length n).
        costs: list of BFS path costs from depot (length n, lower = better).
        populations: list of population counts at each target (length n, higher = better).
        frontier_indices: list/set of indices that lie on the Pareto frontier.
        title: plot title.
    """
    MONASH_BLUE = "#006DAE"
    UTIL_RED = "#EA001F"
    GREY_2 = "#969696"

    fig, ax = plt.subplots(figsize=(8, 6))

    frontier_set = set(frontier_indices)
    for i, (name, c, p) in enumerate(zip(target_names, costs, populations)):
        is_frontier = i in frontier_set
        colour = MONASH_BLUE if is_frontier else GREY_2
        marker = 'o' if is_frontier else 'x'
        size = 140 if is_frontier else 90
        label_suffix = " (Pareto)" if is_frontier else " (dominated)"
        ax.scatter(c, p, s=size, c=colour, marker=marker,
                   edgecolors='black' if is_frontier else GREY_2,
                   linewidths=1.0 if is_frontier else 0.5, zorder=3)
        ax.annotate(f" {name}{label_suffix}", xy=(c, p),
                    xytext=(8, 4), textcoords='offset points',
                    fontsize=9,
                    color='black' if is_frontier else GREY_2)

    if frontier_indices:
        ordered = sorted(frontier_indices, key=lambda i: costs[i])
        fx = [costs[i] for i in ordered]
        fy = [populations[i] for i in ordered]
        # Draw the frontier as a step line: at each Pareto point, you accept
        # higher cost for higher population.
        ax.plot(fx, fy, '--', color=MONASH_BLUE, linewidth=1.5, alpha=0.5,
                label="Pareto frontier", zorder=2)

    ax.set_xlabel("Path cost (steps from depot — lower is better)", fontsize=11)
    ax.set_ylabel("Population served (higher is better)", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=10)
    plt.tight_layout()
    plt.show()


# Legacy 2-target dict and visualiser — kept for backward compatibility.
# Part 6 of the notebook (post-2026-05-29) uses TARGETS_4 instead.
TARGETS = {
    "Riverside Clinic": {
        "position": (3, 7),
        "population": 50,
        "description": "Small clinic, 50 people, close to depot",
    },
    "Hillside Shelter": {
        "position": (15, 15),
        "population": 800,
        "description": "Large shelter, 800 displaced people, far from depot",
    },
}


def visualise_two_targets(grid, title="Two Areas Need Help"):
    """Draw the grid with both target areas marked."""
    fig, ax = plt.subplots(figsize=(max(grid.cols * 0.6, 5),
                                    max(grid.rows * 0.6, 5)))

    # Draw grid
    grid_img = np.ones((grid.rows, grid.cols, 3))
    for (r, c) in grid.obstacles:
        grid_img[r, c] = [0.3, 0.3, 0.3]

    # Depot
    grid_img[grid.depot[0], grid.depot[1]] = [0.2, 0.8, 0.2]

    # Targets
    for name, info in TARGETS.items():
        r, c = info["position"]
        grid_img[r, c] = [0.9, 0.2, 0.2]

    ax.imshow(grid_img, interpolation='nearest')

    for x in range(grid.cols + 1):
        ax.axvline(x - 0.5, color='grey', linewidth=0.5)
    for y in range(grid.rows + 1):
        ax.axhline(y - 0.5, color='grey', linewidth=0.5)

    # Labels
    dr, dc = grid.depot
    ax.text(dc, dr, 'D', ha='center', va='center', fontsize=14,
            fontweight='bold', color='white')

    for name, info in TARGETS.items():
        r, c = info["position"]
        pop = info["population"]
        short_name = name.split()[0]  # "Riverside" or "Hillside"
        ax.text(c, r, f'{short_name}\n({pop})', ha='center', va='center',
                fontsize=7, fontweight='bold', color='white')

    ax.set_title(title)
    ax.set_xticks(range(grid.cols))
    ax.set_yticks(range(grid.rows))
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")

    legend_items = [
        mpatches.Patch(color=[0.2, 0.8, 0.2], label='D = Supply Depot'),
        mpatches.Patch(color=[0.9, 0.2, 0.2], label='Target Areas'),
        mpatches.Patch(color=[0.3, 0.3, 0.3], label='Blocked or Unsafe'),
    ]
    ax.legend(handles=legend_items, loc='upper left',
              bbox_to_anchor=(1.02, 1), fontsize=8)

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Search Algorithms (provided — same as P2 and P3)
# ---------------------------------------------------------------------------

def reconstruct_path(came_from, start, goal):
    """Reconstruct the path from start to goal using came_from dict."""
    if goal not in came_from:
        return None
    path = []
    current = goal
    while current is not None:
        path.append(current)
        current = came_from[current]
    path.reverse()
    return path


def bfs_search(grid, start, goal):
    """BFS — guaranteed shortest path. Provided for comparison."""
    frontier = deque([start])
    came_from = {start: None}
    nodes_expanded = 0

    while frontier:
        current = frontier.popleft()
        nodes_expanded += 1

        if current == goal:
            break

        row, col = current
        for neighbour in grid.get_neighbours(row, col):
            if neighbour not in came_from:
                frontier.append(neighbour)
                came_from[neighbour] = current

    path = reconstruct_path(came_from, start, goal)
    explored = set(came_from.keys())
    return path, explored, nodes_expanded


def dfs_search(grid, start, goal):
    """DFS — finds a path, but not necessarily the shortest path."""
    frontier = [start]
    came_from = {start: None}
    nodes_expanded = 0

    while frontier:
        current = frontier.pop()
        nodes_expanded += 1

        if current == goal:
            break

        row, col = current
        for neighbour in grid.get_neighbours(row, col):
            if neighbour not in came_from:
                frontier.append(neighbour)
                came_from[neighbour] = current

    path = reconstruct_path(came_from, start, goal)
    explored = set(came_from.keys())
    return path, explored, nodes_expanded


def greedy_best_first_search(grid, start, goal, heuristic):
    """Greedy best-first search — same as P3."""
    frontier = []
    counter = 0
    heapq.heappush(frontier, (heuristic(start, goal), counter, start))
    counter += 1
    came_from = {start: None}
    nodes_expanded = 0

    while frontier:
        _, _, current = heapq.heappop(frontier)
        nodes_expanded += 1

        if current == goal:
            break

        row, col = current
        for neighbour in grid.get_neighbours(row, col):
            if neighbour not in came_from:
                h = heuristic(neighbour, goal)
                heapq.heappush(frontier, (h, counter, neighbour))
                counter += 1
                came_from[neighbour] = current

    path = reconstruct_path(came_from, start, goal)
    explored = set(came_from.keys())
    return path, explored, nodes_expanded


# ---------------------------------------------------------------------------
# Distance Functions (same as P3)
# ---------------------------------------------------------------------------

def manhattan_distance(a, b):
    """Manhattan distance between two (row, col) positions."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def euclidean_distance(a, b):
    """Euclidean distance between two (row, col) positions."""
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    zone = create_disaster_zone()
    print(zone)
    print(f"Depot: {zone.depot}, Target: {zone.target}")
    zone.visualise(title="Disaster Zone — Humanitarian Routing")
