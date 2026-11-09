"""
P2 Helper Classes — BFS on a GridWorld

FIT1061 Introduction to Artificial Intelligence

This module provides the data structures you need for P2.
You do NOT need to modify this file. Import it in your notebook:

    from p2_helpers import Queue, Stack, GridWorld

Classes:
    Queue       — First-In, First-Out (FIFO) data structure
    Stack       — Last-In, First-Out (LIFO) data structure
    GridWorld   — A 2D grid with obstacles, represented as an adjacency dict
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# ---------------------------------------------------------------------------
# Queue (FIFO)
# ---------------------------------------------------------------------------

class Queue:
    """A simple FIFO queue.

    Usage:
        q = Queue()
        q.enqueue("A")
        q.enqueue("B")
        print(q.dequeue())   # "A"
        print(q.is_empty())  # False
    """

    def __init__(self):
        self._items = []

    def enqueue(self, item):
        """Add an item to the back of the queue."""
        self._items.append(item)

    def dequeue(self):
        """Remove and return the item at the front of the queue.

        Raises IndexError if the queue is empty.
        """
        if self.is_empty():
            raise IndexError("dequeue from an empty queue")
        return self._items.pop(0)

    def is_empty(self):
        """Return True if the queue has no items."""
        return len(self._items) == 0

    def size(self):
        """Return the number of items in the queue."""
        return len(self._items)

    def __repr__(self):
        return f"Queue({self._items})"


# ---------------------------------------------------------------------------
# Stack (LIFO)
# ---------------------------------------------------------------------------

class Stack:
    """A simple LIFO stack.

    Usage:
        s = Stack()
        s.push("A")
        s.push("B")
        print(s.pop())       # "B"
        print(s.is_empty())  # False
    """

    def __init__(self):
        self._items = []

    def push(self, item):
        """Add an item to the top of the stack."""
        self._items.append(item)

    def pop(self):
        """Remove and return the item at the top of the stack.

        Raises IndexError if the stack is empty.
        """
        if self.is_empty():
            raise IndexError("pop from an empty stack")
        return self._items.pop()

    def is_empty(self):
        """Return True if the stack has no items."""
        return len(self._items) == 0

    def size(self):
        """Return the number of items in the stack."""
        return len(self._items)

    def __repr__(self):
        return f"Stack({self._items})"


# ---------------------------------------------------------------------------
# GridWorld
# ---------------------------------------------------------------------------

class GridWorld:
    """A 2D grid world with optional obstacles.

    The grid is represented internally as an adjacency dict:
    each cell (row, col) maps to a list of its walkable neighbours.

    Neighbours are the 4-connected cells (up, down, left, right)
    that are within bounds and not obstacles.

    Usage:
        gw = GridWorld(5, 5, obstacles={(1,1), (2,1), (3,1)})
        print(gw.get_neighbours(0, 0))  # [(0, 1), (1, 0)]
    """

    def __init__(self, rows, cols, obstacles=None):
        """Create a grid world.

        Args:
            rows: Number of rows.
            cols: Number of columns.
            obstacles: A set of (row, col) tuples that are impassable.
        """
        self.rows = rows
        self.cols = cols
        self.obstacles = obstacles if obstacles is not None else set()
        self._graph = self._build_graph()

    def _build_graph(self):
        """Build the adjacency dict for the grid."""
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
        """Return the list of walkable neighbours for a cell.

        Args:
            row: Row index.
            col: Column index.

        Returns:
            List of (row, col) tuples.
        """
        return self._graph.get((row, col), [])

    def is_walkable(self, row, col):
        """Return True if the cell is within bounds and not an obstacle."""
        return (row, col) in self._graph

    def visualise(self, path=None, explored=None, start=None, goal=None,
                  title="GridWorld"):
        """Visualise the grid, optionally showing a path and explored cells.

        Args:
            path: List of (row, col) tuples forming the path.
            explored: Set of (row, col) tuples that were explored.
            start: (row, col) of the start cell.
            goal: (row, col) of the goal cell.
            title: Plot title.
        """
        grid = np.ones((self.rows, self.cols, 3))  # white

        # Obstacles: dark grey
        for (r, c) in self.obstacles:
            grid[r, c] = [0.3, 0.3, 0.3]

        # Explored cells: light blue
        if explored:
            for (r, c) in explored:
                if (r, c) not in self.obstacles:
                    grid[r, c] = [0.78, 0.89, 1.0]

        # Path: orange
        if path:
            for (r, c) in path:
                grid[r, c] = [1.0, 0.65, 0.0]

        # Start: green
        if start:
            grid[start[0], start[1]] = [0.2, 0.8, 0.2]

        # Goal: red
        if goal:
            grid[goal[0], goal[1]] = [0.9, 0.2, 0.2]

        fig, ax = plt.subplots(1, 1, figsize=(max(self.cols * 0.8, 4),
                                               max(self.rows * 0.8, 4)))
        ax.imshow(grid, interpolation='nearest')

        # Grid lines
        for x in range(self.cols + 1):
            ax.axvline(x - 0.5, color='grey', linewidth=0.5)
        for y in range(self.rows + 1):
            ax.axhline(y - 0.5, color='grey', linewidth=0.5)

        # Labels on path
        if path:
            for idx, (r, c) in enumerate(path):
                ax.text(c, r, str(idx), ha='center', va='center',
                        fontsize=8, fontweight='bold')

        ax.set_title(title)
        ax.set_xticks(range(self.cols))
        ax.set_yticks(range(self.rows))
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")

        # Legend
        legend_items = []
        if start:
            legend_items.append(mpatches.Patch(color=[0.2, 0.8, 0.2],
                                               label='Start'))
        if goal:
            legend_items.append(mpatches.Patch(color=[0.9, 0.2, 0.2],
                                               label='Goal'))
        if path:
            legend_items.append(mpatches.Patch(color=[1.0, 0.65, 0.0],
                                               label='Path'))
        if explored:
            legend_items.append(mpatches.Patch(color=[0.78, 0.89, 1.0],
                                               label='Explored'))
        legend_items.append(mpatches.Patch(color=[0.3, 0.3, 0.3],
                                           label='Obstacle'))
        if legend_items:
            ax.legend(handles=legend_items, loc='upper left',
                      bbox_to_anchor=(1.02, 1), fontsize=8)

        plt.tight_layout()
        plt.show()

    def __repr__(self):
        return (f"GridWorld(rows={self.rows}, cols={self.cols}, "
                f"obstacles={len(self.obstacles)})")


# ---------------------------------------------------------------------------
# Hand-trace graph (for P2.1)
# ---------------------------------------------------------------------------

# A small 7-node undirected graph for hand-tracing BFS in P2.1.
# Intentionally DIFFERENT from the 7-node graph used in the W2 lecture
# worked-example (which uses S, A, B, C, D, E, G — all caps). Here the
# intermediate nodes are lowercase a, b, c, d, e so students can't pattern-
# match the lecturer's trace and must run BFS in their own head.
# Same structural difficulty: 7 nodes, multiple cycles, S→G shortest path 4 nodes.
HAND_TRACE_GRAPH = {
    "S": ["a", "b"],
    "a": ["S", "c"],
    "b": ["S", "c", "d"],
    "c": ["a", "b", "G"],
    "d": ["b", "e"],
    "e": ["d", "G"],
    "G": ["c", "e"],
}


def draw_hand_trace_graph(graph=None, start="S", goal="G"):
    """Draw the hand-trace graph for P2.1.

    Args:
        graph: Adjacency-list dict. Defaults to HAND_TRACE_GRAPH (module
            constant) for backward compatibility. Pass a per-student graph
            (e.g. from `task_instances.generate_instance(...)["trace"].graph`)
            for personalised rendering.
        start: Node label to highlight green. Defaults to "S".
        goal: Node label to highlight red. Defaults to "G".
    """
    if graph is None:
        graph = HAND_TRACE_GRAPH

    # Node positions (hand-tuned for clarity)
    pos = {
        "S": (0, 1),
        "a": (1.2, 2),
        "b": (1.2, 0),
        "c": (2.5, 1.5),
        "d": (2.5, -0.3),
        "e": (3.5, 0.5),
        "G": (3.8, 2),
    }

    fig, ax = plt.subplots(figsize=(7, 5))

    # Draw edges
    drawn = set()
    for node, neighbours in graph.items():
        for neighbour in neighbours:
            edge = tuple(sorted([node, neighbour]))
            if edge not in drawn:
                x = [pos[node][0], pos[neighbour][0]]
                y = [pos[node][1], pos[neighbour][1]]
                ax.plot(x, y, 'k-', linewidth=1.5, zorder=1)
                drawn.add(edge)

    # Draw nodes — colour by start/goal/other
    for node, (x, y) in pos.items():
        color = ('#4CAF50' if node == start
                 else '#F44336' if node == goal
                 else '#2196F3')
        ax.plot(x, y, 'o', markersize=30, color=color, zorder=2)
        ax.text(x, y, node, ha='center', va='center',
                fontsize=14, fontweight='bold', color='white', zorder=3)

    ax.set_xlim(-0.5, 4.4)
    ax.set_ylim(-1, 3.0)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(f"P2.1 Hand-Trace Graph: BFS/DFS from {start} to {goal}",
                 fontsize=13)

    # Legend
    legend_items = [
        mpatches.Patch(color='#4CAF50', label=f'Start ({start})'),
        mpatches.Patch(color='#F44336', label=f'Goal ({goal})'),
        mpatches.Patch(color='#2196F3', label='Other nodes'),
    ]
    ax.legend(handles=legend_items, loc='lower right', fontsize=9)

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Dead-end grid (for P2.4 — BFS vs DFS comparison)
# ---------------------------------------------------------------------------

# A 7×7 maze with a deep dead-end branch.
# From S=(0,0), the goal G=(6,6) is reachable via a shortest path of 13 cells.
# DFS (using a Stack with the helper's neighbour order) dives into a deep
# dead-end branch first, so its discovered path is 19 cells — substantially
# longer than the BFS optimum. Canonical numbers (used by the W2 slides):
#   BFS: path 13, nodes expanded 28
#   DFS: path 19, nodes expanded 32
DEAD_END_OBSTACLES = {
    (0, 3),
    (1, 1), (1, 3), (1, 5),
    (2, 1), (2, 3), (2, 5),
    (3, 1), (3, 5),
    (4, 1), (4, 3), (4, 4), (4, 5),
    (5, 1), (5, 3),
}


def make_dead_end_grid():
    """Return a 7×7 GridWorld that exposes BFS vs DFS exploration differences."""
    return GridWorld(7, 7, obstacles=DEAD_END_OBSTACLES)


# ---------------------------------------------------------------------------
# Path reconstruction (provided to students)
# ---------------------------------------------------------------------------

def reconstruct_path(came_from, start, goal):
    """Reconstruct the path from start to goal using the came_from dict.

    Args:
        came_from: Dict mapping each node to the node it was reached from.
        start: The start node.
        goal: The goal node.

    Returns:
        A list of nodes from start to goal, or None if no path exists.
    """
    if goal not in came_from:
        return None

    path = []
    current = goal
    while current is not None:
        path.append(current)
        current = came_from[current]
    path.reverse()
    return path


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Quick demo: create a small grid and show it
    gw = GridWorld(5, 5, obstacles={(1, 1), (2, 1), (3, 1), (1, 3), (2, 3)})
    gw.visualise(start=(0, 0), goal=(4, 4), title="Demo: 5×5 GridWorld")
    print(f"Neighbours of (0,0): {gw.get_neighbours(0, 0)}")
    print(f"Neighbours of (2,2): {gw.get_neighbours(2, 2)}")

    # Show hand-trace graph
    draw_hand_trace_graph()
