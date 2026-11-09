"""
P3 Helper Classes — Greedy Best-First Search on a GridWorld

FIT1061 Introduction to Artificial Intelligence

This module provides the data structures you need for P3.
You do NOT need to modify this file. Import it in your notebook:

    from p3_helpers import PriorityQueue, Queue, GridWorld, reconstruct_path

Classes:
    PriorityQueue — A queue where items are dequeued in order of priority (lowest first)
    Queue         — First-In, First-Out (FIFO) data structure (from P2)
    GridWorld     — A 2D grid with obstacles
"""

import heapq
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# ---------------------------------------------------------------------------
# PriorityQueue
# ---------------------------------------------------------------------------

class PriorityQueue:
    """A priority queue where the item with the lowest priority is dequeued first.

    This wraps Python's heapq module. You don't need to understand
    how heapq works — just use push, pop, and is_empty.

    Usage:
        pq = PriorityQueue()
        pq.push("A", priority=5)
        pq.push("B", priority=2)
        pq.push("C", priority=8)
        print(pq.pop())   # "B" (lowest priority = 2)
        print(pq.pop())   # "A" (next lowest = 5)
    """

    def __init__(self):
        self._items = []
        self._counter = 0  # Tie-breaker for equal priorities

    def push(self, item, priority):
        """Add an item with a given priority (lower = higher priority).

        Args:
            item: The item to add.
            priority: A number. Lower values are dequeued first.
        """
        heapq.heappush(self._items, (priority, self._counter, item))
        self._counter += 1

    def pop(self):
        """Remove and return the item with the lowest priority.

        Raises IndexError if the queue is empty.
        """
        if self.is_empty():
            raise IndexError("pop from an empty priority queue")
        priority, _counter, item = heapq.heappop(self._items)
        return item

    def is_empty(self):
        """Return True if the priority queue has no items."""
        return len(self._items) == 0

    def size(self):
        """Return the number of items in the priority queue."""
        return len(self._items)

    def __repr__(self):
        items_str = [(p, item) for p, _c, item in sorted(self._items)]
        return f"PriorityQueue({items_str})"


# ---------------------------------------------------------------------------
# Queue (FIFO) — same as P2, included for BFS comparison
# ---------------------------------------------------------------------------

class Queue:
    """A simple FIFO queue (same as P2)."""

    def __init__(self):
        self._items = []

    def enqueue(self, item):
        self._items.append(item)

    def dequeue(self):
        if self.is_empty():
            raise IndexError("dequeue from an empty queue")
        return self._items.pop(0)

    def is_empty(self):
        return len(self._items) == 0

    def size(self):
        return len(self._items)

    def __repr__(self):
        return f"Queue({self._items})"


# ---------------------------------------------------------------------------
# GridWorld — same as P2
# ---------------------------------------------------------------------------

class GridWorld:
    """A 2D grid world with optional obstacles.

    Same as P2. Each cell (row, col) is a node. Neighbours are the
    4-connected walkable cells (up, down, left, right).
    """

    def __init__(self, rows, cols, obstacles=None):
        self.rows = rows
        self.cols = cols
        self.obstacles = obstacles if obstacles is not None else set()
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

    def visualise(self, path=None, explored=None, start=None, goal=None,
                  title="GridWorld"):
        grid = np.ones((self.rows, self.cols, 3))

        for (r, c) in self.obstacles:
            grid[r, c] = [0.3, 0.3, 0.3]

        if explored:
            for (r, c) in explored:
                if (r, c) not in self.obstacles:
                    grid[r, c] = [0.78, 0.89, 1.0]

        if path:
            for (r, c) in path:
                grid[r, c] = [1.0, 0.65, 0.0]

        if start:
            grid[start[0], start[1]] = [0.2, 0.8, 0.2]
        if goal:
            grid[goal[0], goal[1]] = [0.9, 0.2, 0.2]

        fig, ax = plt.subplots(1, 1, figsize=(max(self.cols * 0.8, 4),
                                               max(self.rows * 0.8, 4)))
        ax.imshow(grid, interpolation='nearest')

        for x in range(self.cols + 1):
            ax.axvline(x - 0.5, color='grey', linewidth=0.5)
        for y in range(self.rows + 1):
            ax.axhline(y - 0.5, color='grey', linewidth=0.5)

        if path:
            for idx, (r, c) in enumerate(path):
                ax.text(c, r, str(idx), ha='center', va='center',
                        fontsize=8, fontweight='bold')

        ax.set_title(title)
        ax.set_xticks(range(self.cols))
        ax.set_yticks(range(self.rows))
        ax.set_xlabel("Column")
        ax.set_ylabel("Row")

        legend_items = []
        if start:
            legend_items.append(mpatches.Patch(color=[0.2, 0.8, 0.2], label='Start'))
        if goal:
            legend_items.append(mpatches.Patch(color=[0.9, 0.2, 0.2], label='Goal'))
        if path:
            legend_items.append(mpatches.Patch(color=[1.0, 0.65, 0.0], label='Path'))
        if explored:
            legend_items.append(mpatches.Patch(color=[0.78, 0.89, 1.0], label='Explored'))
        legend_items.append(mpatches.Patch(color=[0.3, 0.3, 0.3], label='Obstacle'))
        if legend_items:
            ax.legend(handles=legend_items, loc='upper left',
                      bbox_to_anchor=(1.02, 1), fontsize=8)

        plt.tight_layout()
        plt.show()

    def __repr__(self):
        return (f"GridWorld(rows={self.rows}, cols={self.cols}, "
                f"obstacles={len(self.obstacles)})")


# ---------------------------------------------------------------------------
# Hand-trace graph for P3.1
# ---------------------------------------------------------------------------

# A small 7-node graph with heuristic values for hand-tracing greedy best-first.
# Same structure as P2 so students can compare traces directly.
HAND_TRACE_GRAPH = {
    "S": ["A", "B"],
    "A": ["S", "C", "G"],
    "B": ["S", "C", "D"],
    "C": ["A", "B", "D"],
    "D": ["B", "C", "E"],
    "E": ["D", "G"],
    "G": ["A", "E"]
}

# Heuristic: estimated distance from each node to G.
# Hand-chosen so greedy takes the LONGER path (S→B→D→E→G, 4 edges)
# while BFS finds the short path (S→A→G, 2 edges).
# B, D, E have low heuristics (look close); A has high heuristic (looks far).
HEURISTIC_TO_G = {
    "S": 6,
    "A": 5,
    "B": 2,
    "C": 3,
    "D": 1.5,
    "E": 1,
    "G": 0
}


def draw_hand_trace_graph_with_heuristics(graph=None, heuristic=None):
    """Draw the hand-trace graph with heuristic values annotated.

    Args:
        graph: Adjacency-list dict. Defaults to HAND_TRACE_GRAPH (module
            constant) for backward compatibility. Pass a per-student graph
            (e.g. from `task_instances.generate_instance(...)["trace"].graph`)
            for personalised rendering.
        heuristic: Dict of h(n) values. Defaults to HEURISTIC_TO_G.
    """
    if graph is None:
        graph = HAND_TRACE_GRAPH
    if heuristic is None:
        heuristic = HEURISTIC_TO_G

    pos = {
        "S": (0, 1),
        "A": (1, 2),
        "B": (1, 0),
        "C": (2, 2.5),
        "D": (2, 1),
        "E": (2, -0.5),
        "G": (3, 1.5),
    }

    fig, ax = plt.subplots(figsize=(8, 5.5))

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

    # Draw nodes with heuristic labels
    for node, (x, y) in pos.items():
        color = '#4CAF50' if node == 'S' else '#F44336' if node == 'G' else '#2196F3'
        ax.plot(x, y, 'o', markersize=30, color=color, zorder=2)
        ax.text(x, y, node, ha='center', va='center',
                fontsize=14, fontweight='bold', color='white', zorder=3)

        # Heuristic value label
        h = heuristic[node]
        ax.text(x + 0.3, y + 0.25, f"h={h}",
                fontsize=9, color='#555555', style='italic', zorder=3,
                bbox=dict(boxstyle='round,pad=0.15', facecolor='#FFF9C4',
                         edgecolor='#F9A825', linewidth=0.8))

    ax.set_xlim(-0.8, 4.0)
    ax.set_ylim(-1, 3.2)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title("P3.1 Hand-Trace Graph: Greedy Best-First from S to G\n"
                 "(h = heuristic estimate of distance to G)",
                 fontsize=12)

    legend_items = [
        mpatches.Patch(color='#4CAF50', label='Start (S)'),
        mpatches.Patch(color='#F44336', label='Goal (G)'),
        mpatches.Patch(color='#2196F3', label='Other nodes'),
        mpatches.Patch(color='#FFF9C4', label='h = heuristic to G'),
    ]
    ax.legend(handles=legend_items, loc='lower right', fontsize=9)

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Path reconstruction (same as P2)
# ---------------------------------------------------------------------------

def reconstruct_path(came_from, start, goal):
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
    # Demo priority queue
    pq = PriorityQueue()
    pq.push("far", priority=10)
    pq.push("close", priority=2)
    pq.push("medium", priority=5)
    print(f"Priority queue: {pq}")
    print(f"Pop: {pq.pop()}")  # "close"
    print(f"Pop: {pq.pop()}")  # "medium"

    # Show hand-trace graph
    draw_hand_trace_graph_with_heuristics()
