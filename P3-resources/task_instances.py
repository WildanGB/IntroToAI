"""
Personalised task instance generator for P3 (Greedy Best-First Search) — STUDENT-FACING.

Each student gets a personalised heuristic h(n) over the same 7-node hand-trace
graph. Topology is fixed; only h(n) values differ per student. The heuristic
bands are designed so that:

  - h(A) is HIGH  -> greedy avoids the S-A-G shortcut (BFS finds it, greedy
                    does not).
  - h(B), h(D), h(E) are LOW -> biasing the long route through B-D-E-G.
  - h(C) is sampled across LOW and MEDIUM bands -> main source of inter-
                    student variation. Depending on h(C), greedy either visits
                    C as a side trip (h(C) low) or skips it (h(C) medium).

This deterministic design lets the tutor CLI emit a single canonical
expansion order + path per student.

Uses the shared anti-autocomplete framework at Tasks/_lib/.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

_HERE = Path(__file__).resolve().parent
# task_instances_lib ships beside this file in the OnTrack bundle and lives at
# Tasks/_lib/ in the design repo — add whichever location actually has it.
for _cand in (_HERE, _HERE.parents[1] / "_lib"):
    if (_cand / "task_instances_lib").is_dir():
        if str(_cand) not in sys.path:
            sys.path.insert(0, str(_cand))
        break

from task_instances_lib import seed_for  # noqa: E402

TASK_ID = "P3"

# Fixed topology (same as the original HAND_TRACE_GRAPH in p3_helpers.py).
NODES: List[str] = ["S", "A", "B", "C", "D", "E", "G"]
EDGES: List[Tuple[str, str]] = [
    ("S", "A"), ("S", "B"),
    ("A", "C"), ("A", "G"),
    ("B", "C"), ("B", "D"),
    ("C", "D"),
    ("D", "E"),
    ("E", "G"),
]


def _adjacency() -> Dict[str, List[str]]:
    """Build sorted adjacency list from EDGES. Sorted neighbours give
    deterministic neighbour-iteration order for the reference greedy."""
    adj: Dict[str, List[str]] = {n: [] for n in NODES}
    for u, v in EDGES:
        adj[u].append(v)
        adj[v].append(u)
    for n in adj:
        adj[n].sort()
    return adj


@dataclass
class TraceInstance:
    """A personalised hand-trace instance for greedy best-first search."""

    student_id: str
    graph: Dict[str, List[str]]
    heuristic: Dict[str, float]
    start: str = "S"
    goal: str = "G"

    def describe(self) -> str:
        lines = [
            f"Trace instance for student {self.student_id}",
            f"  start = {self.start}, goal = {self.goal}",
            "  graph (adjacency list):",
        ]
        for n in NODES:
            lines.append(f"    {n} -> {self.graph[n]}")
        lines.append("  heuristic h(n) (estimated distance to goal):")
        for n in NODES:
            lines.append(f"    h({n}) = {self.heuristic[n]}")
        return "\n".join(lines)


def generate_trace_instance(student_id: str) -> TraceInstance:
    """Generate a hand-trace instance personalised to `student_id`.

    Heuristic bands:
      - A: [5, 6, 7]     HIGH  (blocks S-A-G shortcut for greedy)
      - B: [2, 3]        LOW
      - C: [1, 2, 4, 5]  MIXED (main inter-student variation)
      - D: [1, 2]        LOW
      - E: [1, 2]        LOW
      - G: 0
      - S: max(h) + 1    (ensures S looks "farther" than its neighbours)
    """
    rng = seed_for(TASK_ID, "trace", student_id)
    graph = _adjacency()

    bands = {
        "A": [5, 6, 7],
        "B": [2, 3],
        "C": [1, 2, 4, 5],
        "D": [1, 2],
        "E": [1, 2],
    }
    h: Dict[str, float] = {n: float(rng.choice(bands[n])) for n in bands}
    h["G"] = 0.0
    h["S"] = float(max(h.values()) + 1)

    return TraceInstance(student_id=str(student_id), graph=graph, heuristic=h)


def generate_instance(student_id: str) -> Dict[str, object]:
    """Top-level entry point. Returns `{"trace": TraceInstance}`."""
    return {"trace": generate_trace_instance(student_id)}


if __name__ == "__main__":
    sid = sys.argv[1] if len(sys.argv) > 1 else "demo123"
    inst = generate_instance(sid)
    print(inst["trace"].describe())
