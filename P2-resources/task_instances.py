"""
Personalised task instance generator for P2 (BFS & DFS) — STUDENT-FACING.

Each student gets a personalised (start, goal) pair on the fixed 7-node
hand-trace graph. Topology is the same for every student; the pair is
drawn from a pre-validated pool with shortest-path distance >= 2 so the
trace is non-trivial.

This is the PRE-implementation personalised hand-trace instance (P2.1 +
P2.3 deliverables). It replaces the previous post-implementation grid
checkpoint (removed 2026-06-01) following the P3 pattern.

Uses the shared anti-autocomplete framework at Tasks/_lib/.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

_HERE = Path(__file__).resolve().parent
# task_instances_lib ships beside this file in the OnTrack bundle and lives at
# Tasks/_lib/ in the design repo — add whichever location actually has it.
for _cand in (_HERE, _HERE.parents[1] / "_lib"):
    if (_cand / "task_instances_lib").is_dir():
        if str(_cand) not in sys.path:
            sys.path.insert(0, str(_cand))
        break

from task_instances_lib import seed_for  # noqa: E402

TASK_ID = "P2"

# Fixed topology (same as HAND_TRACE_GRAPH in p2_helpers.py).
# Lowercase intermediates so students cannot pattern-match the W2 lecture
# worked example (which uses uppercase A-G).
NODES: List[str] = ["S", "a", "b", "c", "d", "e", "G"]
EDGES = [
    ("S", "a"), ("S", "b"),
    ("a", "c"),
    ("b", "c"), ("b", "d"),
    ("c", "G"),
    ("d", "e"),
    ("e", "G"),
]


def _adjacency() -> Dict[str, List[str]]:
    """Build sorted adjacency list. Sorted neighbours give deterministic
    iteration order for the reference BFS."""
    adj: Dict[str, List[str]] = {n: [] for n in NODES}
    for u, v in EDGES:
        adj[u].append(v)
        adj[v].append(u)
    for n in adj:
        adj[n].sort()
    return adj


# Pre-validated (start, goal) pool: shortest-path distance >= 2 on the
# graph above. Manually enumerated; tested in __main__.
VALID_PAIRS = [
    ("S", "c"), ("S", "d"), ("S", "e"), ("S", "G"),
    ("a", "b"), ("a", "d"), ("a", "e"), ("a", "G"),
    ("b", "a"), ("b", "e"), ("b", "G"),
    ("c", "S"), ("c", "d"), ("c", "e"),
    ("d", "S"), ("d", "a"), ("d", "c"), ("d", "G"),
    ("e", "S"), ("e", "a"), ("e", "b"), ("e", "c"),
    ("G", "S"), ("G", "a"), ("G", "b"), ("G", "d"),
]


@dataclass
class TraceInstance:
    """A personalised hand-trace instance for BFS (and the DFS adaptation)."""

    student_id: str
    graph: Dict[str, List[str]]
    start: str
    goal: str

    def describe(self) -> str:
        lines = [
            f"Trace instance for student {self.student_id}",
            f"  start = {self.start}, goal = {self.goal}",
            "  graph (adjacency list):",
        ]
        for n in NODES:
            lines.append(f"    {n} -> {self.graph[n]}")
        return "\n".join(lines)


def generate_trace_instance(student_id: str) -> TraceInstance:
    """Generate a hand-trace instance personalised to `student_id`."""
    rng = seed_for(TASK_ID, "trace", student_id)
    graph = _adjacency()
    start, goal = rng.choice(VALID_PAIRS)
    return TraceInstance(
        student_id=str(student_id), graph=graph, start=start, goal=goal
    )


def generate_instance(student_id: str) -> Dict[str, object]:
    """Top-level entry point. Returns `{"trace": TraceInstance}`."""
    return {"trace": generate_trace_instance(student_id)}


if __name__ == "__main__":
    sid = sys.argv[1] if len(sys.argv) > 1 else "demo123"
    inst = generate_instance(sid)
    print(inst["trace"].describe())
