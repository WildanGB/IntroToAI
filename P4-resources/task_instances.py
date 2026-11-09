"""
Personalised task instance generator for P4 (Hill Climbing) — STUDENT-FACING.

Each student gets:
  - A personalised 8-queens starting board (rejection-sampled so initial
    violations are in [5, 16]).
  - A personalised set of 4 candidate neighbour boards drawn from the 56
    possible single-queen moves. The student hand-traces ONE hill-climbing
    step on this bounded set (count violations for the 4 candidates,
    identify the best move).

This is the PRE-implementation personalised hand-trace instance (P4.1).
Replaces the previous post-implementation Part 4b checkpoint (removed
2026-06-01) following the P3 pattern.

Uses the shared anti-autocomplete framework at Tasks/_lib/.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

_HERE = Path(__file__).resolve().parent
# task_instances_lib ships beside this file in the OnTrack bundle and lives at
# Tasks/_lib/ in the design repo — add whichever location actually has it.
for _cand in (_HERE, _HERE.parents[1] / "_lib"):
    if (_cand / "task_instances_lib").is_dir():
        if str(_cand) not in sys.path:
            sys.path.insert(0, str(_cand))
        break

from task_instances_lib import seed_for  # noqa: E402

TASK_ID = "P4"


@dataclass
class BoardInstance:
    """A personalised starting board + 4 candidate neighbours for 8-queens."""

    student_id: str
    board: List[int]                       # n=8 column positions
    candidates: List[List[int]] = field(default_factory=list)  # 4 neighbour boards

    def describe(self) -> str:
        lines = [
            f"Personalised starting board for student {self.student_id}",
            f"  board      = {self.board}",
            f"  violations = {_count_violations(self.board)}",
            "  candidates (4 of 56 possible single-queen moves):",
        ]
        for i, c in enumerate(self.candidates):
            v = _count_violations(c)
            lines.append(f"    cand[{i}] = {c}  (violations = {v})")
        return "\n".join(lines)


def _count_violations(board: List[int]) -> int:
    """Count attacking pairs (mirrors p4_helpers.count_violations)."""
    n = len(board)
    v = 0
    for i in range(n):
        for j in range(i + 1, n):
            if board[i] == board[j]:
                v += 1
            elif abs(board[i] - board[j]) == abs(i - j):
                v += 1
    return v


def _all_neighbours(board: List[int]) -> List[List[int]]:
    """All boards reachable by moving one queen within its row."""
    n = len(board)
    neighbours: List[List[int]] = []
    for row in range(n):
        for col in range(n):
            if col == board[row]:
                continue
            new_board = list(board)
            new_board[row] = col
            neighbours.append(new_board)
    return neighbours


def generate_board_instance(student_id: str, n: int = 8) -> BoardInstance:
    """Generate a personalised n-queens starting board + 4 candidate neighbours.

    The board is rejection-sampled to have initial violations in [5, 16].
    The 4 candidates are drawn from the n*(n-1) possible single-queen moves
    so that the student's hand trace covers:
      - one BEST neighbour (fewest violations)
      - one WORST neighbour (most violations)
      - two RANDOM neighbours from the middle of the distribution

    This gives students a tractable view of the local search landscape
    without enumerating all 56 candidates by hand.
    """
    rng = seed_for(TASK_ID, "board", student_id)
    board = None
    for _ in range(500):
        candidate = [rng.randint(0, n - 1) for _ in range(n)]
        v = _count_violations(candidate)
        if 5 <= v <= 16:
            board = candidate
            break
    if board is None:
        board = [rng.randint(0, n - 1) for _ in range(n)]

    neighbours = _all_neighbours(board)
    # Sort by violation count for deterministic best/worst pick.
    scored = sorted(((_count_violations(nb), idx, nb)
                     for idx, nb in enumerate(neighbours)),
                    key=lambda t: (t[0], t[1]))
    best = scored[0][2]
    worst = scored[-1][2]
    # Sample two random middle candidates (not best, not worst).
    middle_pool = [t[2] for t in scored[1:-1]]
    rng.shuffle(middle_pool)
    mid1, mid2 = middle_pool[0], middle_pool[1]

    candidates = [best, mid1, mid2, worst]
    rng.shuffle(candidates)  # Hide which is best/worst from position.

    return BoardInstance(
        student_id=str(student_id), board=board, candidates=candidates
    )


def generate_instance(student_id: str) -> dict:
    return {"board": generate_board_instance(student_id)}


if __name__ == "__main__":
    sid = sys.argv[1] if len(sys.argv) > 1 else "demo123"
    inst = generate_instance(sid)
    print(inst["board"].describe())
