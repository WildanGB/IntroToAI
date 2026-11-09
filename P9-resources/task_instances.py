"""
Personalised task instance generator for P9 (Gradient Descent) — STUDENT-FACING.

Each student gets a personalised 1D quadratic `f(w) = (w - c)^2` and
starting point `w_0`. They hand-trace 5 steps of gradient descent on
THEIR quadratic before implementing.

Updated 2026-06-01: pre-implementation P9.1 deliverable. The previous
post-implementation Part 8 checkpoint (non-convex basin run) was retired
in the same commit.

f(w)        = (w - c)^2
f'(w)       = 2 * (w - c)
update rule = w <- w - alpha * f'(w)

With fixed alpha = 0.1 and integer (c, w_0), the trajectory is:
   w_{n+1} = c + 0.8 * (w_n - c)
i.e. distance from c shrinks by factor 0.8 each step.

Uses the shared anti-autocomplete framework at Tasks/_lib/.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
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

TASK_ID = "P9"

# Minimum location pool. Excludes 0 (that's the canonical f(w)=w^2).
C_POOL = [-3, -2, -1, 1, 2, 3]

# Initial distance |w_0 - c|. Larger -> more dramatic trajectory.
D_POOL = [3, 4, 5]

# Direction of approach (above or below the minimum).
DIRECTION_POOL = [+1, -1]


@dataclass
class QuadraticInstance:
    """A personalised 1D quadratic f(w) = (w - c)^2 + starting point."""

    student_id: str
    c: int           # minimum location: f(w) = (w - c)^2 has min at w = c
    w0: int          # starting point
    alpha: float     # learning rate (fixed at 0.1 for clean arithmetic)
    steps: int       # number of GD steps to trace (5)

    @property
    def f_repr(self) -> str:
        if self.c >= 0:
            return f"f(w) = (w - {self.c})^2"
        else:
            return f"f(w) = (w + {-self.c})^2"

    @property
    def df_repr(self) -> str:
        if self.c >= 0:
            return f"f'(w) = 2 * (w - {self.c})"
        else:
            return f"f'(w) = 2 * (w + {-self.c})"

    def describe(self) -> str:
        return "\n".join([
            f"Personalised quadratic for student {self.student_id}",
            f"  function:   {self.f_repr}",
            f"  derivative: {self.df_repr}",
            f"  minimum at  w = {self.c}",
            f"  start       w_0 = {self.w0}",
            f"  alpha       = {self.alpha}",
            f"  steps       = {self.steps}",
        ])


def generate_quadratic_instance(student_id: str) -> QuadraticInstance:
    """Generate a per-student (c, w_0) for the P9.1 hand trace.

    alpha and steps are fixed (0.1, 5) so the trajectory schema is uniform
    across the cohort. Only c and the starting point w_0 vary.

    w_0 is constructed as c + direction * d so that the integer signs of
    (w - c) are clean. Initial distance d is in {3, 4, 5}, giving
    5-step trajectories that visibly converge but don't reach c.
    """
    rng = seed_for(TASK_ID, "quadratic", student_id)
    c = rng.choice(C_POOL)
    d = rng.choice(D_POOL)
    direction = rng.choice(DIRECTION_POOL)
    w0 = c + direction * d
    return QuadraticInstance(
        student_id=str(student_id), c=c, w0=w0, alpha=0.1, steps=5
    )


def generate_instance(student_id: str) -> dict:
    return {"quad": generate_quadratic_instance(student_id)}


if __name__ == "__main__":
    sid = sys.argv[1] if len(sys.argv) > 1 else "demo123"
    inst = generate_instance(sid)
    q = inst["quad"]
    print(q.describe())
    # Demo trajectory.
    w = float(q.w0)
    print(f"\n  step | w       | f(w)    | f'(w)   | update    | new_w")
    for step in range(q.steps):
        fw = (w - q.c) ** 2
        dfw = 2 * (w - q.c)
        update = q.alpha * dfw
        new_w = w - update
        print(f"  {step:4d} | {w:+.4f} | {fw:+.4f} | {dfw:+.4f} | {update:+.4f}  | {new_w:+.4f}")
        w = new_w
