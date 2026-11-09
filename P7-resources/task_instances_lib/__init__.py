"""Shared infrastructure for personalised, anti-autocomplete task instances.

Conventions:
    - Every task has a `task_id` (e.g. "P3_prototype").
    - Every student has a `student_id` (their Monash enrolled ID).
    - Per-task instance generators live in `Tasks/<task>/student/task_instances.py`.
    - Per-task solution generators live in `Tasks/<task>/staff/task_solution.py`.
    - All generators import from this library for seeded RNG, integrity
      fingerprints, and CLI scaffolding.

Public API:
    seed_for(task_id, part, student_id) -> random.Random
    fingerprint(path) -> str
    make_tutor_cli(...) -> argparse.ArgumentParser factory
    is_grid_reachable(rows, cols, start, goal, obstacles) -> bool
    sample_reachable_grid(rng, rows, cols, n_obstacles, start, goal) -> list
"""

from .seeded import seed_for
from .integrity import fingerprint
from .cli import make_tutor_cli
from .grids import is_grid_reachable, sample_reachable_grid

__all__ = [
    "seed_for",
    "fingerprint",
    "make_tutor_cli",
    "is_grid_reachable",
    "sample_reachable_grid",
]
