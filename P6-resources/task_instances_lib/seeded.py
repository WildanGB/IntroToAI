"""Deterministic seeded RNGs for personalised task instances."""

from __future__ import annotations

import hashlib
import random


def seed_for(task_id: str, part: str, student_id: str) -> random.Random:
    """Return a `random.Random` seeded deterministically for this triple.

    The seed is the leading 64 bits of `sha256("<task_id>:<part>:<student_id>")`.
    Stable across Python versions, machines, and operating systems.

    Args:
        task_id: A short task identifier, e.g. "P3_prototype".
        part: A label distinguishing instance components, e.g. "trace", "grid".
            Using distinct labels for distinct components prevents accidental
            correlation between them (e.g. the trace graph and the grid layout
            being seeded by the same number).
        student_id: The student's enrolled ID. Stringified before hashing.

    Returns:
        A `random.Random` instance ready for sampling.
    """
    payload = f"{task_id}:{part}:{student_id}".encode("utf-8")
    seed = int(hashlib.sha256(payload).hexdigest()[:16], 16)
    return random.Random(seed)
