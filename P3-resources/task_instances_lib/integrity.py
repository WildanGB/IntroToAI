"""Integrity-check helpers for per-task generators."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable, Union

PathLike = Union[str, Path]


def _resolve(path: PathLike) -> Path:
    """Resolve `path` against cwd, falling back to beside this package.

    Notebooks call `fingerprint("task_instances.py")` as a plain relative
    path, expecting cwd to be the folder holding that file. That holds when
    a full task submission is graded, but a bare .ipynb graded via an
    automarker export has no such file at cwd — only the marker's own
    bundled copy, shipped beside this package. Try cwd first (unchanged
    behaviour), then that bundled copy.
    """
    candidate = Path(path)
    if candidate.is_absolute() or candidate.exists():
        return candidate
    bundled = Path(__file__).resolve().parent.parent / candidate
    return bundled if bundled.exists() else candidate


def fingerprint(path: PathLike) -> str:
    """Return the SHA-256 hex digest of `path`.

    The student notebook prints this fingerprint at startup; the tutor CLI
    prints the canonical fingerprint. A mismatch indicates the student
    modified or replaced the generator module.
    """
    return hashlib.sha256(_resolve(path).read_bytes()).hexdigest()


def fingerprint_many(paths: Iterable[PathLike]) -> str:
    """Combine the digests of multiple files into a single fingerprint.

    Useful when a task depends on more than one generator file (e.g. a
    `task_instances.py` plus a `helpers.py`).
    """
    h = hashlib.sha256()
    for p in sorted(str(x) for x in paths):
        h.update(p.encode("utf-8"))
        h.update(b"\n")
        h.update(Path(p).read_bytes())
        h.update(b"\n")
    return h.hexdigest()
