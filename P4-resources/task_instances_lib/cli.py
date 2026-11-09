"""Tutor CLI factory.

Each per-task tutor module calls `make_tutor_cli(...)` and gets back a
ready-to-run argparse-based CLI with:
    - single-ID emission (instance + solution)
    - cohort batch emission as CSV
    - canonical fingerprint emission

The per-task module supplies:
    - task_id (str)
    - student_module_path (Path)   — for fingerprinting
    - generate_solution(student_id) -> dict   — the answer-key function
    - format_one(student_id, sol) -> str       — single-ID pretty print
    - cohort_row(student_id, sol) -> list      — one cohort CSV row
    - cohort_header() -> list                  — CSV header row

Usage in a per-task tutor module:

    from task_instances_lib import make_tutor_cli
    from pathlib import Path

    parser, run = make_tutor_cli(
        task_id="P3_prototype",
        student_module_path=Path(__file__).parents[1] / "student" / "task_instances.py",
        generate_solution=generate_solution,
        format_one=_format_one,
        cohort_row=_cohort_row,
        cohort_header=_cohort_header,
    )

    if __name__ == "__main__":
        run()
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Callable, List, Tuple

from .integrity import fingerprint


def make_tutor_cli(
    task_id: str,
    student_module_path: Path,
    generate_solution: Callable[[str], object],
    format_one: Callable[[str, object], str],
    cohort_row: Callable[[str, object], list],
    cohort_header: Callable[[], list],
) -> Tuple[argparse.ArgumentParser, Callable[[], None]]:
    """Build a tutor CLI for one task. Returns (parser, run)."""

    def canonical_fingerprint() -> str:
        return fingerprint(student_module_path)

    def emit_one(student_id: str) -> None:
        sol = generate_solution(student_id)
        print(f"Task: {task_id}")
        print(f"Canonical student-module fingerprint:")
        print(f"  {canonical_fingerprint()}")
        print()
        print(format_one(student_id, sol))

    def emit_cohort(cohort_csv: str) -> None:
        writer = csv.writer(sys.stdout)
        writer.writerow(cohort_header())
        with open(cohort_csv, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sid = row["student_id"]
                sol = generate_solution(sid)
                writer.writerow(cohort_row(sid, sol))

    parser = argparse.ArgumentParser(description=f"Tutor CLI for {task_id}")
    parser.add_argument("student_id", nargs="?",
                        help="Single student ID (Monash enrolled ID).")
    parser.add_argument("--cohort", metavar="CSV",
                        help="Path to a CSV with a `student_id` column.")
    parser.add_argument("--fingerprint", action="store_true",
                        help=f"Print canonical {student_module_path.name} "
                             f"SHA-256 and exit.")

    def run() -> None:
        args = parser.parse_args()
        if args.fingerprint:
            print(canonical_fingerprint())
            return
        if args.cohort:
            emit_cohort(args.cohort)
            return
        if args.student_id:
            emit_one(args.student_id)
            return
        parser.error("Provide a student_id, --cohort, or --fingerprint.")

    return parser, run
