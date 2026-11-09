"""
Personalised task instance generator for P8 (Weighted Sums) — STUDENT-FACING.

Each student gets a personalised weight vector + bias `(w, b)`. They apply
their classifier to the SHARED 6-point HAND_TRACE dataset (in p8_helpers.py)
by hand: compute `w·x + b` for each point, apply the sign-threshold, predict
the class, compare with the true labels.

Updated 2026-06-01: pre-implementation P8.1 deliverable. The previous
post-implementation Part 7 checkpoint was retired in the same commit.
`test_point` field removed (unused after the refactor).

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

TASK_ID = "P8"

# Same 6 hand-trace points as p8_helpers.HAND_TRACE_POINTS.
# Replicated here so the generator + tutor can validate (w, b) without
# importing numpy in the personalisation module.
_HAND_TRACE_POINTS = [
    (3, 1), (1, 3), (4, 0), (1, 4), (5, 2), (0, 2),
]
_HAND_TRACE_LABELS = [1, -1, 1, -1, 1, -1]


@dataclass
class LinearClassifierInstance:
    """A personalised weight vector + bias for the P8.1 hand trace."""

    student_id: str
    weights: List[int]   # [w1, w2]
    bias: int

    def describe(self) -> str:
        return "\n".join([
            f"Personalised linear classifier for student {self.student_id}",
            f"  weights w = {self.weights}",
            f"  bias    b = {self.bias}",
        ])


def _classify_all(weights: List[int], bias: int):
    """Return (sums, predictions, accuracy) over the 6 hand-trace points."""
    sums = []
    preds = []
    correct = 0
    for (x1, x2), true_label in zip(_HAND_TRACE_POINTS, _HAND_TRACE_LABELS):
        s = weights[0] * x1 + weights[1] * x2 + bias
        pred = 1 if s >= 0 else -1
        sums.append(s)
        preds.append(pred)
        if pred == true_label:
            correct += 1
    return sums, preds, correct


def generate_classifier_instance(student_id: str) -> LinearClassifierInstance:
    """Generate a personalised (w, b).

    Constraints:
      - w1, w2 in {-2, -1, 1, 2}  (no zero weight, keeps both dims active)
      - b in {-4, -3, -2, -1, 0, 1, 2}
      - All 6 weighted sums must be non-zero (avoids boundary-tie ambiguity).
      - Predictions cover BOTH classes (avoids degenerate all-+1 or all--1).
      - Accuracy in {3, 4, 5} out of 6 (avoids trivial 6/6 and useless 0/6).

    The 3-5 accuracy band with mixed-class predictions makes the hand trace
    pedagogically rich: students see their classifier is partially wrong,
    motivating the Part 2 'experiment with different weights' exercise.
    """
    rng = seed_for(TASK_ID, "linclf", student_id)
    for _ in range(500):
        w1 = rng.choice([-2, -1, 1, 2])
        w2 = rng.choice([-2, -1, 1, 2])
        b = rng.choice([-4, -3, -2, -1, 0, 1, 2])
        sums, preds, correct = _classify_all([w1, w2], b)
        if any(s == 0 for s in sums):
            continue
        if 1 not in preds or -1 not in preds:
            continue  # require at least one +1 and one -1 prediction
        if correct in (3, 4, 5):
            return LinearClassifierInstance(
                student_id=str(student_id), weights=[w1, w2], bias=b
            )
    # Fallback: a known-good 5/6 classifier.
    return LinearClassifierInstance(
        student_id=str(student_id), weights=[2, -1], bias=-2
    )


def generate_instance(student_id: str) -> dict:
    return {"linclf": generate_classifier_instance(student_id)}


if __name__ == "__main__":
    sid = sys.argv[1] if len(sys.argv) > 1 else "demo123"
    inst = generate_instance(sid)
    lc = inst["linclf"]
    print(lc.describe())
    sums, preds, correct = _classify_all(lc.weights, lc.bias)
    print(f"  sums        = {sums}")
    print(f"  predictions = {preds}")
    print(f"  accuracy    = {correct}/6")
