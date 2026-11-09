"""
Personalised task instance generator for P6 (Probability + Bayes) — STUDENT-FACING.

Produces a personalised Bayes-rule scenario with three parameters:
    - base_rate (prevalence of the condition)
    - sensitivity (P(positive | condition))
    - specificity (P(negative | no condition))

The student computes P(condition | positive) using Bayes' rule on
THEIR personalised numbers.

Uses the shared anti-autocomplete framework at Tasks/_lib/.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

_HERE = Path(__file__).resolve().parent
# task_instances_lib ships beside this file in the OnTrack bundle and lives at
# Tasks/_lib/ in the design repo — add whichever location actually has it.
for _cand in (_HERE, _HERE.parents[1] / "_lib"):
    if (_cand / "task_instances_lib").is_dir():
        if str(_cand) not in sys.path:
            sys.path.insert(0, str(_cand))
        break

from task_instances_lib import seed_for  # noqa: E402

TASK_ID = "P6"


@dataclass
class BayesScenario:
    """A personalised Bayes-rule scenario."""
    student_id: str
    scenario_name: str
    base_rate: float
    sensitivity: float
    specificity: float

    def describe(self) -> str:
        return "\n".join([
            f"Personalised Bayes scenario for student {self.student_id}",
            f"  scenario:    {self.scenario_name}",
            f"  base_rate    P(condition)              = {self.base_rate:.4f}",
            f"  sensitivity  P(positive | condition)   = {self.sensitivity:.4f}",
            f"  specificity  P(negative | no condition)= {self.specificity:.4f}",
        ])


SCENARIOS = [
    "rare disease screening",
    "predictive maintenance fault detector",
    "fraud-detection model",
    "spam filter on legitimate inbox",
    "automated radiology triage",
]


def generate_bayes_scenario(student_id: str) -> BayesScenario:
    """Generate a personalised Bayes scenario.

    Parameter bands chosen so the posterior P(condition|positive) is
    meaningfully different across students and never trivially close
    to 0 or 1. base_rate sweeps from 0.5% to 8%, sensitivity from 0.88
    to 0.98, specificity from 0.92 to 0.995.
    """
    rng = seed_for(TASK_ID, "bayes", student_id)
    base_rate = round(rng.uniform(0.005, 0.08), 4)
    sensitivity = round(rng.uniform(0.88, 0.98), 4)
    specificity = round(rng.uniform(0.92, 0.995), 4)
    scenario = rng.choice(SCENARIOS)
    return BayesScenario(student_id=str(student_id), scenario_name=scenario,
                         base_rate=base_rate, sensitivity=sensitivity,
                         specificity=specificity)


def generate_instance(student_id: str) -> dict:
    return {"bayes": generate_bayes_scenario(student_id)}


if __name__ == "__main__":
    sid = sys.argv[1] if len(sys.argv) > 1 else "demo123"
    inst = generate_instance(sid)
    print(inst["bayes"].describe())
