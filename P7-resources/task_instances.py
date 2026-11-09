"""
Personalised task instance generator for P7 (Naive Bayes) — STUDENT-FACING.

Each student gets a personalised TEST EMAIL to classify by hand against
the SHARED HAND_TRACE_DATA training set in p7_helpers.py. The email is
engineered so that EVERY student hits a zero-frequency wipeout on one
side (spam or ham), preserving the P7.1 pedagogical beat.

Updated 2026-06-01: pre-implementation P7.1 deliverable. The previous
post-implementation Part 6 checkpoint was retired in the same commit.

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

TASK_ID = "P7"

# Vocabulary derived from HAND_TRACE_DATA (p7_helpers.py).
#
# HAND_TRACE_DATA:
#   spam: "free prize winner", "free offer click", "winner call now"
#   ham:  "meeting agenda tuesday", "project update meeting", "free lunch friday"
#
# Categories:
#   spam-only: prize, winner, offer, click, call, now
#   ham-only:  meeting, agenda, tuesday, project, update, lunch, friday
#   overlap:   free  (appears in both classes)
#
# Including a HAM-ONLY word in the test => P(word|spam) = 0 => spam branch
# dies (matches the canonical "free meeting" pedagogy).
# Including a SPAM-ONLY word => P(word|ham) = 0 => ham branch dies.
SPAM_ONLY = ["prize", "winner", "offer", "click", "call", "now"]
HAM_ONLY = ["meeting", "agenda", "tuesday", "project", "update", "lunch", "friday"]
OVERLAP = ["free"]


@dataclass
class EmailInstance:
    """A personalised test email for the P7.1 hand trace."""

    student_id: str
    email_text: str
    n_words: int
    wipeout_side: str  # "spam" or "ham" — which class branch dies

    def describe(self) -> str:
        return "\n".join([
            f"Personalised test email for student {self.student_id}",
            f'  email_text   = "{self.email_text}"',
            f"  n_words      = {self.n_words}",
            f"  wipeout_side = {self.wipeout_side}  (the class with a zero-frequency word)",
        ])


def generate_email_instance(student_id: str) -> EmailInstance:
    """Generate a personalised test email with a guaranteed zero-frequency wipeout.

    Each email has 2 or 3 words. Always includes one word from the chosen
    wipeout side (HAM_ONLY => spam dies; SPAM_ONLY => ham dies), one OVERLAP
    word ("free"), and optionally a second word from the SAME side.

    Half the cohort gets a spam-side wipeout (the canonical "free meeting"
    pattern); half gets a ham-side wipeout (mirror image — useful contrast
    when tutors compare student traces.)
    """
    rng = seed_for(TASK_ID, "email", student_id)

    if rng.choice([True, False]):
        wipeout_side = "spam"
        wipeout_pool = HAM_ONLY
    else:
        wipeout_side = "ham"
        wipeout_pool = SPAM_ONLY

    n_words = rng.choice([2, 3])
    words: List[str] = []
    words.append(rng.choice(wipeout_pool))
    words.append(OVERLAP[0])  # "free"
    if n_words == 3:
        extra = rng.choice([w for w in wipeout_pool if w not in words])
        words.append(extra)

    rng.shuffle(words)
    return EmailInstance(
        student_id=str(student_id),
        email_text=" ".join(words),
        n_words=len(words),
        wipeout_side=wipeout_side,
    )


def generate_instance(student_id: str) -> dict:
    return {"email": generate_email_instance(student_id)}


if __name__ == "__main__":
    sid = sys.argv[1] if len(sys.argv) > 1 else "demo123"
    inst = generate_instance(sid)
    print(inst["email"].describe())
