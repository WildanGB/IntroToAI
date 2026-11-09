"""
P7 Helper Data — Naive Bayes Spam Classifier

FIT1061 Introduction to Artificial Intelligence

This module provides the dataset and utility functions for P7.
You do NOT need to modify this file. Import it in your notebook:

    from p7_helpers import (TRAIN_DATA, TEST_DATA, HAND_TRACE_DATA,
                            HAND_TRACE_TEST, tokenise, build_vocabulary,
                            print_dataset_summary, print_confusion_matrix,
                            plot_log_score_bars)

The dataset uses plain Python (no NumPy). Each message is a tuple
of (label, text), where label is "spam" or "ham".
"""


# ---------------------------------------------------------------------------
# Tokeniser
# ---------------------------------------------------------------------------

def tokenise(text):
    """Split text into lowercase words. Returns a set (unique words only).

    Args:
        text: A string.

    Returns:
        A set of lowercase words.
    """
    return set(text.lower().split())


def build_vocabulary(data):
    """Build the set of all unique words in the dataset.

    Args:
        data: A list of (label, text) tuples.

    Returns:
        A sorted list of unique words.
    """
    vocab = set()
    for label, text in data:
        vocab.update(tokenise(text))
    return sorted(vocab)


# ---------------------------------------------------------------------------
# Hand-Trace Dataset (for P7.1)
# ---------------------------------------------------------------------------
# Small enough to trace by hand. 6 training emails, 1 test email.

HAND_TRACE_DATA = [
    ("spam", "free prize winner"),
    ("spam", "free offer click"),
    ("spam", "winner call now"),
    ("ham",  "meeting agenda tuesday"),
    ("ham",  "project update meeting"),
    ("ham",  "free lunch friday"),
]

HAND_TRACE_TEST = ("???", "free meeting")

# Expected hand trace:
# P(spam) = 3/6 = 0.5, P(ham) = 3/6 = 0.5
# Words in test: {"free", "meeting"}
#
# P(free|spam) = 2/3 (appears in 2 of 3 spam)
# P(free|ham)  = 1/3 (appears in 1 of 3 ham)  — note: "free lunch friday"
# P(meeting|spam) = 0/3 = 0  ← zero-frequency!
# P(meeting|ham)  = 2/3
#
# score_spam = 0.5 × (2/3) × (0/3) = 0  ← killed by zero!
# score_ham  = 0.5 × (1/3) × (2/3) = 1/9 ≈ 0.111
#
# Prediction: ham (because spam score is 0)
# But "free" is a spam signal — the zero-frequency problem is hiding it.
# This motivates Laplace smoothing (C2 extension).


# ---------------------------------------------------------------------------
# Main Dataset (for P7.2 and P7.3)
# ---------------------------------------------------------------------------
# 40 messages: 15 spam, 25 ham. Pre-split into train (32) and test (8).
# Deterministic split — every student gets the same results.

TRAIN_DATA = [
    # Spam (12 in training)
    ("spam", "congratulations you have won a free prize call now"),
    ("spam", "free offer limited time act now click here"),
    ("spam", "winner you have won free tickets call this number"),
    ("spam", "claim your free reward now text back to win"),
    ("spam", "urgent reply now to claim your free cash prize"),
    ("spam", "free entry to win a brand new phone text win"),
    ("spam", "you have been selected to receive a free gift call"),
    ("spam", "congratulations claim your cash prize now call today"),
    ("spam", "act now to receive your exclusive bonus today"),
    ("spam", "your mobile number has won a guaranteed cash prize"),
    ("spam", "free ringtone reply now to download your prize"),
    ("spam", "text win to enter our free weekly draw for prizes"),
    # Ham (20 in training)
    ("ham", "hey are you free for lunch tomorrow"),
    ("ham", "meeting at 3pm tuesday agenda attached"),
    ("ham", "project deadline extended to friday let me know"),
    ("ham", "can you review the quarterly report please"),
    ("ham", "team lunch on friday at the usual place"),
    ("ham", "meeting notes from today attached for review"),
    ("ham", "your order has been shipped delivery by thursday"),
    ("ham", "reminder dentist appointment tomorrow at 10am"),
    ("ham", "happy birthday hope you have a great day"),
    ("ham", "thanks for the update on the project timeline"),
    ("ham", "are you coming to the meeting this afternoon"),
    ("ham", "please find attached the invoice for last month"),
    ("ham", "dinner tonight at 7 let me know if you can make it"),
    ("ham", "call me when you get a chance need to discuss"),
    ("ham", "good morning how was your weekend"),
    ("ham", "the report is ready for your review"),
    ("ham", "can we reschedule our meeting to wednesday"),
    ("ham", "just checking in on the project status"),
    ("ham", "running late will be there in 20 minutes"),
    ("ham", "thanks for your help with the presentation"),
]

TEST_DATA = [
    # Spam (3 in test)
    ("spam", "you won a free holiday reply to claim your prize now"),
    ("spam", "free msg claim your reward call the number below"),
    ("spam", "urgent your account has been compromised call now"),
    # Ham (5 in test)
    ("ham", "see you at the meeting tomorrow morning"),
    ("ham", "can you pick up some groceries on your way home"),
    ("ham", "great job on the presentation today well done"),
    ("ham", "are you free this weekend for a hike"),
    ("ham", "please confirm your attendance for the event"),
]


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def print_dataset_summary(data, label="Dataset"):
    """Print a summary of spam/ham counts."""
    spam = sum(1 for l, _ in data if l == "spam")
    ham = sum(1 for l, _ in data if l == "ham")
    print(f"{label}: {len(data)} messages ({spam} spam, {ham} ham)")


def print_confusion_matrix(tp, fp, fn, tn):
    """Pretty-print a confusion matrix."""
    print(f"                    Predicted")
    print(f"                  spam    ham")
    print(f"  Actual spam  |  {tp:3d}  |  {fn:3d}  |")
    print(f"  Actual ham   |  {fp:3d}  |  {tn:3d}  |")
    print()
    print(f"  True Positives (TP):  {tp}")
    print(f"  False Positives (FP): {fp}")
    print(f"  False Negatives (FN): {fn}")
    print(f"  True Negatives (TN):  {tn}")


# ---------------------------------------------------------------------------
# Prediction trace visualisation (P7.4d)
# ---------------------------------------------------------------------------

def plot_log_score_bars(words, log_spam_contribs, log_ham_contribs,
                        log_prior_spam, log_prior_ham, prediction,
                        actual_label=None, title=None):
    """Render a per-word log-score bar chart for one Naive Bayes prediction.

    Students compute the contributions; this helper only renders.

    Args:
        words: List of words from the email that are in the vocabulary.
        log_spam_contribs: List of log P(word | spam) values, same order as words.
        log_ham_contribs:  List of log P(word | ham) values, same order as words.
        log_prior_spam: Scalar log P(spam).
        log_prior_ham:  Scalar log P(ham).
        prediction: "spam" or "ham" — the model's call.
        actual_label: Optional "spam"/"ham" — for a correctness annotation.
        title: Optional plot title.

    Notes:
        Bars are paired per word (spam left/blue, ham right/red). The prior
        appears as a separate paired bar on the left. Sum equals the total
        log-score for each class, so visual height ordering equals
        score ordering.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    MONASH_BLUE = "#006DAE"
    UTIL_RED = "#EA001F"
    GREY_1 = "#5A5A5A"

    labels = ["log P(prior)"] + [f'"{w}"' for w in words]
    spam_vals = [log_prior_spam] + list(log_spam_contribs)
    ham_vals = [log_prior_ham] + list(log_ham_contribs)

    x = np.arange(len(labels))
    width = 0.4

    fig, ax = plt.subplots(figsize=(max(7, 1.2 * len(labels)), 5))
    ax.bar(x - width / 2, spam_vals, width, label="spam",
           color=MONASH_BLUE, edgecolor="black", linewidth=0.5)
    ax.bar(x + width / 2, ham_vals, width, label="ham",
           color=UTIL_RED, edgecolor="black", linewidth=0.5)

    ax.axhline(0, color=GREY_1, linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("log-likelihood contribution")
    if title is None:
        title = "Per-word log contributions"
    correctness = ""
    if actual_label is not None:
        correctness = "  ✓" if actual_label == prediction else "  ✗"
    ax.set_title(f"{title}\nprediction: {prediction}{correctness}")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    total_spam = sum(spam_vals)
    total_ham = sum(ham_vals)
    ax.text(0.99, 0.02,
            f"sum(spam) = {total_spam:.2f}\nsum(ham)  = {total_ham:.2f}",
            transform=ax.transAxes, ha="right", va="bottom",
            family="monospace", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", edgecolor=GREY_1))

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print_dataset_summary(TRAIN_DATA, "Training set")
    print_dataset_summary(TEST_DATA, "Test set")
    print(f"\nHand-trace training: {len(HAND_TRACE_DATA)} emails")
    print(f"Hand-trace test: {HAND_TRACE_TEST}")
    vocab = build_vocabulary(TRAIN_DATA)
    print(f"Training vocabulary: {len(vocab)} words")
