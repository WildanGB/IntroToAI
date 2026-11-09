"""
C2 Helper Data — Classification & Fairness

FIT1061 Introduction to Artificial Intelligence

This module provides data and utilities for C2. It builds on your
P7 (Naive Bayes) and P8 (weighted sums) work.

    from c2_helpers import (STOP_WORDS, top_n_words,
                            HARD_TEST_DATA,
                            LOAN_X_TRAIN, LOAN_Y_TRAIN, LOAN_GROUP_TRAIN,
                            LOAN_X_TEST, LOAN_Y_TEST, LOAN_GROUP_TEST,
                            discretise, confusion_metrics,
                            plot_fairness_bars, print_metrics_table)
"""

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# C2.1: Feature Engineering utilities
# ---------------------------------------------------------------------------

# Common English stop words — words that appear frequently but carry
# little information about whether a message is spam or ham.
STOP_WORDS = {
    'a', 'an', 'the', 'is', 'it', 'in', 'on', 'at', 'to', 'for',
    'of', 'and', 'or', 'be', 'am', 'are', 'was', 'were', 'been',
    'has', 'have', 'had', 'do', 'does', 'did', 'will', 'would',
    'can', 'could', 'should', 'may', 'might', 'shall', 'must',
    'i', 'me', 'my', 'we', 'us', 'our', 'you', 'your', 'he', 'him',
    'his', 'she', 'her', 'they', 'them', 'their', 'this', 'that',
    'if', 'but', 'not', 'no', 'so', 'up', 'out', 'with', 'from',
    'by', 'as', 'just', 'about', 'all', 'some', 'any', 'each',
}


# Harder test set for feature engineering experiments (C2.1).
# These messages are deliberately ambiguous — they mix spam-like and
# ham-like vocabulary to stress-test feature selection.
HARD_TEST_DATA = [
    ("spam", "free entry in a weekly competition to win prizes"),
    ("spam", "you have won a cash prize call to claim now"),
    ("spam", "urgent update your account has been selected for a free reward"),
    ("spam", "congratulations you are today lucky winner text back"),
    ("spam", "act now free offer for a limited number of customers"),
    ("spam", "your mobile number won a guaranteed prize reply to claim"),
    ("spam", "click now to receive your exclusive free bonus today"),
    ("ham",  "hey are you free for a meeting on friday"),
    ("ham",  "the project deadline has been updated please review"),
    ("ham",  "can you call me when you get a chance"),
    ("ham",  "free lunch at the office tomorrow let me know"),
    ("ham",  "your order update delivery is scheduled for today"),
    ("ham",  "reminder you have a meeting at 3pm today"),
    ("ham",  "can you review the report and let me know your thoughts"),
    ("ham",  "good news the project has been approved for the next phase"),
    ("ham",  "are you coming to the team lunch on friday"),
    ("ham",  "call me back about the update on the delivery schedule"),
    ("ham",  "just checking in hope you have a great day"),
    ("ham",  "your meeting has been moved to tuesday please confirm"),
    ("ham",  "free parking available at the new office building"),
]


def top_n_words(data, n, exclude=None):
    """Find the n most frequent words across a dataset.

    Args:
        data: List of (label, text) tuples.
        n: Number of top words to return.
        exclude: Optional set of words to exclude (e.g., STOP_WORDS).

    Returns:
        A list of the n most frequent words, sorted by frequency (descending).
    """
    counts = {}
    for label, text in data:
        for word in text.lower().split():
            if exclude and word in exclude:
                continue
            counts[word] = counts.get(word, 0) + 1
    ranked = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    return [word for word, count in ranked[:n]]


# ---------------------------------------------------------------------------
# C2.2–C2.3: Loan Application Dataset
# ---------------------------------------------------------------------------
# Synthetic loan data with two demographic groups. Group A has higher
# income/credit on average (systemic advantage). Group B has lower
# averages. This creates different base rates — essential for seeing
# why fairness definitions conflict.

def _generate_loan_data(n=500, seed=42):
    """Generate synthetic loan data. Internal — use the pre-split constants."""
    rng = np.random.RandomState(seed)

    n_a = n // 2
    n_b = n - n_a

    # Group A: higher income/credit on average
    income_a = rng.randn(n_a) * 1.0 + 1.5
    credit_a = rng.randn(n_a) * 1.0 + 1.5
    logit_a = 0.8 * income_a + 0.8 * credit_a + rng.randn(n_a) * 0.8
    y_a = (logit_a > 1.5).astype(int)

    # Group B: lower income/credit on average
    income_b = rng.randn(n_b) * 1.0 + 0.5
    credit_b = rng.randn(n_b) * 1.0 + 0.5
    logit_b = 0.8 * income_b + 0.8 * credit_b + rng.randn(n_b) * 0.8
    y_b = (logit_b > 1.5).astype(int)

    X = np.vstack([
        np.column_stack([income_a, credit_a]),
        np.column_stack([income_b, credit_b]),
    ])
    y = np.concatenate([y_a, y_b])
    group = np.array(['A'] * n_a + ['B'] * n_b)

    return X, y, group


# Pre-generate and split (deterministic)
_X, _y, _group = _generate_loan_data(n=500, seed=42)
_rng = np.random.RandomState(123)
_indices = _rng.permutation(len(_X))
_split = int(0.7 * len(_X))
_train_idx = _indices[:_split]
_test_idx = _indices[_split:]

LOAN_X_TRAIN = _X[_train_idx]
LOAN_Y_TRAIN = _y[_train_idx]
LOAN_GROUP_TRAIN = _group[_train_idx]

LOAN_X_TEST = _X[_test_idx]
LOAN_Y_TEST = _y[_test_idx]
LOAN_GROUP_TEST = _group[_test_idx]

# Feature names for display
LOAN_FEATURE_NAMES = ['income_score', 'credit_score']


def discretise(X, n_bins=3):
    """Discretise continuous features into categorical bins.

    Useful for running Naive Bayes on numeric data. Each feature is
    split into n_bins equal-width bins labelled 'low', 'medium', 'high'
    (for 3 bins) or numbered (for other bin counts).

    Args:
        X: NumPy array of shape (n_samples, n_features).
        n_bins: Number of bins per feature (default 3).

    Returns:
        List of lists of strings — one inner list per sample.
        Each string is "feature_name=bin_label".
    """
    bin_labels = ['low', 'medium', 'high'] if n_bins == 3 else \
                 [str(i) for i in range(n_bins)]
    names = LOAN_FEATURE_NAMES

    result = []
    # Compute bin edges per feature
    edges = []
    for j in range(X.shape[1]):
        col_min, col_max = X[:, j].min(), X[:, j].max()
        e = np.linspace(col_min, col_max, n_bins + 1)
        e[0] = -np.inf
        e[-1] = np.inf
        edges.append(e)

    for i in range(X.shape[0]):
        features = []
        for j in range(X.shape[1]):
            bin_idx = np.digitize(X[i, j], edges[j]) - 1
            bin_idx = min(bin_idx, n_bins - 1)
            features.append(f"{names[j]}={bin_labels[bin_idx]}")
        result.append(features)

    return result


# ---------------------------------------------------------------------------
# Evaluation utilities
# ---------------------------------------------------------------------------

def confusion_metrics(y_true, y_pred):
    """Compute confusion matrix and derived metrics.

    Args:
        y_true: Array of true labels (0 or 1).
        y_pred: Array of predicted labels (0 or 1).

    Returns:
        Dict with: tp, fp, fn, tn, precision, recall, fpr, accuracy.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    accuracy = (tp + tn) / (tp + fp + fn + tn)

    return {
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
        'precision': precision, 'recall': recall,
        'fpr': fpr, 'accuracy': accuracy,
    }


def print_metrics_table(metrics_dict, title="Metrics"):
    """Print evaluation metrics in a readable table.

    Args:
        metrics_dict: Dict from confusion_metrics().
        title: Table title.
    """
    m = metrics_dict
    print(f"\n{title}")
    print(f"  Accuracy:  {m['accuracy']:.2%}")
    print(f"  Precision: {m['precision']:.2%}")
    print(f"  Recall:    {m['recall']:.2%}")
    print(f"  FPR:       {m['fpr']:.2%}")
    print(f"  (TP={m['tp']}, FP={m['fp']}, FN={m['fn']}, TN={m['tn']})")


def plot_loan_data(X, y, group, title="Loan Applications"):
    """Scatter plot of loan data coloured by outcome and shaped by group.

    Args:
        X: Feature array (n, 2).
        y: Labels (0/1).
        group: Group array ('A'/'B').
        title: Plot title.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    for g, marker in [('A', 'o'), ('B', 's')]:
        for label, colour in [(1, '#2ecc71'), (0, '#e74c3c')]:
            mask = (group == g) & (y == label)
            lbl = f"Group {g}, {'repaid' if label == 1 else 'defaulted'}"
            ax.scatter(X[mask, 0], X[mask, 1], c=colour, marker=marker,
                       alpha=0.5, edgecolors='black', linewidths=0.3,
                       label=lbl, s=40)

    ax.set_xlabel('Income Score', fontsize=11)
    ax.set_ylabel('Credit Score', fontsize=11)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=8, loc='upper left')
    plt.tight_layout()
    plt.show()


def plot_fairness_bars(group_metrics, title="Fairness Metrics by Group"):
    """Bar chart comparing metrics across demographic groups.

    Args:
        group_metrics: Dict like {'A': {'approval_rate': ..., 'tpr': ..., ...},
                                   'B': {...}}.
        title: Plot title.
    """
    metrics = ['approval_rate', 'tpr', 'fpr', 'precision']
    labels = ['Approval\nRate', 'TPR\n(Recall)', 'FPR', 'Precision']
    groups = sorted([k for k in group_metrics if k != 'gaps'])

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))

    for i, g in enumerate(groups):
        vals = [group_metrics[g].get(m, 0) for m in metrics]
        offset = -width / 2 + i * width
        bars = ax.bar(x + offset, vals, width, label=f'Group {g}',
                       color=['#3498db', '#e74c3c'][i], alpha=0.8)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=9)

    ax.set_ylabel('Value', fontsize=11)
    ax.set_title(title, fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("C2 Helpers")
    print(f"  Stop words: {len(STOP_WORDS)}")
    print(f"  Loan train: {LOAN_X_TRAIN.shape[0]} samples "
          f"({LOAN_Y_TRAIN.sum()} repaid, {(LOAN_Y_TRAIN == 0).sum()} defaulted)")
    print(f"  Loan test:  {LOAN_X_TEST.shape[0]} samples "
          f"({LOAN_Y_TEST.sum()} repaid, {(LOAN_Y_TEST == 0).sum()} defaulted)")

    # Base rates by group
    for g in ['A', 'B']:
        mask_tr = LOAN_GROUP_TRAIN == g
        mask_te = LOAN_GROUP_TEST == g
        rate_tr = LOAN_Y_TRAIN[mask_tr].mean()
        rate_te = LOAN_Y_TEST[mask_te].mean()
        print(f"  Group {g}: train repay rate={rate_tr:.2%}, "
              f"test repay rate={rate_te:.2%}")
