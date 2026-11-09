"""
C4 Helper Data — Deployment & Model Cards

FIT1061 Introduction to Artificial Intelligence

This module provides a pre-trained Naive Bayes classifier, test datasets
(same-distribution and shifted), carbon estimation constants, and plotting
utilities for C4.

    from c4_helpers import (
        classifier, TRAIN_DATA,
        TEST_SAME, TEST_SHIFTED,
        CARBON_CONSTANTS, COMPARISON_VALUES,
        WEAK_MODEL_CARD,
        compute_metrics, plot_shift_comparison, plot_carbon_scaling,
    )

The classifier is the same Naive Bayes architecture from W7, pre-trained
on 2020-era spam/ham messages. You will test it on same-distribution and
shifted-distribution data.
"""

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Naive Bayes Classifier (same architecture as W7)
# ---------------------------------------------------------------------------

class NaiveBayesClassifier:
    """Naive Bayes from W7 — pre-trained for deployment analysis.

    You built this in W7. Here it's provided pre-trained so you can
    focus on what happens AFTER training: deployment and evaluation.
    """

    def __init__(self, alpha=1):
        self.alpha = alpha
        self.vocab = []
        self.log_prior = {}
        self.log_likelihood = {}

    def fit(self, texts, labels):
        classes = set(labels)
        n_total = len(labels)
        vocab_set = set()
        for text in texts:
            vocab_set.update(text.lower().split())
        self.vocab = sorted(vocab_set)
        vocab_size = len(self.vocab)

        for cls in classes:
            cls_texts = [t for t, l in zip(texts, labels) if l == cls]
            n_cls = len(cls_texts)
            self.log_prior[cls] = np.log(n_cls / n_total)
            word_counts = {}
            for text in cls_texts:
                for word in set(text.lower().split()):
                    word_counts[word] = word_counts.get(word, 0) + 1
            self.log_likelihood[cls] = {}
            for word in self.vocab:
                count = word_counts.get(word, 0)
                prob = (count + self.alpha) / (n_cls + self.alpha * vocab_size)
                self.log_likelihood[cls][word] = np.log(prob)

    def predict_one(self, text):
        words = set(text.lower().split())
        scores = {}
        for cls in self.log_prior:
            score = self.log_prior[cls]
            for word in words:
                if word in self.log_likelihood[cls]:
                    score += self.log_likelihood[cls][word]
            scores[cls] = score
        return max(scores, key=scores.get)

    def predict(self, texts):
        return [self.predict_one(t) for t in texts]


# ---------------------------------------------------------------------------
# Training data (2020-era messages)
# ---------------------------------------------------------------------------

TRAIN_DATA = [
    ('ham',  'hey are you free for lunch today'),
    ('ham',  'meeting at three in the conference room'),
    ('ham',  'can you send me the notes from class'),
    ('ham',  'thanks for your help with the project'),
    ('ham',  'running late be there in ten minutes'),
    ('ham',  'happy birthday hope you have a great day'),
    ('ham',  'see you tomorrow at the office'),
    ('ham',  'do you want to grab coffee later'),
    ('ham',  'great job on the presentation today'),
    ('ham',  'i left my keys can you check your bag'),
    ('ham',  'dinner reservation is at seven tonight'),
    ('ham',  'just finished the report will send now'),
    ('ham',  'what time does the movie start'),
    ('ham',  'ok sounds good see you then'),
    ('ham',  'can you pick up the kids from school'),
    ('spam', 'congratulations you have won a free prize call now'),
    ('spam', 'urgent reply to claim your free cash prize'),
    ('spam', 'free entry to win a brand new phone today'),
    ('spam', 'you have been selected for a free gift call now'),
    ('spam', 'winner claim your cash prize now call today'),
    ('spam', 'free ringtone reply now to download your prize'),
    ('spam', 'act now to receive your exclusive cash bonus'),
    ('spam', 'text win to enter our free weekly prize draw'),
]

# ---------------------------------------------------------------------------
# Test data — same distribution (2020-era)
# ---------------------------------------------------------------------------

TEST_SAME = [
    ('ham',  'are you coming to the party this weekend'),
    ('ham',  'got your message will reply later'),
    ('ham',  'the meeting has been moved to room five'),
    ('ham',  'reminder your dentist appointment is tomorrow at ten'),
    ('ham',  'flight confirmation your booking reference is attached'),
    ('spam', 'claim your free reward now text back to win'),
    ('spam', 'urgent your account needs verification call now'),
    ('spam', 'you won a free holiday reply with your details'),
    ('spam', 'exclusive offer free phone upgrade call today'),
    ('spam', 'congratulations claim your cash prize before midnight'),
]

# ---------------------------------------------------------------------------
# Test data — shifted distribution (2024-era)
# ---------------------------------------------------------------------------

TEST_SHIFTED = [
    ('ham',  'are you coming to the party this weekend'),
    ('ham',  'got your message will reply later'),
    ('ham',  'the meeting has been moved to room five'),
    ('ham',  'reminder your dentist appointment is tomorrow at ten'),
    ('ham',  'flight confirmation your booking reference is attached'),
    ('spam', 'hi i noticed your profile and wanted to connect about an investment opportunity'),
    ('spam', 'your package delivery failed please verify your address at this link'),
    ('spam', 'hey i found an old photo of us check it out here'),
    ('spam', 'your account security update requires immediate action please confirm'),
    ('spam', 'i saw your resume and have a perfect remote position for you'),
]

# ---------------------------------------------------------------------------
# Pre-trained classifier
# ---------------------------------------------------------------------------

classifier = NaiveBayesClassifier(alpha=1)
classifier.fit(
    [text for _, text in TRAIN_DATA],
    [label for label, _ in TRAIN_DATA],
)

# ---------------------------------------------------------------------------
# Carbon estimation constants
# ---------------------------------------------------------------------------

CARBON_CONSTANTS = {
    # GPT-4 training energy: Epoch AI estimate, ~50 GWh (order-of-magnitude;
    # true value not disclosed by OpenAI). Even if off by 2×, the scale
    # comparison with student use holds.
    'gpt4_energy_gwh': 50,
    'us_carbon_intensity': 0.39,          # kg CO2 per kWh (2023 US average, EIA)
    'au_carbon_intensity': 0.66,          # kg CO2 per kWh (Australian average, CER)
    # Semester lab computation
    'semester_students': 200,
    'semester_labs': 12,
    'lab_hours': 2,
    'laptop_power_watts': 30,
    # Student AI inference (ChatGPT / chatbot queries during semester)
    # ~0.3 Wh per query (2025 estimate, Epoch AI); conservative 50 queries/week
    'query_energy_wh': 0.3,
    'student_queries_per_week': 50,
    'semester_weeks': 12,
    # Break-even analysis: estimated GPT-4 daily query volume
    # ~200M weekly active users (OpenAI, 2024) × ~5 queries/day average
    'gpt4_queries_per_day': 1_000_000_000,
}

COMPARISON_VALUES = {
    'car_annual_co2_kg': 4600,            # kg CO2 per year, average US car
    'flight_syd_lon_co2_kg': 2500,        # kg CO2, one-way Sydney to London
    'household_annual_co2_kg': 7500,      # kg CO2 per year, average US household electricity
}

# Historical training cost data for scaling visualisation
TRAINING_COST_HISTORY = [
    ('AlexNet\n(2012)',    0.003),         # tonnes CO2
    ('ResNet\n(2015)',     0.05),
    ('BERT\n(2018)',       1.5),
    ('GPT-3\n(2020)',      1300),
    ('GPT-4\n(2023)',      19500),
]

# ---------------------------------------------------------------------------
# Weak Model Card (for peer comparison exercise)
# ---------------------------------------------------------------------------

WEAK_MODEL_CARD = """
============================================================
MODEL CARD: Spam Classifier
============================================================

1. MODEL DETAILS
   - It's a Naive Bayes classifier
   - Trained on some messages

2. INTENDED USE
   - Classifying messages

3. PERFORMANCE
   - Accuracy: 100%

4. LIMITATIONS
   - None

5. ETHICAL CONSIDERATIONS
   - N/A

6. DEPLOYMENT RECOMMENDATION
   - Ready to deploy
============================================================
"""


# ---------------------------------------------------------------------------
# Utility: compute classification metrics
# ---------------------------------------------------------------------------

def compute_metrics(true_labels, predicted_labels):
    """Compute accuracy, spam recall, and ham precision.

    Args:
        true_labels: List of true labels ('spam' or 'ham').
        predicted_labels: List of predicted labels.

    Returns:
        Dictionary with 'accuracy', 'spam_recall', and 'ham_precision'.
    """
    true_labels = list(true_labels)
    predicted_labels = list(predicted_labels)
    n = len(true_labels)
    if n == 0:
        return {'accuracy': 0.0, 'spam_recall': 0.0, 'ham_precision': 0.0}

    tp = sum(1 for a, p in zip(true_labels, predicted_labels)
             if a == 'spam' and p == 'spam')
    fp = sum(1 for a, p in zip(true_labels, predicted_labels)
             if a == 'ham' and p == 'spam')
    fn = sum(1 for a, p in zip(true_labels, predicted_labels)
             if a == 'spam' and p == 'ham')
    tn = sum(1 for a, p in zip(true_labels, predicted_labels)
             if a == 'ham' and p == 'ham')

    accuracy = (tp + tn) / n
    spam_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    ham_precision = tn / (tn + fn) if (tn + fn) > 0 else 0.0

    return {
        'accuracy': accuracy,
        'spam_recall': spam_recall,
        'ham_precision': ham_precision,
    }


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_shift_comparison(metrics_same, metrics_shifted):
    """Bar chart comparing metrics before and after distribution shift.

    Args:
        metrics_same: Dict from compute_metrics on same-distribution data.
        metrics_shifted: Dict from compute_metrics on shifted data.
    """
    labels = ['Accuracy', 'Spam Recall', 'Ham Precision']
    keys = ['accuracy', 'spam_recall', 'ham_precision']
    vals_same = [metrics_same[k] for k in keys]
    vals_shifted = [metrics_shifted[k] for k in keys]

    x = np.arange(len(labels))
    width = 0.3

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.bar(x - width / 2, vals_same, width, label='Same distribution (2020)',
           color='#2ecc71', edgecolor='black', linewidth=0.5)
    ax.bar(x + width / 2, vals_shifted, width, label='Shifted distribution (2024)',
           color='#e74c3c', edgecolor='black', linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel('Score', fontsize=11)
    ax.set_title('Before and After Distribution Shift',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.grid(True, alpha=0.3, axis='y')

    for i, (v1, v2) in enumerate(zip(vals_same, vals_shifted)):
        ax.text(i - width / 2, v1 + 0.03, f'{v1:.0%}',
                ha='center', fontsize=9, fontweight='bold')
        ax.text(i + width / 2, v2 + 0.03, f'{v2:.0%}',
                ha='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.show()


def plot_carbon_scaling():
    """Bar chart of training carbon footprint over time (log scale)."""
    names = [m[0] for m in TRAINING_COST_HISTORY]
    co2_tonnes = [m[1] for m in TRAINING_COST_HISTORY]

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    bars = ax.bar(range(len(names)), co2_tonnes, color='#e74c3c',
                  edgecolor='black', linewidth=0.5)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylabel('Training CO₂ (tonnes, log scale)', fontsize=11)
    ax.set_title('Training Carbon Footprint: Exponential Growth',
                 fontsize=13, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')

    for bar, val in zip(bars, co2_tonnes):
        label = f'{val:,.0f}t' if val >= 1 else f'{val * 1000:.0f}kg'
        ax.text(bar.get_x() + bar.get_width() / 2, val * 1.5,
                label, ha='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.show()


def plot_efficiency_table():
    """Print an accuracy-vs-compute efficiency comparison table."""
    comparisons = [
        ('Naive Bayes (W7)',       0.95,  0.001),
        ('Perceptron (W8)',        0.93,  0.002),
        ('Logistic Regression',    0.96,  0.01),
        ('Small Neural Network',   0.97,  10),
        ('BERT fine-tuned',        0.98,  100),
        ('GPT-4 (few-shot)',       0.99,  50000),
    ]

    print(f"{'Model':>25s}  {'Accuracy':>8s}  {'Compute':>12s}  {'Acc/kWh':>10s}")
    print("-" * 60)
    for name, acc, kwh in comparisons:
        efficiency = acc / kwh
        print(f"{name:>25s}  {acc:8.0%}  {kwh:10.3f} kWh  {efficiency:10.1f}")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("C4 Helpers — Deployment & Model Cards")
    print(f"  Training data: {len(TRAIN_DATA)} messages")
    print(f"  Test (same): {len(TEST_SAME)} messages")
    print(f"  Test (shifted): {len(TEST_SHIFTED)} messages")
    print(f"  Classifier vocab size: {len(classifier.vocab)}")

    # Quick sanity check
    same_labels = [l for l, _ in TEST_SAME]
    same_texts = [t for _, t in TEST_SAME]
    same_preds = classifier.predict(same_texts)
    same_metrics = compute_metrics(same_labels, same_preds)
    print(f"  Same-dist accuracy: {same_metrics['accuracy']:.0%}")

    shifted_labels = [l for l, _ in TEST_SHIFTED]
    shifted_texts = [t for _, t in TEST_SHIFTED]
    shifted_preds = classifier.predict(shifted_texts)
    shifted_metrics = compute_metrics(shifted_labels, shifted_preds)
    print(f"  Shifted-dist accuracy: {shifted_metrics['accuracy']:.0%}")
