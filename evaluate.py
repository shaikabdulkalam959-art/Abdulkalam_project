"""
Shared metrics used to evaluate both models, plus the confusion-matrix
figure saved to reports/figures/ (referenced in the README's "reports/" section).
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, f1_score

from src.config import FIGURES_DIR


def classifier_scores(y_true, y_pred) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
    }


def save_confusion_matrix(y_true, y_pred, labels, filename="confusion_matrix.png"):
    fig, ax = plt.subplots(figsize=(10, 10))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=labels, xticks_rotation="vertical", ax=ax
    )
    fig.tight_layout()
    out_path = FIGURES_DIR / filename
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def recommender_precision_at_k(relevant_job_ids: set, ranked_job_ids: list, k: int = 10) -> float:
    """
    Precision@k for the recommender: of the top-k ranked jobs, what fraction
    are in the known-relevant set (e.g. jobs sharing the resume's true category).
    """
    top_k = ranked_job_ids[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for jid in top_k if jid in relevant_job_ids)
    return hits / len(top_k)


def mean_reciprocal_rank(relevant_job_ids: set, ranked_job_ids: list) -> float:
    for rank, jid in enumerate(ranked_job_ids, start=1):
        if jid in relevant_job_ids:
            return 1.0 / rank
    return 0.0
