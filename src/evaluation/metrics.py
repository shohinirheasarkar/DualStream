"""
metrics.py

Evaluation metrics for synthetic graph recovery.
"""

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
)


def evaluate_graph_recovery(
    predicted_C: np.ndarray,
    true_graph: np.ndarray,
    k: int = 20,
) -> dict:
    """
    Evaluate predicted directed connectivity against known synthetic graph.

    predicted_C:
        [neurons, neurons]

    true_graph:
        [neurons, neurons]
    """

    C = predicted_C.copy()
    G = true_graph.copy()

    np.fill_diagonal(C, 0)
    np.fill_diagonal(G, 0)

    mask = ~np.eye(G.shape[0], dtype=bool)

    y_true = (G[mask] > 0).astype(int)
    y_score = C[mask]

    auroc = roc_auc_score(y_true, y_score)
    auprc = average_precision_score(y_true, y_score)

    k = min(k, len(y_score))

    top_idx = np.argsort(y_score)[-k:]

    y_pred_top = np.zeros_like(y_true)
    y_pred_top[top_idx] = 1

    true_positives = y_true[top_idx].sum()

    precision_at_k = true_positives / k
    recall_at_k = true_positives / max(y_true.sum(), 1)
    f1_at_k = f1_score(y_true, y_pred_top, zero_division=0)

    return {
        "AUROC": float(auroc),
        "AUPRC": float(auprc),
        f"Precision@{k}": float(precision_at_k),
        f"Recall@{k}": float(recall_at_k),
        "F1@K": float(f1_at_k),
        "EdgeDirectionAccuracy@K": float(precision_at_k),
    }


def reconstruction_error(reconstruction, target):
    return float(np.mean((reconstruction - target) ** 2))


def future_prediction_error(prediction, target):
    return float(np.mean((prediction[:, :, :-1] - target[:, :, 1:]) ** 2))