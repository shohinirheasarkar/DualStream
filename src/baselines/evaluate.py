"""
evaluate.py

Baseline connectivity methods + evaluation utilities for NeuroMamba.

This file implements several classical connectivity estimation methods:

1. Pearson correlation
2. Lagged correlation
3. Granger causality

It also includes:

- comparison against the known synthetic graph
- ROC-AUC scoring
- Precision@K
- connectivity heatmaps
- diagnostic plots for posters/reports

The goal is to compare simple classical methods against the future
DualStream / NeuroMamba architecture.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score
from statsmodels.tsa.stattools import grangercausalitytests
from sklearn.metrics import roc_curve


# ============================================================
# 1. Pearson correlation
# ============================================================


def pearson_connectivity(traces: np.ndarray) -> np.ndarray:
    """
    Compute a Pearson correlation connectivity matrix.

    Input shape:
        (neurons, time)

    Output shape:
        (neurons, neurons)

    Important:
        Pearson correlation is symmetric and undirected.
    """

    print("\nComputing Pearson correlation matrix...")

    corr = np.corrcoef(traces)

    # Remove self-connections.
    np.fill_diagonal(corr, 0)

    return corr.astype(np.float32)


# ============================================================
# 2. Lagged correlation
# ============================================================


def lagged_correlation_connectivity(
    traces: np.ndarray,
    lag: int = 3,
) -> np.ndarray:
    """
    Compute a directed lagged correlation matrix.

    Direction convention:
        connectivity[target, source]

    Idea:
        Does neuron j at time t-lag correlate with
        neuron i at time t?
    """

    print("\nComputing lagged correlation matrix...")
    print(f"Lag: {lag}")

    num_neurons, num_timepoints = traces.shape

    connectivity = np.zeros((num_neurons, num_neurons), dtype=np.float32)

    for target in range(num_neurons):
        for source in range(num_neurons):

            if target == source:
                continue

            source_past = traces[source, :-lag]
            target_future = traces[target, lag:]

            corr = np.corrcoef(source_past, target_future)[0, 1]

            if np.isnan(corr):
                corr = 0.0

            connectivity[target, source] = corr

    return connectivity.astype(np.float32)


# ============================================================
# 3. Granger causality
# ============================================================


def granger_connectivity_matrix(
    traces: np.ndarray,
    max_lag: int = 5,
) -> np.ndarray:
    """
    Compute a directed Granger causality matrix.

    Direction convention:
        connectivity[target, source]

    We use the minimum p-value across lags and convert it into:

        score = -log10(p)

    so stronger Granger evidence becomes larger.
    """

    print("\nComputing Granger causality matrix...")
    print(f"Max lag: {max_lag}")

    num_neurons = traces.shape[0]

    connectivity = np.zeros((num_neurons, num_neurons), dtype=np.float32)

    for target in range(num_neurons):
        for source in range(num_neurons):

            if target == source:
                continue

            try:
                data = np.column_stack(
                    [
                        traces[target],
                        traces[source],
                    ]
                )

                results = grangercausalitytests(
                    data,
                    maxlag=max_lag,
                    verbose=False,
                )

                p_values = []

                for lag in range(1, max_lag + 1):
                    p = results[lag][0]["ssr_ftest"][1]
                    p_values.append(p)

                best_p = np.min(p_values)

                score = -np.log10(best_p + 1e-12)

                connectivity[target, source] = score

            except Exception:
                connectivity[target, source] = 0.0

    return connectivity.astype(np.float32)


# ============================================================
# 4. Thresholding utility
# ============================================================


def threshold_matrix(
    matrix: np.ndarray,
    percentile: float = 90,
) -> np.ndarray:
    """
    Convert a weighted matrix into a binary graph.

    Keeps only the strongest edges.
    """

    threshold = np.percentile(matrix[matrix != 0], percentile)

    binary = (matrix >= threshold).astype(np.float32)

    np.fill_diagonal(binary, 0)

    return binary


# ============================================================
# 5. Precision@K
# ============================================================


def precision_at_k(
    predicted: np.ndarray,
    true_graph: np.ndarray,
    k: int = 20,
) -> float:
    """
    Precision among the top-K strongest predicted edges.
    """

    pred = predicted.flatten()
    truth = (true_graph.flatten() > 0).astype(np.float32)

    top_k_indices = np.argsort(pred)[-k:]

    correct = truth[top_k_indices].sum()

    precision = correct / k

    return float(precision)


# ============================================================
# 6. ROC-AUC
# ============================================================


def compute_roc_auc(
    predicted: np.ndarray,
    true_graph: np.ndarray,
) -> float:
    """
    Compute ROC-AUC against the true graph.

    Ground truth:
        edge exists or not.
    """

    y_true = (true_graph.flatten() > 0).astype(np.float32)
    y_score = predicted.flatten()

    mask = ~np.eye(true_graph.shape[0], dtype=bool).flatten()

    auc = roc_auc_score(y_true[mask], y_score[mask])

    return float(auc)


# ============================================================
# 7. Heatmap plotting
# ============================================================


def plot_heatmap(
    matrix: np.ndarray,
    title: str,
    save_path: str,
):
    """
    Save a connectivity heatmap.
    """

    plt.figure(figsize=(6, 5))

    plt.imshow(matrix, aspect="auto")
    plt.colorbar(label="Connectivity strength")

    plt.title(title)
    plt.xlabel("Source neuron")
    plt.ylabel("Target neuron")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_binary_edge_recovery(
    predicted: np.ndarray,
    true_graph: np.ndarray,
    title: str,
    save_path: str,
    percentile: float = 90,
):
    """
    Compare recovered binary edges against the true graph.
    """

    pred_binary = threshold_matrix(predicted, percentile=percentile)
    true_binary = (true_graph > 0).astype(np.float32)

    comparison = np.zeros_like(true_binary)

    # True positive
    comparison[(pred_binary == 1) & (true_binary == 1)] = 1

    # False positive
    comparison[(pred_binary == 1) & (true_binary == 0)] = 2

    # False negative
    comparison[(pred_binary == 0) & (true_binary == 1)] = 3

    plt.figure(figsize=(6, 5))

    plt.imshow(comparison, aspect="auto")

    cbar = plt.colorbar()
    cbar.set_ticks([0, 1, 2, 3])
    cbar.set_ticklabels(
        [
            "True Negative",
            "True Positive",
            "False Positive",
            "False Negative",
        ]
    )

    plt.title(title)
    plt.xlabel("Source neuron")
    plt.ylabel("Target neuron")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_roc_curves(
    methods: dict,
    true_graph: np.ndarray,
    save_path: str,
):
    """
    Plot ROC curves for all methods.
    """

    plt.figure(figsize=(6, 6))

    y_true = (true_graph.flatten() > 0).astype(np.float32)

    mask = ~np.eye(true_graph.shape[0], dtype=bool).flatten()

    for name, matrix in methods.items():

        y_score = matrix.flatten()

        fpr, tpr, _ = roc_curve(
            y_true[mask],
            y_score[mask],
        )

        auc = compute_roc_auc(matrix, true_graph)

        plt.plot(
            fpr,
            tpr,
            label=f"{name} (AUC={auc:.3f})",
        )

    plt.plot([0, 1], [0, 1], linestyle="--")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves for Connectivity Recovery")
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


# ============================================================
# 8. Main evaluation visualization
# ============================================================


def compare_methods(
    traces: np.ndarray,
    true_graph: np.ndarray,
    output_dir: str = "outputs/figures/baselines",
):
    """
    Run all baseline methods and compare them.
    """

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    pearson = pearson_connectivity(traces)
    lagged = lagged_correlation_connectivity(traces)
    granger = granger_connectivity_matrix(traces)

    methods = {
        "Pearson": pearson,
        "LaggedCorrelation": lagged,
        "Granger": granger,
    }

    print("\n" + "=" * 60)
    print("BASELINE COMPARISON RESULTS")
    print("=" * 60)

    plot_heatmap(
        true_graph,
        title="Ground Truth Graph",
        save_path=output_path / "ground_truth_graph.png",
    )

    for name, matrix in methods.items():

        auc = compute_roc_auc(matrix, true_graph)
        p_at_20 = precision_at_k(matrix, true_graph, k=20)

        print(f"\n{name}")
        print(f"ROC-AUC: {auc:.4f}")
        print(f"Precision@20: {p_at_20:.4f}")

        plot_heatmap(
            matrix,
            title=f"{name} Connectivity Matrix",
            save_path=output_path / f"{name.lower()}_heatmap.png",
        )

        plot_binary_edge_recovery(
            predicted=matrix,
            true_graph=true_graph,
            title=f"{name} Edge Recovery",
            save_path=output_path / f"{name.lower()}_edge_recovery.png",
        )

    # --------------------------------------------------------
    # Combined poster/report figure
    # --------------------------------------------------------

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))

    matrices = [
        ("Ground Truth", true_graph),
        ("Pearson", pearson),
        ("Lagged", lagged),
        ("Granger", granger),
    ]

    for ax, (title, matrix) in zip(axes, matrices):
        im = ax.imshow(matrix, aspect="auto")
        ax.set_title(title)
        ax.set_xlabel("Source")
        ax.set_ylabel("Target")

    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8)

    plt.tight_layout()

    combined_path = output_path / "baseline_comparison_panel.png"

    plt.savefig(combined_path, dpi=300)
    plt.close()

    plot_roc_curves(
        methods=methods,
        true_graph=true_graph,
        save_path=output_path / "roc_curves.png",
    )

    print(f"\nSaved visualizations to: {output_path}")


# ============================================================
# 9. Standalone demo
# ============================================================


if __name__ == "__main__":
    """
    Run this file directly with:

        python src/baselines/evaluate.py

    It loads the synthetic benchmark generated earlier and compares
    baseline methods against the known graph.
    """

    print("\nLoading synthetic benchmark...")

    base_path = Path("data/synthetic/example_001")

    traces = np.load(base_path / "observed_dff.npy")
    true_graph = np.load(base_path / "true_graph.npy")

    print(f"Loaded traces shape: {traces.shape}")
    print(f"Loaded graph shape: {true_graph.shape}")

    compare_methods(
        traces=traces,
        true_graph=true_graph,
        output_dir="outputs/figures/baseline_comparisons",
    )

    print("\nBaseline benchmark complete.")