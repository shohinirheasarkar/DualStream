"""
connectivity_matrix.py

Utilities for computing a directed neuron-to-neuron connectivity matrix C.

Goal
----
Given neural activity or fast latent traces shaped:

    (neurons, time)

or batched traces shaped:

    (batch, neurons, time)

compute a directed matrix:

    C[target, source]

where C[i, j] estimates how strongly the past activity of source neuron j
predicts the present/future activity of target neuron i.

This is the standalone prototype version of the connectivity computation.
Later, this can be integrated directly into the DualStream model's
connectivity head.

Core idea
---------
For each target neuron i and source neuron j:

    compare source_j(t - lag) with target_i(t)

If source j tends to be active shortly before target i, then C[i, j] should be high.
"""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. Shape helper
# ============================================================

def ensure_3d(traces: np.ndarray) -> np.ndarray:
    """
    Make sure traces are shaped:

        (batch, neurons, time)

    Accepts:
        (neurons, time)
        (batch, neurons, time)
    """

    traces = np.asarray(traces)

    if traces.ndim == 2:
        traces = traces[None, :, :]

    if traces.ndim != 3:
        raise ValueError(
            f"Expected traces with shape (neurons, time) or "
            f"(batch, neurons, time), but got {traces.shape}"
        )

    return traces.astype(np.float32)


# ============================================================
# 2. Standardization helper
# ============================================================

def zscore_per_neuron(traces: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Z-score each neuron independently.

    This prevents high-amplitude neurons from dominating the connectivity score.
    """

    traces = ensure_3d(traces)

    mean = traces.mean(axis=-1, keepdims=True)
    std = traces.std(axis=-1, keepdims=True)

    return ((traces - mean) / (std + eps)).astype(np.float32)


# ============================================================
# 3. Lagged similarity score
# ============================================================

def compute_lagged_score(
    traces: np.ndarray,
    lag: int,
) -> np.ndarray:
    """
    Compute directed lagged similarity for one lag.

    Direction:
        C[target, source]

    For each target i and source j:

        C[i, j] = corr(source_j at time t-lag, target_i at time t)

    Input:
        traces shape = (batch, neurons, time)

    Output:
        scores shape = (batch, neurons, neurons)
    """

    traces = ensure_3d(traces)

    batch_size, num_neurons, num_timepoints = traces.shape

    if lag <= 0:
        raise ValueError("lag must be positive.")

    if lag >= num_timepoints:
        raise ValueError("lag must be smaller than number of timepoints.")

    source_past = traces[:, :, :-lag]
    target_present = traces[:, :, lag:]

    scores = np.zeros(
        (batch_size, num_neurons, num_neurons),
        dtype=np.float32,
    )

    for b in range(batch_size):
        for target in range(num_neurons):
            for source in range(num_neurons):

                if target == source:
                    continue

                x = source_past[b, source]
                y = target_present[b, target]

                corr = np.corrcoef(x, y)[0, 1]

                if np.isnan(corr):
                    corr = 0.0

                scores[b, target, source] = corr

    return scores


# ============================================================
# 4. Multi-lag aggregation
# ============================================================

def compute_multilag_connectivity(
    traces: np.ndarray,
    min_lag: int = 1,
    max_lag: int = 5,
    use_absolute: bool = True,
) -> np.ndarray:
    """
    Compute directed connectivity by aggregating over multiple lags.

    Steps:
        1. Standardize traces
        2. Compute lagged source→target similarity for each lag
        3. Aggregate scores across lags
        4. Average across batch/windows

    Output:
        C shape = (neurons, neurons)
    """

    print("\nComputing multi-lag directed connectivity...")
    print(f"Min lag: {min_lag}")
    print(f"Max lag: {max_lag}")

    traces = zscore_per_neuron(traces)

    all_lag_scores = []

    for lag in range(min_lag, max_lag + 1):
        print(f"  Computing lag {lag}...")
        lag_scores = compute_lagged_score(traces, lag=lag)

        if use_absolute:
            lag_scores = np.abs(lag_scores)

        all_lag_scores.append(lag_scores)

    all_lag_scores = np.stack(all_lag_scores, axis=0)

    # Average across lags.
    avg_over_lags = all_lag_scores.mean(axis=0)

    # Average across batch/windows.
    C = avg_over_lags.mean(axis=0)

    np.fill_diagonal(C, 0)

    print(f"Connectivity matrix shape: {C.shape}")

    return C.astype(np.float32)


# ============================================================
# 5. Sparsification
# ============================================================

def sparsify_top_k(C: np.ndarray, k: int = 20) -> np.ndarray:
    """
    Keep only the top-k strongest edges.

    This produces a sparse directed matrix.
    """

    C_sparse = np.zeros_like(C)

    flat = C.flatten()

    # Ignore diagonal by temporarily forcing it low.
    C_no_diag = C.copy()
    np.fill_diagonal(C_no_diag, -np.inf)

    flat_no_diag = C_no_diag.flatten()

    k = min(k, np.isfinite(flat_no_diag).sum())

    top_indices = np.argsort(flat_no_diag)[-k:]

    C_sparse.flat[top_indices] = C.flat[top_indices]

    np.fill_diagonal(C_sparse, 0)

    return C_sparse.astype(np.float32)


def row_normalize(C: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Normalize each target row so incoming source weights sum to 1.

    Direction:
        row = target
        column = source
    """

    row_sums = C.sum(axis=1, keepdims=True)

    return (C / (row_sums + eps)).astype(np.float32)


# ============================================================
# 6. Saving and plotting
# ============================================================

def save_connectivity_matrix(
    C: np.ndarray,
    output_dir: str = "outputs/connectivity",
    name: str = "directed_connectivity",
) -> None:
    """
    Save connectivity matrix as:
        .npy
        .csv
        .png heatmap
    """

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    npy_path = output_path / f"{name}.npy"
    csv_path = output_path / f"{name}.csv"
    png_path = output_path / f"{name}.png"

    np.save(npy_path, C)
    pd.DataFrame(C).to_csv(csv_path, index=False)

    plt.figure(figsize=(6, 5))
    plt.imshow(C, aspect="auto")
    plt.colorbar(label="Connectivity strength")
    plt.title("Directed Connectivity Matrix\nrows = targets, columns = sources")
    plt.xlabel("Source neuron")
    plt.ylabel("Target neuron")
    plt.tight_layout()
    plt.savefig(png_path, dpi=300)
    plt.close()

    print("\nSaved connectivity matrix:")
    print(f"  {npy_path}")
    print(f"  {csv_path}")
    print(f"  {png_path}")


# ============================================================
# 7. Full wrapper
# ============================================================

def compute_and_save_connectivity(
    traces: np.ndarray,
    output_dir: str = "outputs/connectivity",
    name: str = "directed_connectivity",
    min_lag: int = 1,
    max_lag: int = 5,
    top_k: Optional[int] = 20,
    normalize: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Full pipeline:
        1. Compute directed multi-lag connectivity
        2. Sparsify
        3. Normalize
        4. Save matrix

    Returns:
        C_dense
        C_final
    """

    C_dense = compute_multilag_connectivity(
        traces=traces,
        min_lag=min_lag,
        max_lag=max_lag,
    )

    if top_k is not None:
        C_final = sparsify_top_k(C_dense, k=top_k)
    else:
        C_final = C_dense

    if normalize:
        C_final = row_normalize(C_final)

    save_connectivity_matrix(
        C_final,
        output_dir=output_dir,
        name=name,
    )

    return C_dense, C_final


# ============================================================
# 8. Standalone demo
# ============================================================

if __name__ == "__main__":
    """
    Run with:

        python src/models/connectivity_matrix.py

    This loads the synthetic observed ΔF/F traces and computes a directed matrix.
    """

    print("\nLoading synthetic observed ΔF/F traces...")

    traces = np.load("data/synthetic/example_001/observed_dff.npy")

    print(f"Loaded traces shape: {traces.shape}")

    C_dense, C_sparse = compute_and_save_connectivity(
        traces=traces,
        output_dir="outputs/connectivity/synthetic_example_001",
        name="lagged_directed_connectivity",
        min_lag=1,
        max_lag=5,
        top_k=20,
        normalize=True,
    )

    print("\nDone computing directed connectivity.")