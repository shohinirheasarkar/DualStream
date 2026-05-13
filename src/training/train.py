"""
train.py

Training loop for DualStream prototype.
"""

from pathlib import Path
import json
import random

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from src.models.dual_stream_model import DualStreamModel
from src.losses.losses import compute_total_loss
from src.evaluation.metrics import evaluate_graph_recovery


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_windows(X, window_size=200, stride=100):
    windows = []

    for start in range(0, X.shape[1] - window_size + 1, stride):
        windows.append(X[:, start:start + window_size])

    return np.stack(windows, axis=0).astype(np.float32)


def plot_loss_curve(train_losses, val_losses, output_path):
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("DualStream Training Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_heatmap(matrix, title, output_path):
    plt.figure(figsize=(6, 5))
    plt.imshow(matrix, aspect="auto")
    plt.colorbar(label="Weight")
    plt.title(title)
    plt.xlabel("Source neuron")
    plt.ylabel("Target neuron")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def train_dualstream(
    observed_dff_path="data/synthetic/example_001/observed_dff.npy",
    true_graph_path="data/synthetic/example_001/true_graph.npy",
    output_dir="outputs/dualstream_prototype",
    seed=42,
    window_size=200,
    stride=100,
    epochs=100,
    patience=15,
    lr=1e-3,
):
    """
    Full training loop:
        load synthetic traces
        window data
        train model
        compute C from fast latent stream
        evaluate against true graph
        save outputs
    """

    set_seed(seed)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    X = np.load(observed_dff_path).astype(np.float32)
    true_graph = np.load(true_graph_path).astype(np.float32)

    # Normalize each neuron.
    X = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-8)

    windows = make_windows(X, window_size=window_size, stride=stride)

    split_idx = int(0.8 * len(windows))

    train_windows = windows[:split_idx]
    val_windows = windows[split_idx:]

    train_tensor = torch.tensor(train_windows, dtype=torch.float32).to(device)
    val_tensor = torch.tensor(val_windows, dtype=torch.float32).to(device)

    print(f"Train windows: {train_tensor.shape}")
    print(f"Val windows: {val_tensor.shape}")

    model = DualStreamModel().to(device)

    weights = {
        "reconstruction": 1.0,
        "future_prediction": 0.5,
        "fast_sparsity": 0.001,
        "slow_smoothness": 0.05,
        "decorrelation": 0.05,
        "connectivity_sparsity": 0.001,
        "variance_preservation": 0.1,
        "orthogonality": 0.01,
        "graph_stability": 0.0,
    }

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=1e-4,
    )

    best_val = float("inf")
    bad_epochs = 0

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        outputs = model(train_tensor)
        losses = compute_total_loss(outputs, train_tensor, weights)

        losses["total"].backward()

        # Gradient clipping.
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_outputs = model(val_tensor)
            val_loss_dict = compute_total_loss(val_outputs, val_tensor, weights)

        train_loss = losses["total"].item()
        val_loss = val_loss_dict["total"].item()

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if epoch % 10 == 0:
            print(
                f"Epoch {epoch:03d} | "
                f"Train {train_loss:.4f} | "
                f"Val {val_loss:.4f} | "
                f"Recon {losses['reconstruction'].item():.4f} | "
                f"Pred {losses['future_prediction'].item():.4f}"
            )

        if val_loss < best_val:
            best_val = val_loss
            bad_epochs = 0
            torch.save(model.state_dict(), output_dir / "best_model.pt")
        else:
            bad_epochs += 1

        if bad_epochs >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    # Load best model.
    model.load_state_dict(torch.load(output_dir / "best_model.pt", map_location=device))
    model.eval()

    all_tensor = torch.tensor(windows, dtype=torch.float32).to(device)

    with torch.no_grad():
        final_outputs = model(all_tensor)
        final_losses = compute_total_loss(final_outputs, all_tensor, weights)

    C = final_outputs["connectivity"].detach().cpu().numpy()

    metrics = evaluate_graph_recovery(C, true_graph, k=20)
    metrics["ReconstructionError"] = float(final_losses["reconstruction"].item())
    metrics["FuturePredictionError"] = float(final_losses["future_prediction"].item())

    print("\nFinal metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    np.save(output_dir / "dualstream_connectivity.npy", C)
    pd.DataFrame(C).to_csv(output_dir / "dualstream_connectivity.csv", index=False)

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    plot_loss_curve(
        train_losses,
        val_losses,
        output_dir / "training_loss_curve.png",
    )

    save_heatmap(
        true_graph,
        "Ground Truth Directed Graph",
        output_dir / "ground_truth_graph.png",
    )

    save_heatmap(
        C,
        "DualStream Learned Connectivity",
        output_dir / "dualstream_connectivity.png",
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].imshow(true_graph, aspect="auto")
    axes[0].set_title("Ground Truth")
    axes[0].set_xlabel("Source")
    axes[0].set_ylabel("Target")

    axes[1].imshow(C, aspect="auto")
    axes[1].set_title("DualStream Learned C")
    axes[1].set_xlabel("Source")
    axes[1].set_ylabel("Target")

    plt.tight_layout()
    plt.savefig(output_dir / "ground_truth_vs_dualstream.png", dpi=300)
    plt.close()

    return metrics