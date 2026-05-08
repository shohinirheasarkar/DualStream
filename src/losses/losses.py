"""
losses.py

Loss functions for the DualStream / NeuroMamba prototype.

These losses are designed to encourage the model to:

1. Reconstruct the observed calcium signal
2. Predict future calcium activity
3. Keep the fast stream sparse / event-like
4. Keep the slow stream smooth
5. Encourage fast and slow streams to capture different information
6. Encourage sparse connectivity
7. Preserve enough variance so latent representations do not collapse
"""

from typing import Dict

import torch
import torch.nn.functional as F


# ============================================================
# 1. Reconstruction loss
# ============================================================

def reconstruction_loss(reconstruction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Measures how well the model reconstructs the original calcium traces.

    reconstruction shape:
        (batch, time, neurons)

    target shape:
        (batch, neurons, time)

    We transpose target so shapes match.
    """

    target = target.permute(0, 2, 1)

    return F.mse_loss(reconstruction, target)


# ============================================================
# 2. Future prediction loss
# ============================================================

def future_prediction_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Measures whether the model predicts the next time step.

    prediction shape:
        (batch, time, neurons)

    target shape:
        (batch, neurons, time)

    We compare prediction at time t to true activity at time t+1.
    """

    target = target.permute(0, 2, 1)

    pred_trimmed = prediction[:, :-1, :]
    target_future = target[:, 1:, :]

    return F.mse_loss(pred_trimmed, target_future)


# ============================================================
# 3. Fast sparsity loss
# ============================================================

def fast_sparsity_loss(fast_latent: torch.Tensor) -> torch.Tensor:
    """
    Encourages the fast stream to be sparse / event-like.

    Intuition:
        Fast neuronal transients should not be active everywhere all the time.

    Uses L1 penalty.
    """

    return torch.mean(torch.abs(fast_latent))


# ============================================================
# 4. Slow smoothness loss
# ============================================================

def slow_smoothness_loss(slow_latent: torch.Tensor) -> torch.Tensor:
    """
    Encourages the slow stream to change gradually over time.

    Penalizes large differences between consecutive time points.
    """

    diffs = slow_latent[:, 1:, :] - slow_latent[:, :-1, :]

    return torch.mean(diffs ** 2)


# ============================================================
# 5. Fast / slow decorrelation loss
# ============================================================

def decorrelation_loss(fast_latent: torch.Tensor, slow_latent: torch.Tensor) -> torch.Tensor:
    """
    Encourages fast and slow streams to capture different information.

    We compare feature correlation after matching dimensions.
    """

    min_dim = min(fast_latent.shape[-1], slow_latent.shape[-1])

    fast = fast_latent[..., :min_dim]
    slow = slow_latent[..., :min_dim]

    fast = fast - fast.mean(dim=1, keepdim=True)
    slow = slow - slow.mean(dim=1, keepdim=True)

    fast = fast / (fast.std(dim=1, keepdim=True) + 1e-8)
    slow = slow / (slow.std(dim=1, keepdim=True) + 1e-8)

    corr = torch.mean(fast * slow)

    return corr ** 2


# ============================================================
# 6. Connectivity sparsity loss
# ============================================================

def connectivity_sparsity_loss(connectivity: torch.Tensor) -> torch.Tensor:
    """
    Encourages sparse connectivity.

    Current prototype connectivity is temporal attention:
        (batch, time, time)

    Later this will become:
        (batch, neurons, neurons)
    """

    return torch.mean(torch.abs(connectivity))


# ============================================================
# 7. Variance preservation loss
# ============================================================

def variance_preservation_loss(latent: torch.Tensor, min_variance: float = 0.1) -> torch.Tensor:
    """
    Prevents latent representations from collapsing to constants.

    If variance is below min_variance, penalize it.
    """

    variance = torch.var(latent)

    return F.relu(min_variance - variance)


# ============================================================
# 8. Optional orthogonality loss
# ============================================================

def orthogonality_loss(fast_latent: torch.Tensor, slow_latent: torch.Tensor) -> torch.Tensor:
    """
    Optional stronger version of decorrelation.

    Encourages fast and slow streams to point in different latent directions.
    """

    min_dim = min(fast_latent.shape[-1], slow_latent.shape[-1])

    fast = fast_latent[..., :min_dim]
    slow = slow_latent[..., :min_dim]

    fast = F.normalize(fast, dim=-1)
    slow = F.normalize(slow, dim=-1)

    dot = torch.sum(fast * slow, dim=-1)

    return torch.mean(dot ** 2)


# ============================================================
# 9. Optional graph stability loss
# ============================================================

def graph_stability_loss(connectivity_sequence: torch.Tensor) -> torch.Tensor:
    """
    Optional future loss.

    Encourages connectivity matrices from nearby windows to be similar.

    Expected future shape:
        (num_windows, neurons, neurons)

    For now this function works for any first dimension sequence.
    """

    if connectivity_sequence.shape[0] < 2:
        return torch.tensor(0.0, device=connectivity_sequence.device)

    diffs = connectivity_sequence[1:] - connectivity_sequence[:-1]

    return torch.mean(diffs ** 2)


# ============================================================
# 10. Total loss wrapper
# ============================================================

def compute_total_loss(
    outputs: Dict[str, torch.Tensor],
    batch: torch.Tensor,
    weights: Dict[str, float] = None,
) -> Dict[str, torch.Tensor]:
    """
    Computes all losses and combines them.

    outputs:
        dictionary returned by DualStreamModel

    batch:
        original input batch shaped (batch, neurons, time)

    weights:
        dictionary controlling loss importance
    """

    if weights is None:
        weights = {
            "reconstruction": 1.0,
            "future_prediction": 0.5,
            "fast_sparsity": 0.01,
            "slow_smoothness": 0.1,
            "decorrelation": 0.1,
            "connectivity_sparsity": 0.01,
            "variance_preservation": 0.1,
            "orthogonality": 0.0,
            "graph_stability": 0.0,
        }

    losses = {}

    losses["reconstruction"] = reconstruction_loss(
        outputs["reconstruction"],
        batch,
    )

    losses["future_prediction"] = future_prediction_loss(
        outputs["future_prediction"],
        batch,
    )

    losses["fast_sparsity"] = fast_sparsity_loss(
        outputs["fast_latent"],
    )

    losses["slow_smoothness"] = slow_smoothness_loss(
        outputs["slow_latent"],
    )

    losses["decorrelation"] = decorrelation_loss(
        outputs["fast_latent"],
        outputs["slow_latent"],
    )

    losses["connectivity_sparsity"] = connectivity_sparsity_loss(
        outputs["connectivity"],
    )

    losses["variance_preservation"] = variance_preservation_loss(
        outputs["fused_latent"],
    )

    losses["orthogonality"] = orthogonality_loss(
        outputs["fast_latent"],
        outputs["slow_latent"],
    )

    losses["graph_stability"] = torch.tensor(
        0.0,
        device=batch.device,
    )

    total = torch.tensor(
        0.0,
        device=batch.device,
    )

    for name, loss_value in losses.items():
        weight = weights.get(name, 0.0)
        total = total + weight * loss_value

    losses["total"] = total

    return losses


# ============================================================
# 11. Standalone sanity check
# ============================================================

if __name__ == "__main__":

    print("Running loss function sanity check...")

    batch_size = 2
    neurons = 12
    time = 100

    fake_batch = torch.randn(batch_size, neurons, time)

    fake_outputs = {
        "reconstruction": torch.randn(batch_size, time, neurons),
        "future_prediction": torch.randn(batch_size, time, neurons),
        "fast_latent": torch.randn(batch_size, time, 32),
        "slow_latent": torch.randn(batch_size, time, 64),
        "connectivity": torch.softmax(torch.randn(batch_size, time, time), dim=-1),
        "fused_latent": torch.randn(batch_size, time, 64),
    }

    losses = compute_total_loss(fake_outputs, fake_batch)

    print("\nLoss values:")
    for name, value in losses.items():
        print(f"{name}: {value.item():.6f}")

    print("\nLoss sanity check complete.")