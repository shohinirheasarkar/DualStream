"""
losses.py

Loss functions for the DualStream prototype.

All losses are designed around model outputs from src/models/dual_stream_model.py.
"""

from typing import Dict

import torch
import torch.nn.functional as F


def reconstruction_loss(reconstruction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    reconstruction:
        [batch, neurons, time]

    target:
        [batch, neurons, time]
    """

    return F.mse_loss(reconstruction, target)


def future_prediction_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    prediction:
        [batch, neurons, time]

    target:
        [batch, neurons, time]

    prediction[:, :, t] should predict target[:, :, t+1].
    """

    return F.mse_loss(prediction[:, :, :-1], target[:, :, 1:])


def fast_sparsity_loss(fast_latent: torch.Tensor) -> torch.Tensor:
    """
    Encourages fast latent stream to be sparse/event-like.
    """

    return torch.mean(torch.abs(fast_latent))


def slow_smoothness_loss(slow_latent: torch.Tensor) -> torch.Tensor:
    """
    Encourages slow latent stream to change gradually over time.
    """

    diffs = slow_latent[:, :, 1:, :] - slow_latent[:, :, :-1, :]
    return torch.mean(diffs ** 2)


def decorrelation_loss(fast_latent: torch.Tensor, slow_latent: torch.Tensor) -> torch.Tensor:
    """
    Encourages fast and slow streams to avoid learning identical representations.
    """

    min_dim = min(fast_latent.shape[-1], slow_latent.shape[-1])

    fast = fast_latent[..., :min_dim]
    slow = slow_latent[..., :min_dim]

    fast = (fast - fast.mean()) / (fast.std() + 1e-8)
    slow = (slow - slow.mean()) / (slow.std() + 1e-8)

    return torch.mean(fast * slow) ** 2


def connectivity_sparsity_loss(connectivity: torch.Tensor) -> torch.Tensor:
    """
    Encourages compact connectivity.

    Note:
    Since the current connectivity is row-normalized, this term is mostly a
    gentle placeholder. Later, sparsemax/entmax or top-k can enforce stronger sparsity.
    """

    return torch.mean(torch.abs(connectivity))


def variance_preservation_loss(latent: torch.Tensor, min_variance: float = 0.05) -> torch.Tensor:
    """
    Prevents latent collapse.
    """

    return F.relu(min_variance - torch.var(latent))


def orthogonality_loss(fast_latent: torch.Tensor, slow_latent: torch.Tensor) -> torch.Tensor:
    """
    Optional stronger separation between fast and slow streams.
    """

    min_dim = min(fast_latent.shape[-1], slow_latent.shape[-1])

    fast = F.normalize(fast_latent[..., :min_dim], dim=-1)
    slow = F.normalize(slow_latent[..., :min_dim], dim=-1)

    dot = torch.sum(fast * slow, dim=-1)

    return torch.mean(dot ** 2)


def graph_stability_loss(connectivity_sequence: torch.Tensor) -> torch.Tensor:
    """
    Optional future loss for time-varying graphs.

    For this prototype, we usually only have one graph, so this can be off.
    """

    if connectivity_sequence.shape[0] < 2:
        return torch.tensor(0.0, device=connectivity_sequence.device)

    diffs = connectivity_sequence[1:] - connectivity_sequence[:-1]
    return torch.mean(diffs ** 2)


def compute_total_loss(
    outputs: Dict[str, torch.Tensor],
    batch: torch.Tensor,
    weights: Dict[str, float] = None,
) -> Dict[str, torch.Tensor]:
    """
    Compute weighted total loss.
    """

    if weights is None:
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

    losses = {}

    losses["reconstruction"] = reconstruction_loss(outputs["reconstruction"], batch)
    losses["future_prediction"] = future_prediction_loss(outputs["future_prediction"], batch)
    losses["fast_sparsity"] = fast_sparsity_loss(outputs["fast_latent"])
    losses["slow_smoothness"] = slow_smoothness_loss(outputs["slow_latent"])
    losses["decorrelation"] = decorrelation_loss(outputs["fast_latent"], outputs["slow_latent"])
    losses["connectivity_sparsity"] = connectivity_sparsity_loss(outputs["connectivity"])
    losses["variance_preservation"] = variance_preservation_loss(outputs["fused_latent"])
    losses["orthogonality"] = orthogonality_loss(outputs["fast_latent"], outputs["slow_latent"])

    losses["graph_stability"] = torch.tensor(0.0, device=batch.device)

    total = torch.tensor(0.0, device=batch.device)

    for name, value in losses.items():
        total = total + weights.get(name, 0.0) * value

    losses["total"] = total

    return losses


if __name__ == "__main__":
    from src.models.dual_stream_model import DualStreamModel

    x = torch.randn(4, 12, 200)
    model = DualStreamModel()
    outputs = model(x)

    losses = compute_total_loss(outputs, x)

    for name, value in losses.items():
        print(name, value.item())