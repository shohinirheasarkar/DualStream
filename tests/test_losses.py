"""
Tests for src/losses/losses.py

These tests check that each loss function behaves in the direction we expect.
The goal is not just "does it run?" but "does it reward the behavior we want?"
"""

import torch

from src.losses.losses import (
    reconstruction_loss,
    future_prediction_loss,
    fast_sparsity_loss,
    slow_smoothness_loss,
    decorrelation_loss,
    connectivity_sparsity_loss,
    variance_preservation_loss,
    orthogonality_loss,
    graph_stability_loss,
    compute_total_loss,
)


def test_reconstruction_loss_is_zero_for_perfect_reconstruction():
    batch = torch.randn(2, 4, 10)  # batch, neurons, time
    perfect_recon = batch.permute(0, 2, 1)  # batch, time, neurons

    loss = reconstruction_loss(perfect_recon, batch)

    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)


def test_reconstruction_loss_increases_when_reconstruction_is_bad():
    batch = torch.zeros(2, 4, 10)
    bad_recon = torch.ones(2, 10, 4)

    loss = reconstruction_loss(bad_recon, batch)

    assert loss.item() > 0.5


def test_future_prediction_loss_is_zero_for_perfect_next_step_prediction():
    batch = torch.randn(2, 4, 10)
    target = batch.permute(0, 2, 1)

    prediction = torch.zeros_like(target)
    prediction[:, :-1, :] = target[:, 1:, :]

    loss = future_prediction_loss(prediction, batch)

    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)


def test_fast_sparsity_loss_lower_for_sparse_latent():
    sparse = torch.zeros(2, 10, 5)
    dense = torch.ones(2, 10, 5)

    sparse_loss = fast_sparsity_loss(sparse)
    dense_loss = fast_sparsity_loss(dense)

    assert sparse_loss.item() < dense_loss.item()


def test_slow_smoothness_loss_lower_for_smooth_signal():
    smooth = torch.ones(2, 10, 5)

    noisy = torch.randn(2, 10, 5)

    smooth_loss = slow_smoothness_loss(smooth)
    noisy_loss = slow_smoothness_loss(noisy)

    assert smooth_loss.item() < noisy_loss.item()


def test_decorrelation_loss_higher_when_streams_are_identical():
    fast = torch.randn(2, 20, 8)
    identical_slow = fast.clone()
    random_slow = torch.randn(2, 20, 8)

    identical_loss = decorrelation_loss(fast, identical_slow)
    random_loss = decorrelation_loss(fast, random_slow)

    assert identical_loss.item() > random_loss.item()


def test_connectivity_sparsity_loss_lower_for_sparse_matrix():
    sparse = torch.zeros(2, 6, 6)
    dense = torch.ones(2, 6, 6)

    sparse_loss = connectivity_sparsity_loss(sparse)
    dense_loss = connectivity_sparsity_loss(dense)

    assert sparse_loss.item() < dense_loss.item()


def test_variance_preservation_penalizes_collapsed_latent():
    collapsed = torch.ones(2, 10, 5)
    varied = torch.randn(2, 10, 5)

    collapsed_loss = variance_preservation_loss(collapsed, min_variance=0.1)
    varied_loss = variance_preservation_loss(varied, min_variance=0.1)

    assert collapsed_loss.item() > varied_loss.item()


def test_orthogonality_loss_lower_for_orthogonal_streams():
    fast = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])
    slow_orthogonal = torch.tensor([[[0.0, 1.0], [0.0, 1.0]]])
    slow_same = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])

    orthogonal_value = orthogonality_loss(fast, slow_orthogonal)
    same_value = orthogonality_loss(fast, slow_same)

    assert orthogonal_value.item() < same_value.item()


def test_graph_stability_loss_zero_for_identical_graphs():
    graphs = torch.ones(3, 4, 4)

    loss = graph_stability_loss(graphs)

    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)


def test_graph_stability_loss_positive_for_changing_graphs():
    graphs = torch.randn(3, 4, 4)

    loss = graph_stability_loss(graphs)

    assert loss.item() > 0.0


def test_compute_total_loss_returns_all_expected_keys():
    batch = torch.randn(2, 4, 10)

    outputs = {
        "reconstruction": torch.randn(2, 10, 4),
        "future_prediction": torch.randn(2, 10, 4),
        "fast_latent": torch.randn(2, 10, 8),
        "slow_latent": torch.randn(2, 10, 12),
        "connectivity": torch.randn(2, 10, 10),
        "fused_latent": torch.randn(2, 10, 16),
    }

    losses = compute_total_loss(outputs, batch)

    expected_keys = {
        "reconstruction",
        "future_prediction",
        "fast_sparsity",
        "slow_smoothness",
        "decorrelation",
        "connectivity_sparsity",
        "variance_preservation",
        "orthogonality",
        "graph_stability",
        "total",
    }

    assert set(losses.keys()) == expected_keys

    for value in losses.values():
        assert torch.isfinite(value)


def test_total_loss_changes_when_weights_change():
    batch = torch.randn(2, 4, 10)

    outputs = {
        "reconstruction": torch.randn(2, 10, 4),
        "future_prediction": torch.randn(2, 10, 4),
        "fast_latent": torch.randn(2, 10, 8),
        "slow_latent": torch.randn(2, 10, 12),
        "connectivity": torch.randn(2, 10, 10),
        "fused_latent": torch.randn(2, 10, 16),
    }

    weights_a = {
        "reconstruction": 1.0,
        "future_prediction": 0.0,
        "fast_sparsity": 0.0,
        "slow_smoothness": 0.0,
        "decorrelation": 0.0,
        "connectivity_sparsity": 0.0,
        "variance_preservation": 0.0,
        "orthogonality": 0.0,
        "graph_stability": 0.0,
    }

    weights_b = {
        "reconstruction": 0.0,
        "future_prediction": 1.0,
        "fast_sparsity": 0.0,
        "slow_smoothness": 0.0,
        "decorrelation": 0.0,
        "connectivity_sparsity": 0.0,
        "variance_preservation": 0.0,
        "orthogonality": 0.0,
        "graph_stability": 0.0,
    }

    loss_a = compute_total_loss(outputs, batch, weights=weights_a)["total"]
    loss_b = compute_total_loss(outputs, batch, weights=weights_b)["total"]

    assert not torch.isclose(loss_a, loss_b)