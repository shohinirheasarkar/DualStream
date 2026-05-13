"""
dual_stream_model.py

DualStream prototype model.

This is a working prototype of the project idea:

Observed ΔF/F calcium traces
    -> shared temporal encoder
    -> fast latent stream
    -> slow latent stream
    -> reconstruction decoder
    -> future prediction decoder
    -> directed connectivity matrix from FAST latent stream

Important:
This is not full Mamba yet. The fast and slow streams currently use GRUs as
lightweight stand-ins for future selective state-space/Mamba blocks.

Input shape:
    x = [batch, neurons, time]

Main output:
    C = [neurons, neurons]

Direction convention:
    C[target, source]
"""

import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class DualStreamModel(nn.Module):
    """
    End-to-end DualStream prototype.

    The important design choice:
        Each neuron is processed as its own time series, but the model weights
        are shared across neurons.

    This preserves neuron identity, which lets us compute a directed
    neuron-to-neuron matrix from the fast latent stream.
    """

    def __init__(
        self,
        hidden_dim: int = 32,
        fast_dim: int = 32,
        slow_dim: int = 32,
        fused_dim: int = 32,
        connectivity_dim: int = 16,
    ):
        super().__init__()

        # Each neuron trace is scalar at each time point, so input_dim = 1.
        self.input_projection = nn.Linear(1, hidden_dim)

        # Shared temporal encoder.
        self.shared_encoder = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )

        # Fast stream: meant to capture spike-like transient dynamics.
        self.fast_stream = nn.GRU(
            input_size=hidden_dim,
            hidden_size=fast_dim,
            batch_first=True,
        )

        # Slow stream: meant to capture slow modulatory dynamics.
        self.slow_stream = nn.GRU(
            input_size=hidden_dim,
            hidden_size=slow_dim,
            batch_first=True,
        )

        # Combine fast and slow streams.
        self.fusion = nn.Linear(fast_dim + slow_dim, fused_dim)

        # Decode fused latent back to calcium trace.
        self.reconstruction_decoder = nn.Linear(fused_dim, 1)

        # Predict next-time activity.
        self.future_decoder = nn.Linear(fused_dim, 1)

        # Query-key projections for connectivity from fast stream.
        self.query_projection = nn.Linear(fast_dim, connectivity_dim)
        self.key_projection = nn.Linear(fast_dim, connectivity_dim)

    def forward(
        self,
        x: torch.Tensor,
        min_lag: int = 1,
        max_lag: int = 5,
    ) -> Dict[str, torch.Tensor]:
        """
        x:
            [batch, neurons, time]

        returns dictionary with:
            reconstruction:    [batch, neurons, time]
            future_prediction: [batch, neurons, time]
            fast_latent:       [batch, neurons, time, fast_dim]
            slow_latent:       [batch, neurons, time, slow_dim]
            fused_latent:      [batch, neurons, time, fused_dim]
            connectivity:      [neurons, neurons]
        """

        batch_size, num_neurons, timepoints = x.shape

        # ------------------------------------------------------------
        # Treat each neuron as an independent sequence while sharing weights.
        # [B, N, T] -> [B*N, T, 1]
        # ------------------------------------------------------------
        x_flat = x.reshape(batch_size * num_neurons, timepoints, 1)

        # ------------------------------------------------------------
        # Input projection + shared encoder
        # ------------------------------------------------------------
        h = self.input_projection(x_flat)
        shared, _ = self.shared_encoder(h)

        # ------------------------------------------------------------
        # Fast and slow streams
        # ------------------------------------------------------------
        fast, _ = self.fast_stream(shared)
        slow, _ = self.slow_stream(shared)

        # ------------------------------------------------------------
        # Fusion
        # ------------------------------------------------------------
        fused = torch.cat([fast, slow], dim=-1)
        fused = torch.tanh(self.fusion(fused))

        # ------------------------------------------------------------
        # Decoders
        # ------------------------------------------------------------
        reconstruction = self.reconstruction_decoder(fused).squeeze(-1)
        future_prediction = self.future_decoder(fused).squeeze(-1)

        # Restore neuron identity.
        reconstruction = reconstruction.reshape(batch_size, num_neurons, timepoints)
        future_prediction = future_prediction.reshape(batch_size, num_neurons, timepoints)

        fast_latent = fast.reshape(batch_size, num_neurons, timepoints, -1)
        slow_latent = slow.reshape(batch_size, num_neurons, timepoints, -1)
        fused_latent = fused.reshape(batch_size, num_neurons, timepoints, -1)

        # ------------------------------------------------------------
        # Connectivity from fast latent stream
        # ------------------------------------------------------------
        connectivity = self.compute_connectivity_from_fast_latent(
            fast_latent=fast_latent,
            min_lag=min_lag,
            max_lag=max_lag,
        )

        return {
            "reconstruction": reconstruction,
            "future_prediction": future_prediction,
            "fast_latent": fast_latent,
            "slow_latent": slow_latent,
            "fused_latent": fused_latent,
            "connectivity": connectivity,
        }

    def compute_connectivity_from_fast_latent(
        self,
        fast_latent: torch.Tensor,
        min_lag: int = 1,
        max_lag: int = 5,
    ) -> torch.Tensor:
        """
        Compute directed neuron-to-neuron connectivity from the fast stream.

        fast_latent:
            [batch, neurons, time, fast_dim]

        For each target neuron i and source neuron j:
            compare target_i at time t
            to source_j at time t-lag

        Output:
            C[target, source]
        """

        batch_size, num_neurons, timepoints, _ = fast_latent.shape

        C = torch.zeros(
            num_neurons,
            num_neurons,
            device=fast_latent.device,
        )

        lag_count = 0

        for lag in range(min_lag, max_lag + 1):
            source_past = fast_latent[:, :, :-lag, :]
            target_now = fast_latent[:, :, lag:, :]

            # Query = target present.
            Q = self.query_projection(target_now)

            # Key = source past.
            K = self.key_projection(source_past)

            # scores[b, target, source, time]
            scores = torch.einsum("bitd,bjtd->bijt", Q, K)
            scores = scores / math.sqrt(Q.shape[-1])

            # Average across batch and time.
            C = C + scores.mean(dim=(0, 3))
            lag_count += 1

        C = C / lag_count

        # Remove self-connections before normalization.
        C = C.clone()
        C.fill_diagonal_(float("-inf"))

        # Row-wise softmax:
        # for each target neuron, distribute incoming source weights.
        C = F.softmax(C, dim=1)

        # Force diagonal back to zero after softmax.
        C = C.clone()
        C.fill_diagonal_(0.0)

        # Row-normalize so incoming weights sum to 1.
        C = C / (C.sum(dim=1, keepdim=True) + 1e-8)

        return C


if __name__ == "__main__":
    print("Testing DualStreamModel...")

    x = torch.randn(4, 12, 200)

    model = DualStreamModel()
    outputs = model(x)

    for key, value in outputs.items():
        print(key, value.shape)

    print("Model test complete.")