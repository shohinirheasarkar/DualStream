"""
dual_stream_model.py

Prototype dual-timescale functional connectivity model for the
DualStream / NeuroMamba project.

IMPORTANT:
This is NOT a full Mamba implementation yet.

Instead, this file builds a clean, interpretable prototype architecture
that mimics the idea of selective fast and slow temporal processing.

The current version uses:
    - Conv1D input projection
    - Shared GRU encoder
    - Fast GRU stream
    - Slow GRU stream
    - Fusion block
    - Reconstruction decoder
    - Future prediction decoder
    - Attention-style connectivity head

Future work:
    Replace GRUs with true selective state-space / Mamba modules.

------------------------------------------------------------
INPUT
------------------------------------------------------------

Observed ΔF/F traces

Shape:
    (batch, neurons, time)

------------------------------------------------------------
OUTPUTS
------------------------------------------------------------

1. Reconstruction
    reconstructed observed calcium traces

2. Future prediction
    predicted future neural activity

3. Connectivity matrix
    directed functional connectivity estimate

------------------------------------------------------------
SCIENTIFIC IDEA
------------------------------------------------------------

Fast stream:
    attempts to model fast predictive neuronal dynamics

Slow stream:
    attempts to model slower latent modulatory structure

Connectivity:
    inferred using attention-style query-key interactions
    between present target states and past source states
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. Input projection layer
# ============================================================

class InputProjection(nn.Module):
    """
    Initial temporal feature extraction layer.

    Uses Conv1D across time to extract local temporal structure.

    Input:
        (batch, neurons, time)

    Output:
        (batch, hidden_dim, time)
    """

    def __init__(
        self,
        num_neurons: int,
        hidden_dim: int,
    ):
        super().__init__()

        self.proj = nn.Conv1d(
            in_channels=num_neurons,
            out_channels=hidden_dim,
            kernel_size=3,
            padding=1,
        )

    def forward(self, x):

        print("\n[InputProjection]")
        print(f"Input shape: {x.shape}")

        x = self.proj(x)

        print(f"Projected shape: {x.shape}")

        return x


# ============================================================
# 2. Shared encoder
# ============================================================

class SharedEncoder(nn.Module):
    """
    Shared temporal encoder before stream separation.

    Uses a GRU to learn generic temporal features.

    Input:
        (batch, hidden_dim, time)

    Output:
        (batch, time, hidden_dim)
    """

    def __init__(
        self,
        hidden_dim: int,
        encoder_hidden: int = 64,
    ):
        super().__init__()

        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=encoder_hidden,
            batch_first=True,
        )

    def forward(self, x):

        print("\n[SharedEncoder]")
        print(f"Input shape: {x.shape}")

        # GRU expects:
        # (batch, time, features)

        x = x.permute(0, 2, 1)

        x, _ = self.gru(x)

        print(f"Encoded shape: {x.shape}")

        return x


# ============================================================
# 3. Fast stream
# ============================================================

class FastStream(nn.Module):
    """
    Fast temporal stream.

    Intended to capture:
        fast spike-like transient dynamics
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 32,
    ):
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )

    def forward(self, x):

        print("\n[FastStream]")

        out, _ = self.gru(x)

        print(f"Fast latent shape: {out.shape}")

        return out


# ============================================================
# 4. Slow stream
# ============================================================

class SlowStream(nn.Module):
    """
    Slow temporal stream.

    Intended to capture:
        slow modulatory latent structure
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
    ):
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )

    def forward(self, x):

        print("\n[SlowStream]")

        out, _ = self.gru(x)

        print(f"Slow latent shape: {out.shape}")

        return out


# ============================================================
# 5. Fusion block
# ============================================================

class FusionBlock(nn.Module):
    """
    Combine fast and slow latent streams.
    """

    def __init__(
        self,
        fast_dim: int,
        slow_dim: int,
        fused_dim: int = 64,
    ):
        super().__init__()

        self.linear = nn.Linear(
            fast_dim + slow_dim,
            fused_dim,
        )

    def forward(self, fast, slow):

        print("\n[FusionBlock]")

        fused = torch.cat([fast, slow], dim=-1)

        print(f"Concatenated shape: {fused.shape}")

        fused = self.linear(fused)

        print(f"Fused latent shape: {fused.shape}")

        return fused


# ============================================================
# 6. Reconstruction decoder
# ============================================================

class ReconstructionDecoder(nn.Module):
    """
    Reconstruct observed calcium traces.
    """

    def __init__(
        self,
        input_dim: int,
        num_neurons: int,
    ):
        super().__init__()

        self.linear = nn.Linear(
            input_dim,
            num_neurons,
        )

    def forward(self, x):

        print("\n[ReconstructionDecoder]")

        recon = self.linear(x)

        print(f"Reconstruction shape: {recon.shape}")

        return recon


# ============================================================
# 7. Future prediction decoder
# ============================================================

class FuturePredictionDecoder(nn.Module):
    """
    Predict future neural activity.

    For now:
        predicts one-step future activity.
    """

    def __init__(
        self,
        input_dim: int,
        num_neurons: int,
    ):
        super().__init__()

        self.linear = nn.Linear(
            input_dim,
            num_neurons,
        )

    def forward(self, x):

        print("\n[FuturePredictionDecoder]")

        pred = self.linear(x)

        print(f"Future prediction shape: {pred.shape}")

        return pred


# ============================================================
# 8. Connectivity head
# ============================================================

class ConnectivityHead(nn.Module):
    """
    Attention-style directed connectivity inference.

    Core idea:
        target neuron present
        attends to
        source neuron past

    This is a simplified prototype version.
    """

    def __init__(
        self,
        latent_dim: int,
        connectivity_dim: int = 32,
    ):
        super().__init__()

        self.query_proj = nn.Linear(
            latent_dim,
            connectivity_dim,
        )

        self.key_proj = nn.Linear(
            latent_dim,
            connectivity_dim,
        )

    def forward(self, x):

        """
        Input:
            (batch, time, latent_dim)

        Output:
            (batch, time, time)

        NOTE:
            For now this is temporal attention.
            Later we will upgrade this into
            neuron-wise directed connectivity.
        """

        print("\n[ConnectivityHead]")

        q = self.query_proj(x)
        k = self.key_proj(x)

        print(f"Query shape: {q.shape}")
        print(f"Key shape: {k.shape}")

        scores = torch.matmul(
            q,
            k.transpose(-2, -1),
        )

        scores = scores / np.sqrt(q.shape[-1])

        connectivity = F.softmax(scores, dim=-1)

        print(f"Connectivity shape: {connectivity.shape}")

        return connectivity


# ============================================================
# 9. Full DualStream model
# ============================================================

class DualStreamModel(nn.Module):
    """
    Full prototype architecture.
    """

    def __init__(
        self,
        num_neurons: int = 12,
        input_hidden_dim: int = 32,
        encoder_hidden_dim: int = 64,
        fast_hidden_dim: int = 32,
        slow_hidden_dim: int = 64,
        fused_dim: int = 64,
    ):
        super().__init__()

        # ----------------------------------------------------
        # Input projection
        # ----------------------------------------------------

        self.input_proj = InputProjection(
            num_neurons=num_neurons,
            hidden_dim=input_hidden_dim,
        )

        # ----------------------------------------------------
        # Shared encoder
        # ----------------------------------------------------

        self.shared_encoder = SharedEncoder(
            hidden_dim=input_hidden_dim,
            encoder_hidden=encoder_hidden_dim,
        )

        # ----------------------------------------------------
        # Fast + slow streams
        # ----------------------------------------------------

        self.fast_stream = FastStream(
            input_dim=encoder_hidden_dim,
            hidden_dim=fast_hidden_dim,
        )

        self.slow_stream = SlowStream(
            input_dim=encoder_hidden_dim,
            hidden_dim=slow_hidden_dim,
        )

        # ----------------------------------------------------
        # Fusion
        # ----------------------------------------------------

        self.fusion = FusionBlock(
            fast_dim=fast_hidden_dim,
            slow_dim=slow_hidden_dim,
            fused_dim=fused_dim,
        )

        # ----------------------------------------------------
        # Decoders
        # ----------------------------------------------------

        self.reconstruction_decoder = ReconstructionDecoder(
            input_dim=fused_dim,
            num_neurons=num_neurons,
        )

        self.future_decoder = FuturePredictionDecoder(
            input_dim=fused_dim,
            num_neurons=num_neurons,
        )

        # ----------------------------------------------------
        # Connectivity head
        # ----------------------------------------------------

        self.connectivity_head = ConnectivityHead(
            latent_dim=fused_dim,
        )

    def forward(self, x):

        """
        Input:
            x shape = (batch, neurons, time)

        Returns:
            reconstruction
            future_prediction
            connectivity
            fast_latent
            slow_latent
        """

        print("\n" + "=" * 70)
        print("Running DualStreamModel forward pass")
        print("=" * 70)

        # ----------------------------------------------------
        # Input projection
        # ----------------------------------------------------

        x_proj = self.input_proj(x)

        # ----------------------------------------------------
        # Shared encoding
        # ----------------------------------------------------

        shared = self.shared_encoder(x_proj)

        # ----------------------------------------------------
        # Fast + slow streams
        # ----------------------------------------------------

        fast = self.fast_stream(shared)
        slow = self.slow_stream(shared)

        # ----------------------------------------------------
        # Fusion
        # ----------------------------------------------------

        fused = self.fusion(fast, slow)

        # ----------------------------------------------------
        # Reconstruction
        # ----------------------------------------------------

        reconstruction = self.reconstruction_decoder(fused)

        # ----------------------------------------------------
        # Future prediction
        # ----------------------------------------------------

        future_prediction = self.future_decoder(fused)

        # ----------------------------------------------------
        # Connectivity
        # ----------------------------------------------------

        connectivity = self.connectivity_head(fused)

        return {
            "reconstruction": reconstruction,
            "future_prediction": future_prediction,
            "connectivity": connectivity,
            "fast_latent": fast,
            "slow_latent": slow,
            "fused_latent": fused,
        }


# ============================================================
# 10. Visualization helper
# ============================================================

def visualize_latents(
    fast_latent,
    slow_latent,
    save_path=None,
):
    """
    Plot example latent channels.
    """

    fast = fast_latent[0].detach().cpu().numpy()
    slow = slow_latent[0].detach().cpu().numpy()

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)

    for i in range(min(5, fast.shape[-1])):
        plt.plot(fast[:, i])

    plt.title("Fast latent channels")

    plt.subplot(1, 2, 2)

    for i in range(min(5, slow.shape[-1])):
        plt.plot(slow[:, i])

    plt.title("Slow latent channels")

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300)

    plt.show()


# ============================================================
# 11. Standalone demo
# ============================================================

if __name__ == "__main__":

    print("\nCreating fake calcium batch...")

    batch_size = 2
    num_neurons = 12
    timepoints = 200

    x = torch.randn(
        batch_size,
        num_neurons,
        timepoints,
    )

    print(f"Fake input shape: {x.shape}")

    model = DualStreamModel(
        num_neurons=num_neurons,
    )

    outputs = model(x)

    print("\n" + "=" * 70)
    print("FINAL OUTPUT SHAPES")
    print("=" * 70)

    for key, value in outputs.items():
        print(f"{key}: {value.shape}")

    visualize_latents(
        outputs["fast_latent"],
        outputs["slow_latent"],
    )

    print("\nDone.")