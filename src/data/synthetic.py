"""
synthetic.py

Synthetic calcium imaging data generator for the NeuroMamba / DualStream project.

Why this file exists
--------------------
For this project, we need a way to test whether our model can recover a known
directed connectivity matrix. Real calcium imaging datasets are useful, but they
usually do not give us the true directed neuron-to-neuron graph.

So for the first benchmark, we generate our own controlled synthetic dataset.

This file creates:

1. A sparse ground-truth directed graph
2. Spike trains influenced by that graph
3. Fast calcium traces from spike trains
4. Slow modulatory/glial-like signals
5. Bleaching drift
6. Noise
7. Observed ΔF/F-like calcium traces
8. Saved arrays + diagnostic plots

Main saved outputs
------------------
observed_dff.npy:
    Final mixed calcium signal that the model should receive.

true_graph.npy:
    Ground-truth directed connectivity matrix.

true_spikes.npy:
    Binary spike/event matrix.

fast_calcium.npy:
    Calcium signal generated from spikes only.

slow_modulation.npy:
    Slow shared / cell-specific modulation.

bleaching_drift.npy:
    Slow monotonic bleaching trend.

metadata.json:
    Basic parameters used to generate the dataset.

Shape convention
----------------
All trace-like arrays are shaped:

    (num_neurons, num_timepoints)

The graph is shaped:

    (num_neurons, num_neurons)

where:

    graph[target, source]

means source neuron j influences target neuron i.
"""

from pathlib import Path
from typing import Dict, Tuple
import json

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. Ground-truth directed graph
# ============================================================

def generate_ground_truth_graph(
    num_neurons: int = 12,
    edge_probability: float = 0.15,
    weight_low: float = 0.4,
    weight_high: float = 1.0,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate a sparse directed graph.

    The graph is represented as a matrix A with shape:

        (num_neurons, num_neurons)

    Direction convention:

        A[target, source]

    So if A[i, j] is large, then neuron j helps drive neuron i.

    Self-connections are removed because we want the final connectivity
    benchmark to focus on neuron-to-neuron relationships.
    """

    rng = np.random.default_rng(seed)

    print("\nGenerating ground-truth directed graph...")
    print(f"Number of neurons: {num_neurons}")
    print(f"Edge probability: {edge_probability}")

    # Random binary mask says which directed edges exist.
    edge_mask = rng.random((num_neurons, num_neurons)) < edge_probability

    # Remove self-connections.
    np.fill_diagonal(edge_mask, False)

    # Random positive edge weights.
    weights = rng.uniform(weight_low, weight_high, size=(num_neurons, num_neurons))

    graph = edge_mask * weights

    num_edges = np.count_nonzero(graph)
    possible_edges = num_neurons * (num_neurons - 1)

    print(f"Created {num_edges} directed edges out of {possible_edges} possible edges.")
    print(f"Graph density: {num_edges / possible_edges:.3f}")

    return graph.astype(np.float32)


# ============================================================
# 2. Spike train simulation from graph dynamics
# ============================================================

def simulate_spike_trains(
    graph: np.ndarray,
    num_timepoints: int = 2000,
    base_firing_rate: float = 0.02,
    coupling_strength: float = 0.08,
    refractory_period: int = 3,
    min_lag: int = 1,
    max_lag: int = 5,
    variable_rates: bool = True,
    seed: int = 42,
) -> np.ndarray:
    """
    Simulate simple binary spike/event trains using graph dynamics.

    This is intentionally simple and readable.

    Basic idea:
        Each neuron has some baseline chance of spiking.
        If source neuron j spiked recently, it increases the chance
        that target neuron i spikes after a short lag.

    Parameters
    ----------
    graph:
        Ground-truth graph shaped (target, source).

    num_timepoints:
        Number of time points to simulate.

    base_firing_rate:
        Baseline probability of spiking at each time step.

    coupling_strength:
        How strongly graph-based inputs increase spike probability.

    refractory_period:
        Number of time steps after a spike where a neuron cannot spike again.

    min_lag, max_lag:
        Source spikes can affect targets over this recent time window.

    variable_rates:
        If True, each neuron gets a slightly different baseline rate.

    Returns
    -------
    spikes:
        Binary array shaped (neurons, time).
    """

    rng = np.random.default_rng(seed)

    num_neurons = graph.shape[0]
    spikes = np.zeros((num_neurons, num_timepoints), dtype=np.float32)

    print("\nSimulating spike trains from graph dynamics...")
    print(f"Number of timepoints: {num_timepoints}")
    print(f"Base firing rate: {base_firing_rate}")
    print(f"Coupling strength: {coupling_strength}")
    print(f"Refractory period: {refractory_period}")

    if variable_rates:
        # Give each neuron its own baseline firing tendency.
        neuron_rates = rng.uniform(
            low=0.5 * base_firing_rate,
            high=1.5 * base_firing_rate,
            size=num_neurons,
        )
    else:
        neuron_rates = np.full(num_neurons, base_firing_rate)

    last_spike_time = np.full(num_neurons, -10_000)

    for t in range(max_lag, num_timepoints):
        for target in range(num_neurons):

            # Enforce a simple refractory period.
            if t - last_spike_time[target] <= refractory_period:
                continue

            # Start with baseline firing probability.
            spike_prob = neuron_rates[target]

            # Look at recent source activity from t-max_lag to t-min_lag.
            recent_source_activity = spikes[:, t - max_lag : t - min_lag + 1].sum(axis=1)

            # Weighted graph input into this target neuron.
            graph_input = np.dot(graph[target], recent_source_activity)

            # Convert graph input into a probability boost.
            spike_prob += coupling_strength * graph_input

            # Keep probability in a safe range.
            spike_prob = np.clip(spike_prob, 0.0, 0.8)

            if rng.random() < spike_prob:
                spikes[target, t] = 1.0
                last_spike_time[target] = t

    print(f"Total spikes generated: {int(spikes.sum())}")
    print(f"Average firing probability per timepoint: {spikes.mean():.4f}")

    return spikes.astype(np.float32)


# ============================================================
# 3. Calcium indicator convolution
# ============================================================

def make_calcium_kernel(
    kernel_length: int = 80,
    tau_decay: float = 12.0,
) -> np.ndarray:
    """
    Create a simple exponential calcium decay kernel.

    A spike creates a calcium transient that rises immediately and then
    decays gradually over time.

    This is a simplified version of calcium indicator dynamics.
    """

    t = np.arange(kernel_length)
    kernel = np.exp(-t / tau_decay)

    # Normalize so the peak is 1.
    kernel = kernel / kernel.max()

    return kernel.astype(np.float32)


def spikes_to_calcium(
    spikes: np.ndarray,
    kernel_length: int = 80,
    tau_decay: float = 12.0,
) -> np.ndarray:
    """
    Convert binary spike trains into calcium-like traces by convolution.

    Input:
        spikes shape = (neurons, time)

    Output:
        fast_calcium shape = (neurons, time)
    """

    print("\nConverting spikes to fast calcium traces...")
    print(f"Kernel length: {kernel_length}")
    print(f"Tau decay: {tau_decay}")

    num_neurons, num_timepoints = spikes.shape
    kernel = make_calcium_kernel(kernel_length=kernel_length, tau_decay=tau_decay)

    fast_calcium = np.zeros_like(spikes, dtype=np.float32)

    for neuron in range(num_neurons):
        convolved = np.convolve(spikes[neuron], kernel, mode="full")
        fast_calcium[neuron] = convolved[:num_timepoints]

    print(f"Fast calcium shape: {fast_calcium.shape}")

    return fast_calcium.astype(np.float32)


# ============================================================
# 4. Slow sinusoidal / glial-like modulation
# ============================================================

def generate_slow_modulation(
    num_neurons: int,
    num_timepoints: int,
    amplitude: float = 0.4,
    num_global_components: int = 2,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate slow modulatory structure.

    This is meant to approximate slow calcium fluctuations that could come from:
        - astrocytic / glial-timescale modulation
        - neuromodulatory state changes
        - global brain state shifts
        - slow biological or imaging-related trends

    Important:
        We should NOT claim this is literally astrocyte activity.
        It is a slow latent modulation term.
    """

    rng = np.random.default_rng(seed)

    print("\nGenerating slow modulatory signal...")
    print(f"Amplitude: {amplitude}")
    print(f"Global components: {num_global_components}")

    time = np.linspace(0, 1, num_timepoints)

    global_components = []

    for _ in range(num_global_components):
        freq = rng.uniform(1.0, 4.0)
        phase = rng.uniform(0, 2 * np.pi)

        component = np.sin(2 * np.pi * freq * time + phase)
        global_components.append(component)

    global_components = np.array(global_components)

    # Each neuron gets a random mixture of global slow components.
    mixing = rng.normal(0, 1, size=(num_neurons, num_global_components))
    slow = mixing @ global_components

    # Add small neuron-specific slow drift.
    for neuron in range(num_neurons):
        freq = rng.uniform(0.5, 2.0)
        phase = rng.uniform(0, 2 * np.pi)
        slow[neuron] += 0.5 * np.sin(2 * np.pi * freq * time + phase)

    # Normalize and scale.
    slow = slow / (np.std(slow) + 1e-8)
    slow = amplitude * slow

    print(f"Slow modulation shape: {slow.shape}")

    return slow.astype(np.float32)


# ============================================================
# 5. Bleaching drift
# ============================================================

def generate_bleaching_drift(
    num_neurons: int,
    num_timepoints: int,
    strength: float = 0.3,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate a slow monotonic photobleaching-like drift.

    Bleaching often causes fluorescence to gradually decrease over time.
    Here we model that as a negative exponential trend with slightly
    different strengths across neurons.
    """

    rng = np.random.default_rng(seed)

    print("\nGenerating bleaching drift...")
    print(f"Bleaching strength: {strength}")

    time = np.linspace(0, 1, num_timepoints)
    base_decay = -strength * (1 - np.exp(-3 * time))

    neuron_scales = rng.uniform(0.7, 1.3, size=(num_neurons, 1))
    drift = neuron_scales * base_decay[None, :]

    print(f"Bleaching drift shape: {drift.shape}")

    return drift.astype(np.float32)


# ============================================================
# 6. Noise
# ============================================================

def add_noise(
    signal: np.ndarray,
    noise_std: float = 0.15,
    seed: int = 42,
) -> np.ndarray:
    """
    Add Gaussian noise to calcium traces.
    """

    rng = np.random.default_rng(seed)

    print("\nAdding noise...")
    print(f"Noise std: {noise_std}")

    noise = rng.normal(0, noise_std, size=signal.shape)
    return (signal + noise).astype(np.float32)


# ============================================================
# 7. Full synthetic recording generator
# ============================================================

def generate_synthetic_recording(
    num_neurons: int = 12,
    num_timepoints: int = 2000,
    edge_probability: float = 0.15,
    base_firing_rate: float = 0.02,
    coupling_strength: float = 0.08,
    slow_amplitude: float = 0.4,
    bleaching_strength: float = 0.25,
    noise_std: float = 0.15,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """
    Generate a complete synthetic calcium recording.

    Returns a dictionary containing:
        true_graph
        true_spikes
        fast_calcium
        slow_modulation
        bleaching_drift
        observed_dff
    """

    print("\n" + "=" * 70)
    print("Generating full synthetic calcium recording")
    print("=" * 70)

    graph = generate_ground_truth_graph(
        num_neurons=num_neurons,
        edge_probability=edge_probability,
        seed=seed,
    )

    spikes = simulate_spike_trains(
        graph=graph,
        num_timepoints=num_timepoints,
        base_firing_rate=base_firing_rate,
        coupling_strength=coupling_strength,
        seed=seed + 1,
    )

    fast_calcium = spikes_to_calcium(spikes)

    slow_modulation = generate_slow_modulation(
        num_neurons=num_neurons,
        num_timepoints=num_timepoints,
        amplitude=slow_amplitude,
        seed=seed + 2,
    )

    bleaching_drift = generate_bleaching_drift(
        num_neurons=num_neurons,
        num_timepoints=num_timepoints,
        strength=bleaching_strength,
        seed=seed + 3,
    )

    clean_signal = fast_calcium + slow_modulation + bleaching_drift
    observed_dff = add_noise(clean_signal, noise_std=noise_std, seed=seed + 4)

    print("\nSynthetic recording complete.")
    print(f"Observed ΔF/F shape: {observed_dff.shape}")
    print(f"Ground-truth graph shape: {graph.shape}")

    return {
        "true_graph": graph,
        "true_spikes": spikes,
        "fast_calcium": fast_calcium,
        "slow_modulation": slow_modulation,
        "bleaching_drift": bleaching_drift,
        "observed_dff": observed_dff,
    }


# ============================================================
# 8. Saving
# ============================================================

def save_synthetic_dataset(
    data: Dict[str, np.ndarray],
    output_dir: str = "data/synthetic/example_001",
    metadata: Dict = None,
) -> None:
    """
    Save synthetic dataset arrays and metadata.

    Each component is saved separately so that later we can compare:
        model input      -> observed_dff
        model target     -> true_graph
        latent fast goal -> fast_calcium / true_spikes
        latent slow goal -> slow_modulation
    """

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("\nSaving synthetic dataset...")
    print(f"Output directory: {output_path}")

    for name, array in data.items():
        save_path = output_path / f"{name}.npy"
        np.save(save_path, array)
        print(f"Saved {name}: {save_path}")

    if metadata is not None:
        metadata_path = output_path / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)

        print(f"Saved metadata: {metadata_path}")


# ============================================================
# 9. Diagnostic plotting
# ============================================================

def plot_synthetic_summary(
    data: Dict[str, np.ndarray],
    num_neurons_to_plot: int = 5,
    save_dir: str = "outputs/figures",
    show: bool = True,
) -> None:
    """
    Create diagnostic plots for the synthetic dataset.

    Plots:
        1. Ground-truth graph heatmap
        2. Example spike trains
        3. Fast calcium traces
        4. Slow modulation
        5. Observed mixed ΔF/F traces
    """

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    graph = data["true_graph"]
    spikes = data["true_spikes"]
    fast = data["fast_calcium"]
    slow = data["slow_modulation"]
    observed = data["observed_dff"]

    n_plot = min(num_neurons_to_plot, observed.shape[0])

    print("\nCreating synthetic data diagnostic plots...")

    plt.figure(figsize=(6, 5))
    plt.imshow(graph, aspect="auto")
    plt.colorbar(label="Edge weight")
    plt.title("Ground-truth directed graph\nrows = targets, columns = sources")
    plt.xlabel("Source neuron")
    plt.ylabel("Target neuron")
    plt.tight_layout()
    plt.savefig(save_path / "true_graph_heatmap.png", dpi=200)
    if show:
        plt.show()
    else:
        plt.close()

    plt.figure(figsize=(12, 4))
    for i in range(n_plot):
        spike_times = np.where(spikes[i] > 0)[0]
        plt.vlines(spike_times, i, i + 0.8)
    plt.title("Example true spike trains")
    plt.xlabel("Time")
    plt.ylabel("Neuron")
    plt.tight_layout()
    plt.savefig(save_path / "true_spikes.png", dpi=200)
    if show:
        plt.show()
    else:
        plt.close()

    plt.figure(figsize=(12, 4))
    for i in range(n_plot):
        plt.plot(fast[i] + i * 3, label=f"Neuron {i}")
    plt.title("Fast calcium component from spikes")
    plt.xlabel("Time")
    plt.ylabel("Fast calcium + offset")
    plt.tight_layout()
    plt.savefig(save_path / "fast_calcium.png", dpi=200)
    if show:
        plt.show()
    else:
        plt.close()

    plt.figure(figsize=(12, 4))
    for i in range(n_plot):
        plt.plot(slow[i] + i * 3, label=f"Neuron {i}")
    plt.title("Slow modulatory component")
    plt.xlabel("Time")
    plt.ylabel("Slow signal + offset")
    plt.tight_layout()
    plt.savefig(save_path / "slow_modulation.png", dpi=200)
    if show:
        plt.show()
    else:
        plt.close()

    plt.figure(figsize=(12, 4))
    for i in range(n_plot):
        plt.plot(observed[i] + i * 4, label=f"Neuron {i}")
    plt.title("Observed synthetic ΔF/F = fast + slow + bleaching + noise")
    plt.xlabel("Time")
    plt.ylabel("Observed ΔF/F + offset")
    plt.tight_layout()
    plt.savefig(save_path / "observed_dff.png", dpi=200)
    if show:
        plt.show()
    else:
        plt.close()

    print(f"Saved figures to: {save_path}")


# ============================================================
# 10. Main sanity check
# ============================================================

if __name__ == "__main__":
    """
    Run this file directly with:

        python src/data/synthetic.py

    It will generate one synthetic dataset, save it, and make plots.
    """

    metadata = {
        "dataset_name": "synthetic_example_001",
        "num_neurons": 12,
        "num_timepoints": 2000,
        "edge_probability": 0.15,
        "base_firing_rate": 0.02,
        "coupling_strength": 0.08,
        "slow_amplitude": 0.4,
        "bleaching_strength": 0.25,
        "noise_std": 0.15,
        "seed": 42,
        "shape_convention": "(neurons, time)",
        "graph_convention": "graph[target, source]",
    }

    synthetic_data = generate_synthetic_recording(
        num_neurons=metadata["num_neurons"],
        num_timepoints=metadata["num_timepoints"],
        edge_probability=metadata["edge_probability"],
        base_firing_rate=metadata["base_firing_rate"],
        coupling_strength=metadata["coupling_strength"],
        slow_amplitude=metadata["slow_amplitude"],
        bleaching_strength=metadata["bleaching_strength"],
        noise_std=metadata["noise_std"],
        seed=metadata["seed"],
    )

    save_synthetic_dataset(
        data=synthetic_data,
        output_dir="data/synthetic/example_001",
        metadata=metadata,
    )

    plot_synthetic_summary(
        data=synthetic_data,
        save_dir="outputs/figures/synthetic_example_001",
        show=True,
    )

    print("\nDone generating synthetic benchmark.")