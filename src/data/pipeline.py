"""
pipeline.py

User-friendly data pipeline for NeuroMamba / DualStream.

This file handles the first major step of the project:

1. Load calcium traces from common formats
2. Standardize them into a consistent shape
3. Handle missing values
4. Normalize each neuron/ROI trace
5. Segment long recordings into windows
6. Store metadata
7. Split data by session or animal to avoid leakage

Expected input:
    ΔF/F calcium trace matrix shaped as:

        (num_neurons, num_timepoints)

    Each row = one neuron / ROI
    Each column = one time point

Final model-ready output:
    windows shaped as:

        (num_windows, num_neurons, window_size)

Later, the training code can batch these into:

        (batch, neurons, time)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. Metadata container
# ============================================================

@dataclass
class RecordingMetadata:
    """
    Stores information about one recording/session.

    We keep this separate from the actual trace matrix so that later
    we can split train/val/test by animal or session instead of randomly
    mixing windows from the same recording.

    Parameters
    ----------
    session_id:
        Unique name for this recording session.

    animal_id:
        Animal/mouse identifier. Useful for leakage-safe splitting.

    frame_rate:
        Imaging frame rate in Hz.

    condition:
        Experimental condition, e.g. "baseline", "drug", "stim", etc.

    cell_ids:
        Optional list of neuron/ROI identifiers.
    """

    session_id: str
    animal_id: str
    frame_rate: float
    condition: str = "unknown"
    cell_ids: Optional[List[str]] = None


# ============================================================
# 2. Loading functions
# ============================================================

def load_npy_trace(file_path: str) -> np.ndarray:
    """
    Load ΔF/F traces from a .npy file.

    Expected shape:
        (neurons, time)

    Example:
        X = np.load("my_traces.npy")
    """

    print(f"\nLoading .npy file from: {file_path}")
    X = np.load(file_path)
    print(f"Loaded array with shape: {X.shape}")
    return X


def load_csv_trace(file_path: str) -> np.ndarray:
    """
    Load ΔF/F traces from a .csv file.

    Assumption:
        Rows are time points and columns are neurons OR
        rows are neurons and columns are time points.

    Because labs store these differently, this function loads the CSV
    and then lets standardize_trace_shape() fix the orientation.

    Important:
        This assumes the CSV is mostly numeric.
    """

    print(f"\nLoading .csv file from: {file_path}")
    df = pd.read_csv(file_path)

    print(f"CSV shape before conversion: {df.shape}")
    print("First few columns:", list(df.columns[:5]))

    # Keep only numeric columns, because some CSVs may include time labels.
    numeric_df = df.select_dtypes(include=[np.number])

    if numeric_df.shape[1] != df.shape[1]:
        print("Some non-numeric columns were ignored.")

    X = numeric_df.to_numpy()
    print(f"Numeric array shape after conversion: {X.shape}")
    return X


def load_hdf5_trace(file_path: str, dataset_key: Optional[str] = None) -> np.ndarray:
    """
    Load traces from an .hdf5 / .h5 file.

    This is useful for CaImAn-style outputs, but HDF5 files can store data
    under many possible keys.

    If dataset_key is provided, we load that key directly.

    If dataset_key is not provided, this function prints available keys
    and tries to guess a reasonable trace dataset.

    Common CaImAn-ish candidates might include:
        "estimates/F_dff"
        "estimates/C"
        "F_dff"
        "C"
    """

    import h5py

    print(f"\nLoading .hdf5/.h5 file from: {file_path}")

    with h5py.File(file_path, "r") as f:
        print("Available top-level keys:")
        for key in f.keys():
            print(f"  - {key}")

        if dataset_key is not None:
            print(f"Loading dataset key: {dataset_key}")
            X = np.array(f[dataset_key])
            print(f"Loaded array with shape: {X.shape}")
            return X

        candidate_keys = [
            "estimates/F_dff",
            "estimates/C",
            "F_dff",
            "C",
            "traces",
            "dff",
            "df_f",
        ]

        for key in candidate_keys:
            if key in f:
                print(f"Automatically found candidate dataset: {key}")
                X = np.array(f[key])
                print(f"Loaded array with shape: {X.shape}")
                return X

    raise ValueError(
        "Could not automatically find trace data in HDF5 file. "
        "Please pass dataset_key='your/key/here'."
    )


def load_trace_file(file_path: str, dataset_key: Optional[str] = None) -> np.ndarray:
    """
    General loader for supported trace formats.

    Supported now:
        .npy
        .csv
        .h5 / .hdf5

    Placeholder for later:
        .nwb
    """

    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".npy":
        return load_npy_trace(file_path)

    if suffix == ".csv":
        return load_csv_trace(file_path)

    if suffix in [".h5", ".hdf5"]:
        return load_hdf5_trace(file_path, dataset_key=dataset_key)

    if suffix == ".nwb":
        raise NotImplementedError(
            "NWB loading is not implemented yet. "
            "We can add pynwb support later once we know your NWB structure."
        )

    raise ValueError(f"Unsupported file type: {suffix}")


# ============================================================
# 3. Shape checking and standardization
# ============================================================

def standardize_trace_shape(
    X: np.ndarray,
    expected_orientation: str = "neurons_by_time",
) -> np.ndarray:
    """
    Convert trace matrix to shape:

        (neurons, time)

    Parameters
    ----------
    X:
        Input array.

    expected_orientation:
        "neurons_by_time":
            Assumes X is already shaped (neurons, time).

        "time_by_neurons":
            Assumes X is shaped (time, neurons), so we transpose.

        "auto":
            Tries to guess orientation.
            Usually, timepoints > neurons, so if rows > columns,
            we assume rows are time and transpose.

    Returns
    -------
    X_standard:
        Array shaped (neurons, time).
    """

    X = np.asarray(X)

    if X.ndim != 2:
        raise ValueError(
            f"Expected a 2D trace matrix, but got shape {X.shape}. "
            "Expected either (neurons, time) or (time, neurons)."
        )

    print("\nStandardizing trace shape...")
    print(f"Original shape: {X.shape}")

    if expected_orientation == "neurons_by_time":
        X_standard = X

    elif expected_orientation == "time_by_neurons":
        X_standard = X.T

    elif expected_orientation == "auto":
        rows, cols = X.shape

        if rows > cols:
            print(
                "Auto-detected likely shape (time, neurons), "
                "so transposing to (neurons, time)."
            )
            X_standard = X.T
        else:
            print("Auto-detected likely shape (neurons, time).")
            X_standard = X

    else:
        raise ValueError(
            "expected_orientation must be one of: "
            "'neurons_by_time', 'time_by_neurons', or 'auto'."
        )

    print(f"Standardized shape: {X_standard.shape} = (neurons, time)")
    return X_standard.astype(np.float32)


# ============================================================
# 4. Missing value handling
# ============================================================

def handle_missing_values(X: np.ndarray) -> np.ndarray:
    """
    Handle missing values in calcium traces.

    Strategy:
        1. For each neuron, linearly interpolate NaNs over time.
        2. If NaNs remain at the edges, fill them with the neuron's mean.
        3. If an entire neuron is NaN, replace it with zeros.
    """

    print("\nChecking for missing values...")

    X_clean = X.copy()
    total_nans = np.isnan(X_clean).sum()

    print(f"Total NaNs found: {total_nans}")

    if total_nans == 0:
        print("No missing values detected.")
        return X_clean

    num_neurons, num_timepoints = X_clean.shape
    time = np.arange(num_timepoints)

    for neuron_idx in range(num_neurons):
        trace = X_clean[neuron_idx]
        nan_mask = np.isnan(trace)

        if nan_mask.all():
            print(f"Neuron {neuron_idx} is entirely NaN. Replacing with zeros.")
            X_clean[neuron_idx] = np.zeros(num_timepoints)
            continue

        if nan_mask.any():
            valid_time = time[~nan_mask]
            valid_values = trace[~nan_mask]

            interpolated = np.interp(time, valid_time, valid_values)
            X_clean[neuron_idx] = interpolated

    remaining_nans = np.isnan(X_clean).sum()
    print(f"Remaining NaNs after cleaning: {remaining_nans}")

    return X_clean


# ============================================================
# 5. Normalization
# ============================================================

def normalize_per_neuron(X: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Z-score normalize each neuron independently.

    For each neuron i:

        X_norm[i] = (X[i] - mean_i) / std_i

    Why:
        Calcium amplitudes can vary across cells due to brightness,
        segmentation quality, expression level, or imaging artifacts.
        Per-neuron normalization prevents high-amplitude cells from
        dominating the model.
    """

    print("\nNormalizing each neuron trace...")

    means = X.mean(axis=1, keepdims=True)
    stds = X.std(axis=1, keepdims=True)

    near_constant = (stds.squeeze() < eps).sum()
    if near_constant > 0:
        print(f"Warning: {near_constant} near-constant traces found.")

    X_norm = (X - means) / (stds + eps)

    print("Normalization complete.")
    print(f"Mean after normalization: {X_norm.mean():.4f}")
    print(f"Std after normalization: {X_norm.std():.4f}")

    return X_norm.astype(np.float32)


# ============================================================
# 6. Windowing
# ============================================================

def segment_into_windows(
    X: np.ndarray,
    window_size: int = 500,
    stride: int = 250,
) -> np.ndarray:
    """
    Segment a long recording into overlapping windows.

    Input:
        X shape = (neurons, time)

    Output:
        windows shape = (num_windows, neurons, window_size)

    Example:
        X shape: (80, 12000)
        window_size: 500
        stride: 250

        Output:
            many windows, each shaped (80, 500)
    """

    print("\nSegmenting recording into windows...")
    print(f"Input shape: {X.shape}")
    print(f"Window size: {window_size}")
    print(f"Stride: {stride}")

    num_neurons, num_timepoints = X.shape

    if num_timepoints < window_size:
        raise ValueError(
            f"Recording has only {num_timepoints} timepoints, "
            f"but window_size={window_size}."
        )

    windows = []

    for start in range(0, num_timepoints - window_size + 1, stride):
        end = start + window_size
        window = X[:, start:end]
        windows.append(window)

    windows = np.stack(windows, axis=0)

    print(f"Created windows with shape: {windows.shape}")
    print("Shape means: (num_windows, neurons, time)")

    return windows.astype(np.float32)


# ============================================================
# 7. Metadata + record object
# ============================================================

@dataclass
class ProcessedRecording:
    """
    Stores one processed recording.

    traces:
        Full normalized trace matrix, shape (neurons, time)

    windows:
        Windowed version, shape (num_windows, neurons, window_size)

    metadata:
        Recording metadata
    """

    traces: np.ndarray
    windows: np.ndarray
    metadata: RecordingMetadata


# ============================================================
# 8. Full single-recording pipeline
# ============================================================

def process_single_recording(
    file_path: str,
    metadata: RecordingMetadata,
    dataset_key: Optional[str] = None,
    expected_orientation: str = "auto",
    window_size: int = 500,
    stride: int = 250,
    make_plots: bool = True,
) -> ProcessedRecording:
    """
    Complete pipeline for one recording.

    This is the main function to call when you have one ΔF/F file.

    Steps:
        1. Load data
        2. Standardize shape to (neurons, time)
        3. Handle missing values
        4. Normalize each neuron
        5. Segment into windows
        6. Optionally plot diagnostics
    """

    print("\n" + "=" * 70)
    print(f"Processing recording: {metadata.session_id}")
    print("=" * 70)

    X = load_trace_file(file_path, dataset_key=dataset_key)
    X = standardize_trace_shape(X, expected_orientation=expected_orientation)
    X = handle_missing_values(X)
    X = normalize_per_neuron(X)
    windows = segment_into_windows(X, window_size=window_size, stride=stride)

    if make_plots:
        plot_recording_summary(X, windows, metadata)

    processed = ProcessedRecording(
        traces=X,
        windows=windows,
        metadata=metadata,
    )

    print("\nFinished processing recording.")
    print(f"Session: {metadata.session_id}")
    print(f"Animal: {metadata.animal_id}")
    print(f"Condition: {metadata.condition}")
    print(f"Frame rate: {metadata.frame_rate} Hz")
    print(f"Final trace shape: {processed.traces.shape}")
    print(f"Final windows shape: {processed.windows.shape}")

    return processed


# ============================================================
# 9. Train/val/test split by animal or session
# ============================================================

def split_recordings(
    recordings: List[ProcessedRecording],
    split_by: str = "animal_id",
    train_fraction: float = 0.7,
    val_fraction: float = 0.15,
    seed: int = 42,
) -> Dict[str, List[ProcessedRecording]]:
    """
    Split recordings into train/val/test sets without leakage.

    Important:
        We do NOT randomly split individual windows.
        That would leak nearly identical chunks of the same recording
        into both train and test.

    Instead, we split by:
        - animal_id, or
        - session_id

    Parameters
    ----------
    recordings:
        List of processed recordings.

    split_by:
        Either "animal_id" or "session_id".

    train_fraction:
        Fraction of unique groups used for training.

    val_fraction:
        Fraction of unique groups used for validation.

    Remaining groups go to test.
    """

    print("\nCreating leakage-safe train/val/test split...")
    print(f"Splitting by: {split_by}")

    if split_by not in ["animal_id", "session_id"]:
        raise ValueError("split_by must be either 'animal_id' or 'session_id'.")

    rng = np.random.default_rng(seed)

    if split_by == "animal_id":
        groups = [rec.metadata.animal_id for rec in recordings]
    else:
        groups = [rec.metadata.session_id for rec in recordings]

    unique_groups = np.array(sorted(set(groups)))
    rng.shuffle(unique_groups)

    n_groups = len(unique_groups)
    n_train = int(train_fraction * n_groups)
    n_val = int(val_fraction * n_groups)

    train_groups = set(unique_groups[:n_train])
    val_groups = set(unique_groups[n_train:n_train + n_val])
    test_groups = set(unique_groups[n_train + n_val:])

    split = {"train": [], "val": [], "test": []}

    for rec in recordings:
        group = rec.metadata.animal_id if split_by == "animal_id" else rec.metadata.session_id

        if group in train_groups:
            split["train"].append(rec)
        elif group in val_groups:
            split["val"].append(rec)
        else:
            split["test"].append(rec)

    print(f"Train groups: {train_groups}")
    print(f"Val groups: {val_groups}")
    print(f"Test groups: {test_groups}")

    print(f"Train recordings: {len(split['train'])}")
    print(f"Val recordings: {len(split['val'])}")
    print(f"Test recordings: {len(split['test'])}")

    return split


def combine_windows(recordings: List[ProcessedRecording]) -> np.ndarray:
    """
    Combine windows from multiple recordings into one array.

    Input:
        list of ProcessedRecording objects

    Output:
        array shaped (total_windows, neurons, time)

    Note:
        This assumes all recordings have the same number of neurons.
        For real multi-session data, this may not always be true.
    """

    if len(recordings) == 0:
        return np.empty((0, 0, 0), dtype=np.float32)

    all_windows = [rec.windows for rec in recordings]
    combined = np.concatenate(all_windows, axis=0)

    print(f"Combined windows shape: {combined.shape}")
    return combined.astype(np.float32)


# ============================================================
# 10. Diagnostic plotting
# ============================================================

def plot_recording_summary(
    X: np.ndarray,
    windows: np.ndarray,
    metadata: RecordingMetadata,
    num_traces_to_plot: int = 5,
) -> None:
    """
    Make simple plots so the user can visually verify the pipeline.

    Plots:
        1. A few normalized traces
        2. Mean activity over time
        3. One example window
    """

    print("\nCreating diagnostic plots...")

    num_neurons, num_timepoints = X.shape
    n_plot = min(num_traces_to_plot, num_neurons)

    plt.figure(figsize=(12, 4))
    for i in range(n_plot):
        plt.plot(X[i] + i * 4, label=f"Neuron {i}")

    plt.title(f"Example normalized ΔF/F traces: {metadata.session_id}")
    plt.xlabel("Time")
    plt.ylabel("Normalized activity + offset")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 3))
    plt.plot(X.mean(axis=0))
    plt.title(f"Mean normalized activity over time: {metadata.session_id}")
    plt.xlabel("Time")
    plt.ylabel("Mean normalized ΔF/F")
    plt.tight_layout()
    plt.show()

    example_window = windows[0]

    plt.figure(figsize=(12, 4))
    for i in range(n_plot):
        plt.plot(example_window[i] + i * 4, label=f"Neuron {i}")

    plt.title(f"First model window: {metadata.session_id}")
    plt.xlabel("Window time")
    plt.ylabel("Normalized activity + offset")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()


# ============================================================
# 11. Quick synthetic test
# ============================================================

def make_tiny_fake_dff(
    num_neurons: int = 8,
    num_timepoints: int = 1000,
    seed: int = 42,
) -> np.ndarray:
    """
    Create fake ΔF/F-like traces so we can test the pipeline
    before using real lab data.

    This is NOT the final synthetic graph generator.
    This is just a quick sanity check.
    """

    rng = np.random.default_rng(seed)

    time = np.linspace(0, 20, num_timepoints)
    X = []

    for neuron in range(num_neurons):
        slow = 0.5 * np.sin(0.2 * time + neuron)
        fast = rng.normal(0, 0.2, size=num_timepoints)

        # Add sparse event-like bumps.
        event_times = rng.choice(num_timepoints, size=20, replace=False)
        events = np.zeros(num_timepoints)
        events[event_times] = rng.uniform(1, 3, size=len(event_times))

        trace = slow + fast + events
        X.append(trace)

    return np.array(X, dtype=np.float32)


if __name__ == "__main__":
    """
    This block runs only if you execute:

        python src/data/pipeline.py

    It creates fake ΔF/F data and runs the full pipeline.
    """

    print("Running quick pipeline sanity check with fake ΔF/F data...")

    fake_X = make_tiny_fake_dff()

    save_path = "data/synthetic/tiny_fake_dff.npy"
    Path("data/synthetic").mkdir(parents=True, exist_ok=True)
    np.save(save_path, fake_X)

    metadata = RecordingMetadata(
        session_id="fake_session_001",
        animal_id="fake_mouse_001",
        frame_rate=30.0,
        condition="synthetic_baseline",
        cell_ids=[f"cell_{i}" for i in range(fake_X.shape[0])],
    )

    processed = process_single_recording(
        file_path=save_path,
        metadata=metadata,
        expected_orientation="neurons_by_time",
        window_size=200,
        stride=100,
        make_plots=True,
    )

    print("\nSanity check complete.")