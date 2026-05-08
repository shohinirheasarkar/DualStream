"""
Tests for src/data/pipeline.py

These tests check that the data pipeline behaves correctly on small,
controlled examples before we trust it on real calcium imaging data.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.pipeline import (
    RecordingMetadata,
    ProcessedRecording,
    standardize_trace_shape,
    handle_missing_values,
    normalize_per_neuron,
    segment_into_windows,
    make_tiny_fake_dff,
    process_single_recording,
    split_recordings,
    combine_windows,
)


def test_make_tiny_fake_dff_shape():
    X = make_tiny_fake_dff(num_neurons=5, num_timepoints=200, seed=0)

    assert isinstance(X, np.ndarray)
    assert X.shape == (5, 200)
    assert np.isfinite(X).all()


def test_standardize_trace_shape_neurons_by_time():
    X = np.random.randn(4, 100)

    X_out = standardize_trace_shape(X, expected_orientation="neurons_by_time")

    assert X_out.shape == (4, 100)
    assert X_out.dtype == np.float32


def test_standardize_trace_shape_time_by_neurons():
    X = np.random.randn(100, 4)

    X_out = standardize_trace_shape(X, expected_orientation="time_by_neurons")

    assert X_out.shape == (4, 100)


def test_standardize_trace_shape_auto_transposes_when_rows_greater_than_cols():
    X = np.random.randn(100, 4)

    X_out = standardize_trace_shape(X, expected_orientation="auto")

    assert X_out.shape == (4, 100)


def test_standardize_trace_shape_rejects_3d_input():
    X = np.random.randn(2, 3, 4)

    with pytest.raises(ValueError):
        standardize_trace_shape(X)


def test_handle_missing_values_interpolates_nans():
    X = np.array(
        [
            [1.0, np.nan, 3.0],
            [2.0, 4.0, 6.0],
        ],
        dtype=np.float32,
    )

    X_clean = handle_missing_values(X)

    assert not np.isnan(X_clean).any()
    assert np.isclose(X_clean[0, 1], 2.0)


def test_handle_missing_values_all_nan_trace_becomes_zero():
    X = np.array(
        [
            [np.nan, np.nan, np.nan],
            [1.0, 2.0, 3.0],
        ],
        dtype=np.float32,
    )

    X_clean = handle_missing_values(X)

    assert not np.isnan(X_clean).any()
    assert np.allclose(X_clean[0], 0.0)


def test_normalize_per_neuron_mean_and_std():
    X = np.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [10.0, 20.0, 30.0, 40.0],
        ],
        dtype=np.float32,
    )

    X_norm = normalize_per_neuron(X)

    means = X_norm.mean(axis=1)
    stds = X_norm.std(axis=1)

    assert np.allclose(means, 0.0, atol=1e-6)
    assert np.allclose(stds, 1.0, atol=1e-6)


def test_segment_into_windows_shape():
    X = np.random.randn(3, 100).astype(np.float32)

    windows = segment_into_windows(X, window_size=20, stride=10)

    assert windows.shape == (9, 3, 20)


def test_segment_into_windows_rejects_short_recording():
    X = np.random.randn(3, 10).astype(np.float32)

    with pytest.raises(ValueError):
        segment_into_windows(X, window_size=20, stride=10)


def test_process_single_recording_from_npy(tmp_path):
    X = make_tiny_fake_dff(num_neurons=4, num_timepoints=100, seed=1)

    file_path = tmp_path / "fake_dff.npy"
    np.save(file_path, X)

    metadata = RecordingMetadata(
        session_id="session_001",
        animal_id="mouse_001",
        frame_rate=30.0,
        condition="baseline",
        cell_ids=["cell_0", "cell_1", "cell_2", "cell_3"],
    )

    processed = process_single_recording(
        file_path=str(file_path),
        metadata=metadata,
        expected_orientation="neurons_by_time",
        window_size=25,
        stride=25,
        make_plots=False,
    )

    assert isinstance(processed, ProcessedRecording)
    assert processed.traces.shape == (4, 100)
    assert processed.windows.shape == (4, 4, 25)
    assert processed.metadata.session_id == "session_001"


def test_process_single_recording_from_csv(tmp_path):
    X = make_tiny_fake_dff(num_neurons=3, num_timepoints=50, seed=2)

    # Save as time x neurons to mimic common CSV storage.
    df = pd.DataFrame(X.T, columns=["cell_0", "cell_1", "cell_2"])
    file_path = tmp_path / "fake_dff.csv"
    df.to_csv(file_path, index=False)

    metadata = RecordingMetadata(
        session_id="session_csv",
        animal_id="mouse_csv",
        frame_rate=20.0,
        condition="baseline",
    )

    processed = process_single_recording(
        file_path=str(file_path),
        metadata=metadata,
        expected_orientation="auto",
        window_size=10,
        stride=10,
        make_plots=False,
    )

    assert processed.traces.shape == (3, 50)
    assert processed.windows.shape == (5, 3, 10)


def test_split_recordings_by_animal_id(tmp_path):
    recordings = []

    for idx in range(4):
        X = np.random.randn(2, 40).astype(np.float32)
        windows = segment_into_windows(X, window_size=10, stride=10)

        metadata = RecordingMetadata(
            session_id=f"session_{idx}",
            animal_id=f"mouse_{idx}",
            frame_rate=30.0,
            condition="baseline",
        )

        recordings.append(
            ProcessedRecording(
                traces=X,
                windows=windows,
                metadata=metadata,
            )
        )

    split = split_recordings(
        recordings,
        split_by="animal_id",
        train_fraction=0.5,
        val_fraction=0.25,
        seed=0,
    )

    total = len(split["train"]) + len(split["val"]) + len(split["test"])

    assert total == 4
    assert len(split["train"]) == 2
    assert len(split["val"]) == 1
    assert len(split["test"]) == 1


def test_combine_windows():
    recs = []

    for idx in range(2):
        X = np.random.randn(3, 40).astype(np.float32)
        windows = segment_into_windows(X, window_size=10, stride=10)

        metadata = RecordingMetadata(
            session_id=f"session_{idx}",
            animal_id=f"mouse_{idx}",
            frame_rate=30.0,
        )

        recs.append(
            ProcessedRecording(
                traces=X,
                windows=windows,
                metadata=metadata,
            )
        )

    combined = combine_windows(recs)

    assert combined.shape == (8, 3, 10)


def test_combine_windows_empty_list():
    combined = combine_windows([])

    assert combined.shape == (0, 0, 0)