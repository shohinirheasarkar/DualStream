# NeuroMamba

A research prototype for estimating directed predictive functional connectivity from calcium imaging traces using a dual-timescale sequence model.

## Core Idea

Standard calcium-to-connectivity pipelines often assume that most calcium trace variance is neuronal. This project tests an alternative approach: separate calcium traces into fast neuronal-like dynamics and slow modulatory dynamics before estimating directed functional connectivity.

## Input

The primary input is a matrix of calcium traces:


X ∈ R^(N × T)


## Data Pipeline

The main preprocessing pipeline for this project lives in:

```text
src/data/pipeline.py
```

Right now, the pipeline is designed to work with **ΔF/F calcium traces** that have already gone through ROI extraction / segmentation (ex. CaImAn outputs). The expected input shape is:

```text
(num_neurons, num_timepoints)
```

The pipeline currently supports:

```text
.npy
.csv
.h5 / .hdf5
```

(with NWB support planned later)

The pipeline handles:

- loading calcium traces
- standardizing trace orientation
- interpolating over NaNs / missing values
- per-neuron normalization
- segmenting long recordings into overlapping windows
- metadata storage (animal ID, session ID, frame rate, condition, etc.)
- train/val/test splitting by session or animal to avoid leakage

The final model-ready output shape is:

```text
(num_windows, num_neurons, window_size)
```

The file also includes a tiny synthetic ΔF/F generator so the entire preprocessing pipeline can be tested before using real data.

---

## Environment Setup

Create the conda environment:

```bash
conda create -n dualstream python=3.10
conda activate dualstream
pip install -r requirements.txt
```

Main packages used:

```text
numpy          numerical processing
pandas         CSV handling
torch          deep learning models
matplotlib     visualizations
scikit-learn   metrics and utilities
networkx       graph/network analysis
statsmodels    Granger causality baselines
pytest         testing
```

---

## Running the Pipeline

To run the preprocessing pipeline sanity check:

```bash
python src/data/pipeline.py
```

This generates fake ΔF/F traces, processes them through the pipeline, and produces diagnostic visualizations.

---

## Running Tests

Run the pipeline tests with:

```bash
PYTHONPATH=. pytest tests/test_pipeline.py -v
```

The tests check:

- shape standardization
- missing value handling
- normalization
- windowing
- metadata handling
- leakage-safe splitting
- `.npy` and `.csv` loading
- full pipeline execution

