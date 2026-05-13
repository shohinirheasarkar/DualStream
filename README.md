# DualStream

A research prototype for estimating directed predictive functional connectivity from calcium imaging traces using a dual-timescale sequence model.

## Core Idea

Standard calcium-to-connectivity pipelines often assume that most calcium trace variance is neuronal. This project tests an alternative approach: separate calcium traces into fast neuronal-like dynamics and slow modulatory dynamics before estimating directed functional connectivity.


## Working DualStream Prototype

At this stage, the repo has a working end-to-end prototype of the main idea: take observed calcium traces, separate them into fast and slow latent dynamics, and use the fast latent stream to estimate a directed connectivity matrix.

The current version doesn't use full Mamba yet. Instead, it uses lightweight GRU-based temporal modules as a prototype version of the fast and slow sequence models. The long-term plan is to replace these GRU modules with true selective state-space / Mamba blocks.

---

## Main Pipeline

The full prototype follows this flow:

```text
Observed ΔF/F traces
        ↓
Input projection
        ↓
Shared temporal encoder
        ↓
Fast stream + Slow stream
        ↓
Fusion block
        ↓
Reconstruction decoder + Future prediction decoder
        ↓
Connectivity matrix computed from the fast latent stream
        ↓
Evaluation against synthetic ground-truth graph
```

The input shape is:

```text
(batch, neurons, time)
```

The final connectivity output is:

```text
C ∈ R^(neurons × neurons)
```

where:

```text
C[target, source]
```

means “how strongly the past activity of the source neuron predicts the target neuron.”

---

## Model Architecture

The core model lives in:

```text
src/models/dual_stream_model.py
```

This file defines the main DualStream architecture.

### Input Projection

The input projection layer takes each neuron’s ΔF/F trace and maps it into a higher-dimensional latent space. This gives the model more room to learn useful temporal features from the raw calcium trace.

### Shared Temporal Encoder

The shared temporal encoder learns general time-based patterns from the calcium recording before splitting into fast and slow streams. I think of this as the model first building a common understanding of the trace before deciding which parts are fast neuronal dynamics and which parts are slower modulation.

### Fast Stream

The fast stream is meant to capture rapid, spike-like transient activity. This is the stream used to compute the final directed connectivity matrix, because the project’s main hypothesis is that connectivity should be estimated from fast predictive neural dynamics rather than from raw calcium traces that may be contaminated by slow modulation.

### Slow Stream

The slow stream is meant to capture slower modulatory structure, such as bleaching drift, global low-frequency fluctuations, or other slow biological/technical effects. Importantly, I am not claiming this is automatically astrocyte activity; for now, it is more carefully described as “slow latent modulatory dynamics.”

### Fusion Block

The fusion block combines the fast and slow streams into one shared representation. This lets the model use both fast and slow information when reconstructing or predicting calcium activity.

### Reconstruction Decoder

The reconstruction decoder tries to rebuild the original ΔF/F traces from the learned latent representation. This helps make sure the model does not throw away important information while separating the trace into fast and slow parts.

### Future Prediction Decoder

The future prediction decoder tries to predict upcoming calcium activity. This is important because the goal is not just to compress the data, but to learn temporal structure that is actually predictive.

### Connectivity Head

The connectivity head computes a directed neuron-to-neuron matrix from the fast latent stream. For each possible source-target pair, it compares the source neuron’s past fast latent activity to the target neuron’s current fast latent activity. These scores are averaged across time and lags to produce the final matrix.

---

## Loss Functions

The loss functions live in:

```text
src/losses/losses.py
```

The total loss is a weighted combination of several smaller losses. Each one encourages the model to learn a different useful behavior.

### Reconstruction Loss

Checks whether the model can reconstruct the original calcium traces. This makes sure the latent representation still contains meaningful information from the input.

### Future Prediction Loss

Checks whether the model can predict the next time step. This encourages the model to learn predictive temporal structure instead of just memorizing the input.

### Fast Sparsity Loss

Encourages the fast stream to be sparse and event-like instead of active everywhere all the time. This is meant to make the fast stream behave more like transient neural activity.

### Slow Smoothness Loss

Encourages the slow stream to change gradually over time. This helps the slow stream capture slow modulation rather than noisy fast fluctuations.

### Fast/Slow Decorrelation Loss

Encourages the fast and slow streams to learn different information. Without this, both streams could accidentally learn the same thing.

### Connectivity Sparsity Loss

Encourages the connectivity matrix to be cleaner and less noisy. Biologically, we usually do not expect every neuron to strongly connect to every other neuron.

### Variance Preservation Loss

Prevents the latent representation from collapsing into a constant value. This makes sure the model is still learning meaningful variation.

### Optional Orthogonality and Graph Stability Losses

The orthogonality loss further encourages fast and slow streams to point in different latent directions. The graph stability loss is included as a placeholder for future time-varying connectivity work, where nearby windows should have somewhat stable connectivity estimates.

---

## Connectivity Matrix Computation

The connectivity matrix is now computed from the **fast latent stream**, not directly from raw traces.

This is the important update.

The logic is:

```text
source neuron past fast latent activity
        compared with
target neuron current fast latent activity
```

For each source-target pair, the model computes a similarity score across several time lags. The scores are averaged over time and over lags. The diagonal is removed so neurons do not connect to themselves, and the rows are normalized so that each target neuron’s incoming connection strengths are easier to interpret.

The final output is:

```text
C[target, source]
```

So if `C[i, j]` is large, that means source neuron `j` has strong predictive influence on target neuron `i`.

---

## Training Loop

The training loop lives in:

```text
src/training/train.py
```

The training loop does the following:

1. Loads the synthetic observed ΔF/F traces.
2. Splits the recording into temporal windows.
3. Trains the DualStream model.
4. Tracks training and validation loss.
5. Uses early stopping.
6. Saves the best model checkpoint.
7. Computes the final connectivity matrix from the fast latent stream.
8. Evaluates the learned connectivity matrix against the known synthetic ground-truth graph.
9. Saves metrics and figures.

The one-command driver script is:

```text
scripts/run_dualstream.py
```

Run it with:

```bash
PYTHONPATH=. python scripts/run_dualstream.py
```

---

## Evaluation Metrics

Evaluation utilities live in:

```text
src/evaluation/metrics.py
```

Because the synthetic dataset has a known directed ground-truth graph, the learned connectivity matrix can be evaluated directly.

The current metrics include:

- AUROC
- AUPRC
- Precision@K
- Recall@K
- F1@K
- Edge direction accuracy
- Reconstruction error
- Future prediction error

This gives us a way to test whether the learned connectivity matrix is actually recovering the known graph structure, rather than just producing a nice-looking heatmap.

---

The current GRU-based model is a placeholder for future selective state-space modules. The next major research step is to replace the fast and slow GRU streams with true Mamba/selective SSM layers and compare whether that improves graph recovery, scalability, and interpretability.

