# NeuroMamba

A research prototype for estimating directed predictive functional connectivity from calcium imaging traces using a dual-timescale sequence model.

## Core Idea

Standard calcium-to-connectivity pipelines often assume that most calcium trace variance is neuronal. This project tests an alternative approach: separate calcium traces into fast neuronal-like dynamics and slow modulatory dynamics before estimating directed functional connectivity.

## Input

The primary input is a matrix of calcium traces:

```text
X ∈ R^(N × T)

