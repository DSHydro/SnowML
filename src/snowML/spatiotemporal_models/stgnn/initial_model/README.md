# `initial_model` — Initial ST-GNN sanity-check (HUC12 SWE)

This folder contains the **first ST-GNN/MTGNN-style model** we trained in SnowML, as a **sanity check** when adapting ideas from the ST-GNN groundwater paper ([Taccari et al., 2024](https://www.nature.com/articles/s41598-024-75385-2)) to **snow / SWE forecasting**.

The goal of this stage was **not** to find the best ST-GNN configuration. It was to verify that a graph-based spatiotemporal model can beat/compete with our existing baselines when trained on the **same data + feature set** as the best-performing LSTM run (**Experiment 5**), before moving on to feature analysis and the “final” ST-GNN iterations.

## What’s in this folder

- **`mtgnn.py`**
  - PyTorch implementation of the MTGNN backbone (“Connecting the Dots” style): dilated temporal convolutions + (optional) graph convolution blocks.
  - Supports either:
    - **predefined adjacency** (`build_adj=False`, pass `A_tilde`), or
    - **learned adjacency** (`build_adj=True`, graph constructor can use static features `FE` when `xd` is provided).

- **`train_model.py`**
  - Minimal training + evaluation loop with **MLflow** logging.
  - Implements:
    - Sliding window dataset (on-the-fly windows)
    - Metrics: **MSE**, **MAE**, **R²**, **KGE** (Kling–Gupta efficiency)
    - `train_model(...)` and `test_model(...)`

- **`st_gnn_mlflow.ipynb`**
  - End-to-end pipeline used for the initial ST-GNN run:
    - loads HUC geometries
    - loads “model-ready” time series + static metadata
    - constructs adjacency + tensors
    - trains + tests with MLflow
    - exports predictions/targets + per-HUC metrics

- **`stgnn_huc12_wise_metrics.csv`**
  - Per-HUC12 evaluation table (includes snow type and mean elevation columns used later for comparison plots).

- **`model_comparisons.ipynb`**
  - Compares **initial ST-GNN** vs **ConvLSTM** vs **Ex5 LSTM** by snow type and HUC8.

## How this connects to later work

Once this initial run showed promising results relative to ConvLSTM/LSTM, we moved to:

- **feature analysis / ablations** to find the best ST-GNN configuration
- **final ST-GNN model** notebooks/folders (spatial/temporal splits, improved graph/features, etc.)

