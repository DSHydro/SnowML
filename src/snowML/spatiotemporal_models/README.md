# Spatiotemporal models (adding space to SWE prediction)

This folder contains experiments that ask a simple question:

> If a baseline LSTM only models **time**, what improves (or breaks) when we add **spatial context** for SWE prediction?

To explore that, the repo includes two spatiotemporal model families:

- **ConvLSTM** (`conv_lstm/`): adds spatial structure by predicting on gridded HUC10 tensors (e.g., `24×24` cells) using convolutional recurrence.
- **ST-GNN / MTGNN** (`stgnn/`): adds spatial structure by modeling a graph of HUC12 basins with temporal convolutions + graph operations.

Both model families are evaluated and compared against **temporal-only LSTM baselines** in `model_evaluation/`.

## Folder map

- `conv_lstm/`  
  End-to-end pipeline for ConvLSTM modeling: from gridded data preparation (HUC10 tensors), through model training and predictions, to evaluation and results analysis.

- `stgnn/`  
  Spatiotemporal graph neural network model (ST-GNN / MTGNN): graph construction, in-depth feature explorations and experiments across different splits.

- `model_evaluation/`  
  Notebooks used to compare models and generate final evaluation plots/tables (LSTM vs ConvLSTM vs ST-GNN).

- `dataprep_stgnn/`  
  Data preparation utilities and notebooks for ST-GNN model-ready datasets (e.g., DHSVM/ERA5/SNODAS/LiDAR integration).

