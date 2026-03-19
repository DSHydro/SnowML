# ST-GNN Model Evaluation

This folder holds notebooks and data used to evaluate the Spatio-Temporal Graph Neural Network (ST-GNN) snow water equivalent (SWE) predictions against other models and reference products.

## Contents

| File | Description |
|------|-------------|
| **model_evaluation_graphs.ipynb** | Compares ST-GNN to reference SWE sources (DHSVM, UCLA, UA, ERA5, SNODAS, LiDAR). Loads model outputs and basin configs (Cedar, Green, Snohomish, Stillaguamish), aggregates SWE by basin, and produces time-series and peak-SWE (e.g. April 1) comparison plots. |
| **final_model_comparisons.ipynb** | Compares **ST-GNN**, **ConvLSTM**, and **LSTM** (Ex5) on a common set of HUC12s. Loads time-split ST-GNN and ConvLSTM results from S3 (`spatiotemporal-results`), LSTM metrics from local Ex5 outputs, filters to the same HUC IDs, computes per-HUC KGE/MSE for ST-GNN (UA as reference), and produces box plots of Test KGE by model. |
| **dhsvm_lidar_hucs.csv** | Lookup of HUC12 IDs used for DHSVM/LiDAR evaluation basins (`basin`, `huc_id`, `not_in_training`). Used by `model_evaluation_graphs.ipynb` to select HUCs for basin-level comparison with LiDAR and DHSVM. |

## Data dependencies

- **ST-GNN**: Time-split test results (e.g. `st_gnn_time_split_test_results_all_runs.csv` or from S3).
- **ConvLSTM**: Precomputed HUC12 metrics (e.g. `convlstm_results.csv` or from S3).
- **LSTM (Ex5)**: Montane/Maritime (and optionally Ephemeral) metrics from `Ex5_DataIntegration` notebooks.
- **Reference SWE**: Model-ready UA/ERA5/SNODAS, UCLA from `snowml-gold`, DHSVM from CIG URLs, LiDAR from aggregated HUC12 CSVs (e.g. `lidar_swe_huc12_aggregated_olympic.csv` in the parent `Evaluation` folder).
- **Geometry**: HUC8/HUC12 GeoJSONs and metadata from `snowml-shape` (S3).

## Running the notebooks

1. Ensure the required CSVs and S3 access are available (see Data dependencies).
2. Run cells in order; `model_evaluation_graphs.ipynb` expects config constants (e.g. `BEGIN_DATE`, `END_DATE`) to be set from the model output date range.
3. For `final_model_comparisons.ipynb`, AWS credentials and access to the `spatiotemporal-results` bucket are needed when loading ST-GNN and ConvLSTM from S3.
