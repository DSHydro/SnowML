## Final model: temporal split (time holdout)

This folder contains the “final model” ST-GNN (MTGNN-style) experiment trained using a strict time split: the model is trained/validated on an initial portion of the time series and tested on a held‑out future period.

This setup avoids the generalization failure observed in the spatial (HUC holdout) experiment: instead of requiring the model to extrapolate to unseen basins, it evaluates temporal forecasting skill on the same set of HUC12 basins.

### Files in this folder

- `st_gnn_final_model_and_evaluation.ipynb`  
  End‑to‑end notebook that:
  - loads and prepares model‑ready per‑HUC time series,
  - applies a time split (`train_idx`) and normalizes dynamic features using only the training period,
  - trains the ST‑GNN for multiple random seeds and logs runs to MLflow,
  - evaluates on the held‑out time period,
  - (optionally) rebuilds local CSV/Zarr outputs from MLflow artifacts.

- `mtgnn.py`  
  MTGNN / ST‑GNN architecture used by the notebook.

- `train_eval_pipeline_temporal_holdout.py`  
  Training and evaluation utilities used by the notebook (sliding window dataset, KGE/MSE/MAE/R² metrics, train/val loop, test inference helpers, and MLflow logging).

> **Note:** Both `mtgnn.py` and `train_eval_pipeline_temporal_holdout.py` in this folder are thin wrappers that delegate to canonical implementations in `../core/`. See `core/README.md` for details about the shared code organization and pipeline structure.

- `dhsvm_lidar_hucs.csv`  
  HUC12 list used to identify lidar and DHSVM basins (basin `lidar` vs named DHSVM basins). This is the same file used for spatial split experiments as well, supporting consistent basin filtering and reporting across both setups.

### Temporal split setup

Key settings used in the notebook:

- Sequence length: `SEQ_LEN = 30`
- The training and test split is defined as `TRAIN_SIZE_FRACTION = 0.67`, so 67% of the data is used for training+validation and the remaining 33% for testing (held-out future period).
  - **Training dates:** start from `1983-10-08` and end at `2009-11-17`
  - **Test dates:** start from `2009-11-18` and end at `2022-09-29`
- Within the training portion, an 80/20 split is used for train/validation (`VAL_SPLIT = 0.2`), i.e., 80% of the training data is used for training and 20% for validation.


### Model features

- The notebook uses the same selected feature set as in the feature selection stage (i.e., `['mean_pr', 'mean_tair', 'Mean Elevation', 'mean_swe_lag_7']`).

### Training + evaluation procedure

For each seed (5 seeds), the notebook:

1. Trains a model with `train_model(...)` on the training period.
2. Runs inference on the held‑out time period using `test_model(...)`.
3. Logs:
   - validation metrics (`val_kge`, `val_mse`, etc.) per epoch and best‑epoch summary,
   - test metrics (`test_kge`, `test_mse`, etc.) for the held‑out period,
   - a per‑seed Zarr artifact containing predictions + targets for the test period.

### Results (as printed in the notebook)

- Per‑seed metrics:
  - seed 41: val_kge 0.983588, val_mse 0.000449, test_kge 0.985812, test_mse 0.000539
  - seed 42: val_kge 0.984384, val_mse 0.000404, test_kge 0.986790, test_mse 0.000485
  - seed 43: val_kge 0.987061, val_mse 0.000551, test_kge 0.988548, test_mse 0.000681
  - seed 44: val_kge 0.979953, val_mse 0.000429, test_kge 0.976335, test_mse 0.000517
  - seed 45: val_kge 0.985600, val_mse 0.000645, test_kge 0.988652, test_mse 0.000781

- Aggregate across seeds:
  - Mean val_kge: 0.984117 (std 0.002672)
  - Mean val_mse: 0.000495 (std 0.000100)
  - Mean test_kge: 0.985227 (std 0.005114)
  - Mean test_mse: 0.000601 (std 0.000126)

Overall, the time‑split test performance is strong and stable across seeds, and substantially better behaved than the spatial holdout results.

### Outputs (CSV / Zarr)

The notebook supports two levels of outputs:

1. Per‑seed Zarr artifacts logged to MLflow during training:
   - `st_gnn_seed_<seed>_time_split_test_predictions.zarr`

2. Optional local “all runs combined” outputs rebuilt from MLflow (no retraining):
   - `st_gnn_time_split_test_results_all_runs.csv`
   - `st_gnn_time_split_test_results_all_runs.zarr` (xarray-compatible, consolidated)

The combined CSV includes prediction columns (`pred_run_1` … `pred_run_5`, plus `pred_mean` and `pred_std`) and also carries auxiliary SWE series and static metadata (as built in the notebook), including:

- `ua_swe` (target)
- `era5_swe`
- `snodas_swe`
- `Predominant Snow`
- `Mean Elevation`

> **Note (outputs stored in S3, not committed to git):** If you just want the combined “all runs” outputs (without rebuilding from MLflow), they are available in S3:
> - `s3://spatiotemporal-results/stgnn/temporal_split/st_gnn_time_split_test_results_all_runs.csv.gz` (csv file)
> - `s3://spatiotemporal-results/stgnn/temporal_split/st_gnn_time_split_test_results_all_runs.tar.gz` (archive containing the Zarr store)

### How to reproduce / inspect results

1. Run `st_gnn_final_model_and_evaluation.ipynb` top‑to‑bottom to retrain and log runs to MLflow.
2. To regenerate local CSV/Zarr without retraining:
   - run the “Rebuild local outputs from MLflow (no model run)” section.
3. To quickly inspect the saved Zarr results:
   - open the consolidated Zarr with `xarray.open_zarr("st_gnn_time_split_test_results_all_runs.zarr", consolidated=True)`.

