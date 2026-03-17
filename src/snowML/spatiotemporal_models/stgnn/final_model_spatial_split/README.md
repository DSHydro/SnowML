## Final model: spatial split (HUC holdout)

This folder contains the first attempt at a “final” ST-GNN model trained with a **spatial holdout**: some HUC12 basins are used for training/validation, and a disjoint set of HUC12s is reserved for testing.  
The idea is to test the model’s ability to generalize to entirely unseen locations.

In practice, although validation performance on the training HUCs is strong, the held‑out HUC test performance is poor (including negative KGE).  
Because of this, the project switches to a **time‑split final model** (holding out time on the same HUCs) instead of using this spatial split as the main result.

### Files in this folder

- `st_gnn_final_model_and_evaluation.ipynb`  
  End‑to‑end notebook that:
  - constructs train/test HUC12 splits and adjacency matrices,
  - trains the ST-GNN on train HUCs only (masked loss),
  - evaluates on held‑out HUCs,
  - saves per‑run predictions to CSV and Zarr,
  - computes aggregate test metrics across seeds.

- `mtgnn.py`  
  MTGNN / ST-GNN architecture (graph constructor, temporal convolutions, mix‑hop propagation, etc.), shared with the feature analysis folders.

- `train_eval_pipeline_spatial_holdout.py`  
  Training and evaluation pipeline specialized for spatial holdout:
  - loss and validation metrics computed only on `train_node_idx`,
  - The full adjacency matrix is partitioned to create `A_train` (training basins only) and `A_test` (test basins only):
    - `A_train` contains only train–train edges (all other rows/cols zeroed out).
    - `A_test` contains only test–test edges (all other rows/cols zeroed out), and is used during test evaluation with `test_node_idx`.

> **Note:** Both `mtgnn.py` and `train_eval_pipeline_spatial_holdout.py` in this folder are thin wrappers that delegate to canonical implementations in `../core/`. See `core/README.md` for details about the shared code organization and pipeline structure.

- `dhsvm_lidar_hucs.csv`
  This file contains the list of HUC12 basins available in the lidar (Olympic region) and DHVSM (Green, Cedar, etc.) datasets. Any HUC12s that overlap between the LSTM set and the DHVSM set are removed from training, ensuring that these overlapping basins are used only for testing in the spatial split experiment.


### Spatial split setup

The spatial split is configured as follows in `st_gnn_final_model_and_evaluation.ipynb`:

- Nodes correspond to HUC12 basins.
- The set of HUC12s is partitioned into:
  - `train_node_idx`: indices of HUC12 basins from the LSTM set that are not present in either the DHVSM or lidar datasets (i.e., LSTM-only HUCs). These are used for training and validation.
  - `test_node_idx`: indices of HUC12 basins that are present in the lidar (Olympic region) and DHVSM (Green, Cedar, etc.) datasets as listed in `dhsvm_lidar_hucs.csv`. 
- Two adjacency matrices are built:
  - `A_train`: connectivity among training basins (test rows/cols zeroed out).
  - `A_test`: connectivity among test basins (used at prediction time).
- The pipeline uses:
  - a sliding‑window dataset over time,
  - the same selected feature set as in the feature selection stage (i.e., `['mean_pr', 'mean_tair', 'Mean Elevation', 'mean_swe_lag_7']`),
  - multiple random seeds to train 5 independent runs.

In `train_eval_pipeline_spatial_holdout.train_model`, loss and validation metrics are computed only over `train_node_idx`, so test basins remain fully held out during training (no leakage of target values).

### Training and validation performance

The notebook trains the model for 5 seeds (41–45) and records validation metrics per seed. A typical output (as printed in the notebook) is:

- Validation metrics per seed (approximate):
  - val_kge per seed: around 0.968–0.989
  - val_kge mean: 0.9813, std: 0.0088
  - val_mse mean: 0.000506, std: 0.000133

This indicates that, on the **training HUCs**, the model fits very well: KGE is close to 1 and MSE is small.

### Test evaluation on held‑out HUCs

After training, the notebook:

1. Runs `run_test_predictions` for each trained model, using:
   - `A_test` and `test_node_idx`,
   - the full temporal range for the test basins.
2. Saves:
   - the full prediction tensor `all_preds` (shape `[n_runs, n_test_nodes, n_test_timesteps]`),
   - the corresponding targets, HUC IDs, and dates,
   to both:
   - `st_gnn_huc_split_test_results_all_runs.csv`
   - `st_gnn_huc_split_test_results_all_runs.zarr`
3. Reloads the Zarr store and computes test metrics per run over all test HUCs and all test days.

- Per‑run test metrics:
  - run 1: mse ≈ 0.013, mae ≈ 0.10, r2 ≈ 0.77, kge ≈ 0.15
  - run 2: mse ≈ 0.016, mae ≈ 0.10, r2 ≈ 0.70, kge ≈ 0.60
  - run 3: mse ≈ 0.105, mae ≈ 0.28, r2 ≈ −0.89, kge ≈ −1.40
  - run 4: mse ≈ 0.024, mae ≈ 0.13, r2 ≈ 0.57, kge ≈ −0.07
  - run 5: mse ≈ 0.020, mae ≈ 0.13, r2 ≈ 0.64, kge ≈ −0.11

and the aggregated statistics:

- Test kge mean: about −0.17 (std ≈ 0.67)
- Test mse mean: about 0.036 (std ≈ 0.035)
- Test mae mean: about 0.150
- Test r2 mean: about 0.36

So while some individual runs achieve modestly positive KGE on the test HUCs, others are strongly negative, and the mean KGE over runs is **negative**.

- **Test outputs (stored in S3)**  
  The “all runs combined” outputs for this spatial split are stored in S3:
  - `s3://spatiotemporal-results/stgnn/spatial_split/st_gnn_huc_split_test_results_all_runs.tar.gz` (archive containing the Zarr store)
    - Zarr store containing predictions and targets for all test HUCs and all test days, for each seed/run.
  - `s3://spatiotemporal-results/stgnn/spatial_split/st_gnn_huc_split_test_results_all_runs.csv.gz`
    - The combined CSV is a flat representation of test predictions with columns:
      - `huc_id`, `day`
      - `pred_run_1` … `pred_run_5`
      - `pred_mean`, `pred_std`
      - `UA_swe` (target SWE)

### Why these spatial results are not used as the main final model

In summary:

- Validation metrics on the training HUCs are very strong (KGE near 0.98, low MSE).
- When evaluated on **completely unseen HUC12s**, the model’s performance degrades sharply:
  - average KGE across runs becomes negative,
  - MSE and MAE are much larger than on the training HUCs,
  - r2 is highly variable across seeds.

This is expected to some extent: under a strict spatial holdout,

- the model never sees target values for the test HUCs during training,
- the training adjacency `A_train` contains only train–train edges, so test basins are effectively isolated until prediction time,
- spatial extrapolation to entirely unseen basins is much harder than interpolation in time at known locations.

Given these issues, this spatial‑split experiment is useful as a stress test, but it does not provide the level of skill we want to present as the primary “final model” result.

For that reason, the project shifts to a **time‑split final model** (see the `final_model_temporal_split` folder), where:

- the same HUC12 basins appear in train/validation/test,
- the holdout is over time periods instead of space,
- and the resulting test metrics are more stable and better reflect the model’s practical skill at the chosen locations.

### How to reproduce and inspect results

To rerun or inspect this spatial‑split experiment:

1. Open `st_gnn_final_model_and_evaluation.ipynb`.
2. Run the cells that:
   - construct the train/test HUC splits and adjacency matrices,
   - train the model for the chosen seeds,
   - run test predictions and save CSV/Zarr outputs.
3. To recompute test metrics from the saved data only (without retraining):
   - First, download the required files from the S3 bucket at `spatiotemporal-results/stgnn/spatial_split`.
   - Then, run the final evaluation cells that load `st_gnn_huc_split_test_results_all_runs.zarr` and call `compute_metrics` from `train_eval_pipeline_spatial_holdout.py`.

The CSV and Zarr files can also be used directly (outside the notebook) for any additional spatial analysis of prediction errors across basins and time.

