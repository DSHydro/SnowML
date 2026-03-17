## Feature selection (ST-GNN / MTGNN)

This folder contains the feature selection step for the ST-GNN (MTGNN-style) model, using the ranking insights from `feature_ranking/` as a starting point.  
Here the goal is to move from “this feature looks promising” to a concrete, data-driven feature subset for the final model.

Downstream, we use these selected features in the final spatiotemporal models.

### Files in this folder

- `st_gnn_feature_selection.ipynb`: feature selection experiments using SFFS over candidate features, with results stored and queried from MLflow.
- `mtgnn.py`: MTGNN / ST-GNN architecture used for all selection experiments.
- `train_eval_pipeline_feature_selection.py`: shared training and evaluation pipeline (metrics, early stopping, MLflow logging, and test evaluation) used by the selection notebook.
> **Note:** Both `mtgnn.py` and `train_eval_pipeline_feature_selection.py` in this folder are thin wrappers that delegate to canonical implementations in `../core/`. See `core/README.md` for details about the shared code organization and pipeline structure.

### Method: Sequential Forward Floating Selection (SFFS)

SFFS is a greedy wrapper feature selection method designed to explore the space of feature subsets more flexibly than pure forward selection:

- Start from a base feature set.
- Forward step: try adding one candidate feature at a time, keep the addition that gives the best improvement in the target metric (here: validation KGE, with MSE tracked as well).
- Backward (floating) step: after an addition, optionally remove one of the already‑selected features if that removal further improves the metric.
- Repeat forward + optional backward steps until adding features no longer improves performance or a stopping criterion is met.

This gives a sequence of feature subsets that are locally optimal under the SFFS search, rather than just a simple monotone forward chain.

### Base and candidate feature sets

All three experiments in `st_gnn_feature_selection.ipynb` share the same base configuration:

- Base features:
  - `mean_pr`
  - `mean_tair`
  - `Mean Elevation`

The experiments differ only in which additional features are allowed as SFFS candidates.

- Full candidate set (before restrictions) includes:
  - SWE lag features: `mean_swe_lag_7`, `mean_swe_lag_30`
  - Other static/dynamic features such as `mean_srad`, `snow_cover`, `mean_hum`, `slope`, `Predominant Snow`, `Mean Forest Cover`, `mean_vs`, `mean_pr_djf`, `mean_tair_djf`, and others used in the ranking stage.

The notebook constrains this candidate set differently for each experiment, as described below.

### Experiments and results

#### Experiment 1: `stgnn_swe_sffs`

- Goal: run SFFS starting from the base feature set and allow SFFS to choose freely among all candidate features (including both SWE lag 7 and SWE lag 30).
- Base features:
  - `mean_pr`, `mean_tair`, `Mean Elevation`
- Candidate features for SFFS:
  - all remaining features, including `mean_swe_lag_7` and `mean_swe_lag_30`.

Results (from the SFFS run logged in MLflow):

- Final selected features:
  - `['mean_pr', 'mean_tair', 'Mean Elevation', 'mean_swe_lag_7']`
- Final validation metrics:
  - `val_kge = 0.991888`
  - `val_mse = 0.000496`

Takeaway: when both SWE lag 7 and SWE lag 30 are allowed, SFFS consistently prefers **SWE lag 7** on top of the LSTM‑based base.

#### Experiment 2: `stgnn_swe_sffs_WbaseWOlag7`

- Goal: force SFFS to explore configurations where lag 7 is not available as a candidate, to see whether lag 30 can play a similar role.
- Base features:
  - `mean_pr`, `mean_tair`, `Mean Elevation`
- Candidate features for SFFS:
  - all remaining features **except** `mean_swe_lag_7` (that is, SWE lag 7 is removed from the candidate pool).

Results:

- Final selected features:
  - `['mean_pr', 'mean_tair', 'Mean Elevation', 'mean_swe_lag_30']`
- Final validation metrics:
  - `val_kge = 0.976451`
  - `val_mse = 0.001136`

Takeaway: in the absence of lag 7, SFFS chooses **SWE lag 30** as the best available lag feature to augment the base. However, the resulting KGE/MSE are clearly worse than in Experiment 1.

#### Experiment 3: `stgnn_swe_sffs_nolag`

- Goal: understand how much of the performance gain is attributable specifically to SWE lag features by disallowing all lags.
- Base features:
  - `mean_pr`, `mean_tair`, `Mean Elevation`
- Candidate features for SFFS:
  - remaining features **excluding** all SWE lag features (no `mean_swe_lag_7`, no `mean_swe_lag_30`).

Results:

- Final selected features:
  - `['mean_pr', 'mean_tair', 'Mean Elevation', 'mean_srad']`
- Final validation metrics:
  - `val_kge = 0.923156`
  - `val_mse = 0.007484`

Takeaway: without access to SWE lags, SFFS settles on `mean_srad` as the best additional feature, but the overall KGE drops substantially compared to both lag‑7 and lag‑30 configurations.

### How to inspect / reproduce the SFFS results

The notebook `st_gnn_feature_selection.ipynb` includes a final cell that queries MLflow and builds a summary dataframe of SFFS runs.

- To view the results for a specific experiment:
  - Change `EXPERIMENT_NAME` in that last cell to one of:
    - `"stgnn_swe_sffs"`
    - `"stgnn_swe_sffs_WbaseWOlag7"`
    - `"stgnn_swe_sffs_nolag"`
  - Re-run the cell to pull the corresponding MLflow runs and metrics.

This allows you to regenerate the tables underlying the summaries above, or to inspect intermediate SFFS steps (for example, which features were added or removed at each iteration).

### Overall conclusion from feature selection

Across the three SFFS experiments, the pattern is consistent:

- When SFFS is free to choose among all candidates, the best subset is:
  - `['mean_pr', 'mean_tair', 'Mean Elevation', 'mean_swe_lag_7']`
- Forcing SFFS to work without lag 7 pushes it toward lag 30, but with a noticeable KGE/MSE degradation.
- Removing all lags degrades performance even more, even with other static/dynamic features available.

Taken together, these results support using the feature set

- `['mean_pr', 'mean_tair', 'Mean Elevation', 'mean_swe_lag_7']`

as the final, SFFS-confirmed feature configuration for the ST-GNN model in this project.

