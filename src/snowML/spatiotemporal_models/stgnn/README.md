## ST-GNN / MTGNN (spatiotemporal models)

This folder contains ST-GNN experiments built around an MTGNN-style architecture for SWE prediction across HUC12 basins.

The workflow in this repo follows a simple progression:

- feature ranking (screen promising variables and SWE lags)
- feature selection (confirm a compact feature set via SFFS)
- final model training and evaluation (time split is the primary final setup)

### Folder map

- `feature_analysis/feature_ranking/`  
  One-feature-at-a-time ranking experiments and a folder README summarizing the results.

- `feature_analysis/feature_selection/`  
  SFFS-based feature selection experiments (with MLflow-backed results) and a folder README summarizing the final selected feature set.

- `final_model_spatial_split/`  
  Spatial holdout (HUC split) “final model” attempt. This is kept mainly as a stress test; it fits training HUCs well but generalizes poorly to unseen HUCs (unstable / negative KGE), which motivates using a time split for the main final results.

- `final_model_temporal_split/`  
  The primary final model setup using a strict time holdout on the same HUCs. This produces strong and stable test metrics across seeds and is the recommended entry point for final evaluation.

- `core/`  
  Contains core/shared code used by all MTGNN experiments, including the model architecture and utilities for training, evaluation, and data handling.

### Outputs and results

Across the experiments you will see:

- MLflow runs (metrics and artifacts per experiment/seed)
- CSV outputs (flat per-timestep predictions)
- Zarr outputs (multi-run prediction tensors, easier for analysis at scale)

Each subfolder README documents exactly what is written and how to reproduce or reload results.

