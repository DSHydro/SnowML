## ST-GNN core (shared code)

This folder contains shared, canonical implementations used across the ST-GNN experiment folders.

The experiment folders keep thin wrapper files (for example `final_model_temporal_split/train_eval_pipeline_temporal_holdout.py`) so notebooks can continue to do simple local imports while the actual code lives here.

### What’s in here

- `mtgnn.py`  
  The MTGNN / ST-GNN architecture used throughout this repo.

- `train_eval_feature_ranking.py`  
  Training + evaluation utilities used by the feature ranking notebooks.

- `train_eval_feature_selection.py`  
  Training + evaluation utilities used by feature selection (SFFS) and the time-split final model.

- `train_eval_spatial_holdout.py`  
  Training + evaluation utilities for the strict spatial holdout setup (masked loss/metrics on train nodes + helpers for held-out test nodes).

### Wrapper mapping (where notebooks import from)

- Feature ranking notebooks import:
  - `feature_analysis/feature_ranking/train_eval_pipeline_feature_ranking.py` → `core/train_eval_feature_ranking.py`

- Feature selection notebooks import:
  - `feature_analysis/feature_selection/train_eval_pipeline_feature_selection.py` → `core/train_eval_feature_selection.py`

- Final model (temporal split) notebook imports:
  - `final_model_temporal_split/train_eval_pipeline_temporal_holdout.py` → `core/train_eval_feature_selection.py`

- Final model (spatial split) notebook imports:
  - `final_model_spatial_split/train_eval_pipeline_spatial_holdout.py` → `core/train_eval_spatial_holdout.py`

