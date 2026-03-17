## Feature ranking (ST-GNN / MTGNN)

This folder contains feature ranking experiments for the ST-GNN (MTGNN-style) model. The purpose here is not to perform final feature selection, but rather to carry out a ranking or screening step. We begin with a well-performing “base” feature set, which is defined by prior LSTM experiments as including dynamic features like mean precipitation and mean temperature, as well as the static feature mean elevation, and then systematically test how individual candidate variables (including SWE lag features) impact performance.

Downstream, we re-check and confirm these ranking-stage conclusions using SFFS (Sequential Forward Floating Selection) in the `feature_selection/` folder.

### Files in this folder

- `st_gnn_feature_ranking.ipynb`: initial one-feature-at-a-time ranking on top of the LSTM-best base feature set.
- `st_gnn_feature_ranking_base7.ipynb`: ranking with SWE lag-7 included in the base.
- `st_gnn_feature_ranking_base30.ipynb`: ranking with SWE lag-30 included in the base, to contrast lag 7 vs lag 30 behavior.
- `mtgnn.py`: MTGNN / ST-GNN architecture used by all experiments (graph learning, temporal convolutions, mix-hop propagation, etc.).
- `train_eval_pipeline_feature_ranking.py`: shared training and evaluation pipeline (metrics, early stopping, MLflow logging, and test evaluation) used by the feature ranking notebooks.
> **Note:** Both `mtgnn.py` and `train_eval_pipeline_feature_ranking.py` in this folder are thin wrappers that delegate to canonical implementations in `../core/`. See `core/README.md` for details about the shared code organization and pipeline structure.

### What “feature ranking” means in these notebooks

Across notebooks, the ranking procedure follows the same idea:

- Start from a base feature set (best-performing LSTM feature configuration).
- Add one candidate feature group at a time (or force a specific SWE lag into the base, then add one more feature at a time).
- Train multiple replicates (where enabled) and summarize performance using KGE (primary) and MSE

### Results summary

#### 1) `st_gnn_feature_ranking.ipynb`

**Goal:** Starting from the LSTM-best dynamic + static base, test whether adding a single extra feature improves validation KGE.

**Setup:**  
- Base configuration: LSTM-selected feature set (dynamic + static).  
- Extra features tested individually:
  - snow_cover, mean_hum, mean_srad, mean_vs
  - mean_swe_lag_7, mean_swe_lag_30
  - slope, Predominant Snow, Mean Forest Cover
  - mean_pr_djf, mean_tair_djf, area
- The final results show val_mse and val_kge for each added feature.
 
**Key observation (validation KGE):** 
- Adding `mean_swe_lag_7` ranks highest (for example, val_kge around 0.99).
- Adding `mean_swe_lag_30` is also strong but lower than lag 7.
- The base configuration is clearly below these lag-augmented variants, suggesting that including SWE lag information materially improves validation skill in this sweep.
  
**Takeaway:** 
Under this one-variable-added-to-base experiment, SWE lag 7 emerges as the single most valuable additional feature, outperforming both the original base and the 30‑day SWE lag variant in validation KGE.

#### 2) `st_gnn_feature_ranking_base7.ipynb` (ranking with SWE lag-7 in the base)

**Goal:** 
Assume we always include SWE lag 7. Then, is there any other feature that adds value on top of base + lag 7?

**Setup:**  
- Base = LSTM-best feature set + `mean_swe_lag_7`.
- For each extra static/dynamic feature, run multiple replicates and log validation, UA test, and ERA5 test KGE/MSE.
- Aggregate results across replicates into a ranked table with:
  - val_kge_mean, ua_test_kge_mean, era5_test_kge_mean
  - deltas vs base (delta_*_vs_base columns).

**Key findings**
- Rows with `added_feature = base` represent “base + lag 7 only”.
- Many additions (for example, base+Predominant Snow, base+slope, base+Mean Forest Cover) can match or slightly exceed base+lag 7 in individual replicates.
  - When averaged across replicates, gains versus base+lag 7 are small and unstable.
  - None shows a clean, consistent improvement across validation, UA test, and ERA5 test simultaneously; several features improve one metric while slightly hurting another.

**Takeaway:** 
Once lag 7 is included, no single additional feature yields a robust, cross-metric improvement. The clean, defensible choice is to keep base + lag 7, without extra add-ons for the final model.

#### 3) `st_gnn_feature_ranking_base30.ipynb` (ranking with SWE lag-30 in the base)

**Goal:** 
Mirror the base7 experiment but with SWE lag 30 fixed into the dynamic feature set.

**Setup:** 
- Base = LSTM-best feature set + `mean_swe_lag_30`.
- Same per-feature addition and replicate strategy as base7.

**Key findings**
- Base+lag30 itself is strong, but:
  - Several single additions can reach very high validation KGE in individual replicates (for example, base+area, base+Predominant Snow).
  - However, when averaging over replicates and looking at validation, UA test, and ERA5 test together, none of these additions is cleanly and consistently better than base+lag30.

**Takeaway:**  
With lag 30 forced into the base, the ranking story is similar to the lag‑7 case: extra features do not provide a clear, stable improvement across all metrics. The evidence again points to “base + SWE lag” as the core useful pattern, with additional variables giving only marginal or replicate-dependent gains.

### Practical conclusions carried forward (to be validated by SFFS)

From these three ranking experiments, we carry forward the following working hypotheses into `feature_selection/`:

- SWE lag features are important, with lag‑7 frequently ranking very highly in validation.
- After adding a strong SWE lag feature, many additional variables show diminishing returns (often small, replicate-dependent changes).