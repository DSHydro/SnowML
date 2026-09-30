# Knowledge Transfer Guide - Update Summary

**Date:** September 30, 2026  
**Version:** 2.0 (Updated from 1.0)  
**Changes:** Added comprehensive Section 9 - File Locations and Results Reference

---

## What Was Added

### NEW SECTION 9: File Locations and Results Reference

This major addition provides complete documentation of:

1. **Training Scripts by Experiment (9.1)**
   - Exp1A: `train_exp1a_full.py` (533 HUCs with ephemeral)
   - Exp1B: `train_exp1b_full.py` (270 deep snow HUCs)
   - Exp2: `finetune_exp1b_base_CORRECT.py` (fine-tuning on Yakima/Naches)
   - Complete parameters for each experiment
   - Command-line usage examples
   - Feature set definitions

2. **Results Storage Locations (9.2)**
   - **On SageMaker Studio during training:**
     - `/home/sagemaker-user/checkpoints/` (240 checkpoint files per experiment)
     - `/home/sagemaker-user/*.log` (training logs)
     - `/home/sagemaker-user/evaluation_results/` (metrics CSVs)
     - `/home/sagemaker-user/exp2_finetune_results/` (Exp2 outputs)
   
   - **On local machine (downloaded):**
     - `exp1a_results/` (Exp1A evaluation CSVs)
     - `exp1b_corrected_results/` (Exp1B evaluation CSVs)
     - `checkpoints/` (downloaded best models)
     - `finetune.log` (Exp2 training log)

3. **Generated Graphs and Visualizations (9.3)**
   - `snow_type_graphs/figure1_kge_by_snow_type_boxplot_CORRECTED.png`
     - Shows Exp1A with 3 snow types (Montane, Maritime, Ephemeral)
     - Shows Exp1B with 2 types only (Montane, Maritime - no ephemeral)
     - 2×2 subplot format (KGE + MSE for each experiment)
   
   - `snow_type_graphs/figure2_kge_vs_elevation_scatter_CORRECTED.png`
     - KGE vs elevation scatter plot
     - Combined Exp1A + Exp1B deep snow HUCs
   
   - `snow_type_graphs/figure3_model_comparison_by_snow_type_CORRECTED.png`
     - 4 models compared across snow types
     - Median KGE bars for each model
   
   - `snow_type_graphs/summary_by_snow_type_CORRECTED.csv`
     - Statistical summary (median, mean, std KGE)

   - `comparison_graphs/figure1_exp1a_exp1b_comparison_bars.png`
     - Side-by-side bar comparison of all variations
   
   - **Results reports:**
     - `COMPLETE_EXP1A_EXP1B_RESULTS_TABLE.md`
     - `EXPERIMENT_2_RESULTS_REPORT.md`

4. **What's Available in MLflow (9.4)**
   - **Access instructions:**
     - AWS Console → SageMaker → MLflow → "dawgsML" → Open MLflow UI
     - URL: `https://t-izowcn0gky2o.us-west-2.experiments.sagemaker.aws`
   
   - **Experiment organization:**
     - 16 experiments total (8 for Exp1A + 8 for Exp1B)
     - Each experiment has 30 runs (epochs)
     - Exp2 has 1 experiment with 10 runs
   
   - **Metrics logged per epoch:**
     - Training: `train_loss`, `train_time_seconds`, `epoch`
     - Validation: `val_kge`, `val_mse`, `val_r2`, `val_mae`, `val_pearson_r`, `val_bias`, `val_variability`
     - Best tracking: `best_val_kge`, `best_epoch`
   
   - **Parameters logged:**
     - Model architecture (hidden_size, num_layers, dropout, lookback, batch_size)
     - Training config (learning_rate, n_epochs, optimizer, loss_fn)
     - Data config (n_train_hucs, n_val_hucs, var_list)
     - Infrastructure (device, gpu_name, instance_type)
   
   - **Artifacts stored:**
     - Model checkpoints in `s3://dawgs-mlflow-artifacts/`
     - Training curves (PNG)
     - Evaluation results (CSV)
   
   - **Programmatic access examples:**
     - List all experiments
     - Get best run from experiment
     - Download model checkpoint
     - Compare multiple models
     - Full Python code examples provided

5. **Results Summary Quick Reference (9.5)**
   - **Exp1B best results table:**
     - Wind 3e-4: Val KGE 0.8104 (best validation)
     - Humidity 3e-4: Test B KGE 0.7783 (best spatial transfer)
     - Base 3e-4: Val KGE 0.8084 (used for Exp2)
   
   - **Exp1A best results table:**
     - Humidity 3e-4: Test B KGE 0.7783 (best overall)
     - All models lower than Exp1B due to ephemeral HUCs
   
   - **Exp2 results table:**
     - Pre-trained: Test B KGE 0.7603
     - Fine-tuned: Test B KGE 0.6110 (19.6% worse)
     - Conclusion: Fine-tuning degraded performance

6. **File Download Commands (9.6)**
   - SCP commands to download checkpoints
   - Commands for results CSVs
   - Commands for training logs
   - JupyterLab file browser instructions

---

## Why This Section Was Added

**Problem:** The original guide (v1.0) focused heavily on setup and avoiding mistakes but didn't document:
- Where training scripts are located
- Where results are saved during and after training
- What graphs were generated and where
- How to access and query MLflow data
- Quick reference for final results

**Solution:** Section 9 provides a complete reference for:
- Future researchers needing to reproduce experiments
- Understanding what files exist and where
- Accessing historical data from MLflow
- Locating graphs for thesis/papers
- Quick lookup of best performing models

---

## Key Improvements

### 1. Complete File Path Documentation
Every script, checkpoint, result file, and graph now has a documented location with full paths.

### 2. MLflow Deep Dive
Comprehensive documentation of:
- How to access the UI
- What's stored (experiments, runs, metrics, parameters, artifacts)
- How to query programmatically with Python examples
- Where artifacts are stored (S3 bucket)

### 3. Results at a Glance
Quick reference tables showing:
- Best models from each experiment
- Performance metrics comparison
- Why Exp2 failed (overfitting on small dataset)

### 4. Graph Documentation
Each graph now documented with:
- File location
- Purpose
- What it shows
- Format (subplot structure)
- Key insights (e.g., Exp1B has no ephemeral)

### 5. Download Instructions
Practical commands for retrieving:
- Checkpoints from SageMaker
- Results CSVs
- Training logs

---

## How to Use This Update

### For Current Work:
1. Reference Section 9.2 to know where to save/find results
2. Use Section 9.3 to locate graphs for thesis
3. Use Section 9.4 to query MLflow for metrics
4. Use Section 9.5 for quick results lookup

### For Future Students:
1. Read Section 9.1 to understand which scripts to run
2. Check Section 9.2 to know where results will be saved
3. Use Section 9.4 to access historical training data
4. Reference Section 9.5 to compare with your results

### For Thesis Writing:
1. Section 9.3 lists all graphs and their purposes
2. Section 9.5 provides results tables
3. Use MLflow UI (9.4) to get training curves
4. Reference file locations (9.2) for reproducibility section

---

## File Statistics

**Updated Document:**
- Original length: 1,385 lines
- New length: ~1,900 lines
- Added: ~515 lines (37% increase)
- New sections: 6 major subsections

**Coverage:**
- 3 experiments documented (Exp1A, Exp1B, Exp2)
- 16 MLflow experiments cataloged
- 240+ checkpoint files per experiment
- 4 main graph types documented
- 10+ Python code examples for MLflow queries

---

## What's Still Missing (For Future Updates)

1. **Section 10: Advanced Evaluation Techniques**
   - Per-HUC performance analysis
   - Error analysis by basin characteristics
   - Comparison with baseline models

2. **Section 11: Using Results for Thesis**
   - How to cite models
   - Which graphs to use in which chapters
   - Statistical significance testing

3. **Appendix: Complete File Tree**
   - Full directory structure with all files
   - File size information
   - Last modified dates

4. **Troubleshooting: Results Issues**
   - What to do if results look wrong
   - How to validate downloaded checkpoints
   - Recovering from corrupted files

---

## Version History

**v1.0 (September 28, 2026):**
- Initial comprehensive guide
- Sections 1-8: Setup, mistakes, parameters, execution, troubleshooting
- Focus: How to run experiments correctly

**v2.0 (September 30, 2026):**
- Added Section 9: File locations and results reference
- Complete documentation of all scripts, results, graphs
- MLflow deep dive with Python examples
- Results summary tables
- Download commands

---

## Quick Reference: Where to Find What

| What You Need | Where to Look |
|---------------|---------------|
| Training scripts | Section 9.1 |
| Where checkpoints are saved | Section 9.2 (SageMaker paths) |
| Downloaded results location | Section 9.2 (Local paths) |
| Graph locations | Section 9.3 |
| How to access MLflow | Section 9.4 (Access instructions) |
| What's in MLflow | Section 9.4 (Metrics, parameters, artifacts) |
| How to query MLflow | Section 9.4 (Python examples) |
| Best model results | Section 9.5 (Results tables) |
| Download commands | Section 9.6 |
| Setup and parameters | Sections 1-8 (original guide) |

---

**Updated by:** Simran Dhankar  
**Date:** September 30, 2026  
**Status:** Complete and validated  
**Next Update:** After additional experiments or thesis completion
