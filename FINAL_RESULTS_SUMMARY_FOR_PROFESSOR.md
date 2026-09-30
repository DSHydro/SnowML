# SnowML Thesis Results Summary

**Student:** Simran Dhankar  
**Date:** September 30, 2026  
**Topic:** LSTM Models for Snow Water Equivalent (SWE) Prediction

---

## Executive Summary

This document summarizes the results from three completed experiments:
- **Experiment 1A:** Multi-HUC training on 533 HUCs (all basins including ephemeral)
- **Experiment 1B:** Multi-HUC training on 270 deep snow HUCs (no ephemeral)
- **Experiment 2:** Fine-tuning Exp1B Base model on Yakima/Naches region

All training was conducted on AWS SageMaker Studio (ml.g4dn.xlarge instance) with complete logging to MLflow tracking server.

---

## Table of Contents

1. [Experiment Results Summary](#1-experiment-results-summary)
2. [Where to Find Results](#2-where-to-find-results)
3. [Experiment 2 Findings](#3-experiment-2-findings-fine-tuning)
4. [Generated Visualizations](#4-generated-visualizations)
5. [Accessing MLflow Data](#5-accessing-mlflow-data)
6. [Recommendations](#6-recommendations)

---

## 1. Experiment Results Summary

### 1.1 Experiment 1B Results (270 Deep Snow HUCs)

**Training Configuration:**
- HUCs: 162 train / 54 validation / 55 test (270 total)
- Epochs: 30 per variation
- Learning rates tested: 0.001, 0.0003
- Feature sets: Base, Wind, Humidity, Srad

**Best Performing Models:**

| Model | Learning Rate | Best Epoch | Val KGE | Test A KGE | Test B KGE |
|-------|---------------|------------|---------|------------|------------|
| **Wind** | 0.0003 | 27 | **0.8104** | 0.8127 | 0.7703 |
| **Humidity** | 0.0003 | 24 | 0.8097 | 0.8066 | **0.7783** |
| **Base** | 0.0003 | 14 | 0.8084 | 0.8107 | 0.7603 |
| Srad | 0.0003 | 28 | 0.7902 | 0.7952 | 0.7544 |

**Key Findings:**
- **Wind 3e-4:** Highest validation KGE (0.8104) - best overall model
- **Humidity 3e-4:** Best spatial transferability to Test B (0.7783)
- **Base 3e-4:** Strong baseline performance (0.8084) - used for Exp2
- All models with learning rate 0.0003 outperformed 0.001

**Training Duration:** ~6-7 days per complete run (8 variations × 30 epochs)

---

### 1.2 Experiment 1A Results (533 HUCs with Ephemeral)

**Training Configuration:**
- HUCs: 320 train / 106 validation / 107 test (533 total)
- Same parameters as Exp1B
- Includes ephemeral basins (shallow/intermittent snow)

**Best Performing Models:**

| Model | Learning Rate | Val KGE | Test A KGE | Test B KGE |
|-------|---------------|---------|------------|------------|
| **Humidity** | 0.0003 | 0.7621 | 0.7845 | **0.7783** |
| Wind | 0.0003 | 0.7587 | 0.7732 | 0.7721 |
| Base | 0.0003 | 0.7512 | 0.7644 | 0.7455 |
| Srad | 0.0003 | 0.7189 | 0.7421 | 0.7288 |

**Key Findings:**
- Including ephemeral basins reduced validation KGE by ~0.05 (0.76 vs 0.81)
- Humidity 3e-4 achieved same Test B KGE (0.7783) as in Exp1B
- Demonstrates trade-off between broader coverage vs specialized performance

---

### 1.3 Comparison: Exp1A vs Exp1B

| Metric | Exp1A (533 HUCs) | Exp1B (270 HUCs) | Difference |
|--------|------------------|------------------|------------|
| **Best Val KGE** | 0.7621 (Humidity) | 0.8104 (Wind) | +0.0483 |
| **Best Test A KGE** | 0.7845 (Humidity) | 0.8127 (Wind) | +0.0282 |
| **Best Test B KGE** | 0.7783 (Humidity) | 0.7783 (Humidity) | 0.0000 |

**Interpretation:**
- Deep snow only models (Exp1B) achieve better validation performance
- Both experiments show similar spatial transferability to Yakima/Naches
- Exp1B more suitable when target region is deep snow
- Exp1A more suitable for broad regional coverage

---

## 2. Where to Find Results

### 2.1 Local Files (On Desktop)

```
/Users/simran/Desktop/SnowML/
│
├── exp1a_results/
│   ├── Base_3e-4_Test_A_metrics.csv
│   ├── Base_3e-4_Test_B_metrics.csv
│   ├── Wind_3e-4_Test_A_metrics.csv
│   ├── Wind_3e-4_Test_B_metrics.csv
│   ├── Humidity_3e-4_Test_A_metrics.csv
│   ├── Humidity_3e-4_Test_B_metrics.csv
│   ├── Srad_3e-4_Test_A_metrics.csv
│   └── Srad_3e-4_Test_B_metrics.csv
│
├── exp1b_corrected_results/
│   ├── Base_3e-4_Test_A_metrics.csv
│   ├── Base_3e-4_Test_B_metrics.csv
│   ├── Wind_3e-4_Test_A_metrics.csv
│   ├── Wind_3e-4_Test_B_metrics.csv
│   ├── Humidity_3e-4_Test_A_metrics.csv
│   ├── Humidity_3e-4_Test_B_metrics.csv
│   ├── Srad_3e-4_Test_A_metrics.csv
│   ├── Srad_3e-4_Test_B_metrics.csv
│   └── summary_4_variations_20260926_033829.csv
│
├── snow_type_graphs/
│   ├── figure1_kge_by_snow_type_boxplot_CORRECTED.png
│   ├── figure2_kge_vs_elevation_scatter_CORRECTED.png
│   ├── figure3_model_comparison_by_snow_type_CORRECTED.png
│   └── summary_by_snow_type_CORRECTED.csv
│
├── COMPLETE_EXP1A_EXP1B_RESULTS_TABLE.md
├── EXPERIMENT_2_RESULTS_REPORT.md
└── COMPLETE_KNOWLEDGE_TRANSFER_GUIDE.md
```

**Key Files:**
- `COMPLETE_EXP1A_EXP1B_RESULTS_TABLE.md` - Complete results table for all 16 variations
- `EXPERIMENT_2_RESULTS_REPORT.md` - Detailed Exp2 analysis
- CSV files contain per-HUC metrics (KGE, MSE, R², MAE)

---

### 2.2 AWS SageMaker Studio

**Location:** `/home/sagemaker-user/`

**Checkpoints:**
- Path: `/home/sagemaker-user/checkpoints/`
- Files: 480 checkpoint files (240 Exp1A + 240 Exp1B)
- Format: `Exp1B_Wind_3e-4_epoch27.pth` (best Wind model)
- Size: ~52 GB total

**Training Logs:**
- `exp1a_training.log` - Complete Exp1A training output
- `exp1b_training.log` - Complete Exp1B training output
- `finetune.log` - Exp2 fine-tuning output

**Results:**
- `/home/sagemaker-user/exp1a_results/`
- `/home/sagemaker-user/exp1b_corrected_results/`
- `/home/sagemaker-user/exp2_finetune_results/`

---

### 2.3 AWS S3 Backup

**Base Path:**
```
s3://snowml-model-ready/checkpoints/simran_thesis/20260930/
```

**Contents:**
- Selected checkpoints (Exp1A 3e-4 variations, epochs 0-14)
- All Exp1A results CSVs
- All Exp1B results CSVs
- Exp2 fine-tuning checkpoints and summaries

**Access:**
```bash
# List all backed up files
aws s3 ls s3://snowml-model-ready/checkpoints/simran_thesis/20260930/ --recursive

# Download Exp1B results
aws s3 sync s3://snowml-model-ready/checkpoints/simran_thesis/20260930/exp1b_corrected_results/ ./
```

---

### 2.4 MLflow Tracking Server

**Server:** dawgsML  
**ARN:** `arn:aws:sagemaker:us-west-2:677276086662:mlflow-tracking-server/dawgsML`  
**Web UI:** Available through AWS Console → SageMaker → MLflow

**Experiments Tracked:**
- 16 experiments total (8 Exp1A + 8 Exp1B)
- Each experiment: 30 runs (epochs 1-30)
- 1 Exp2 experiment: 10 runs (epochs 1-10)

**Metrics Logged Per Epoch:**
- Validation: KGE, MSE, R², MAE, Pearson r, bias, variability
- Training: loss, time per epoch
- Best model tracking: best_val_kge, best_epoch

**Artifacts Stored:**
- Model checkpoints (.pth files)
- Training curves (PNG images)
- Evaluation results (CSV files)

**How to Access:**
1. AWS Console → SageMaker → MLflow
2. Click "dawgsML" server
3. Click "Open MLflow UI"
4. Browse experiments: `Exp1B_Wind_3e-4_20260826`

---

## 3. Experiment 2 Findings (Fine-Tuning)

### 3.1 Experiment Design

**Objective:** Improve performance on Yakima/Naches region (Test Set B) through fine-tuning.

**Approach:**
- Pre-trained model: Exp1B Base epoch 14 (Val KGE: 0.8084)
- Target region: 81 Yakima/Naches HUCs (Test Set B)
- Split: 56 train / 25 validation (70/30 split)
- Learning rate: 0.0001 (reduced from 0.0003)
- Dropout: 0.2 (reduced from 0.5)
- Epochs: 10 (vs 30 for pre-training)

**Rationale:** Lower learning rate and dropout to prevent overfitting on small dataset.

---

### 3.2 Results

**Performance Comparison:**

| Metric | Pre-trained (Exp1B Base) | Fine-tuned (Exp2) | Change |
|--------|--------------------------|-------------------|--------|
| **Test B KGE** | **0.7603** | **0.6110** | **-0.1493 (-19.6%)** ❌ |
| Test B MSE | 0.0034 | 0.0066 | +0.0032 (+94.1%) |
| Best Val KGE | 0.8084 | 0.9738 | +0.1654 (+20.4%) |

**Training Progress:**

| Epoch | Validation KGE | Status |
|-------|---------------|---------|
| 1 | 0.5606 | Initial drop |
| **2** | **0.9738** | **Best validation** |
| 3 | 0.9116 | Declining |
| 4 | 0.7501 | Unstable |
| 5-6 | 0.91+ | High variance |
| 7-10 | 0.62-0.93 | Large fluctuations |

---

### 3.3 Analysis: Why Fine-Tuning Failed

**Primary Issue: Overfitting**
- Validation KGE reached 0.9738 (very high, suggesting overfitting)
- Final Test B KGE only 0.6110 (19.6% worse than pre-trained)
- Model optimized for 25 validation HUCs but failed on all 81 HUCs

**Contributing Factors:**

1. **Small Dataset**
   - Only 56 training HUCs (vs 162 in Exp1B pre-training)
   - Only 25 validation HUCs (insufficient to represent full region)
   - High variance in performance across epochs

2. **Catastrophic Forgetting**
   - Model trained on 270 diverse deep snow HUCs
   - Fine-tuning on single region caused model to "forget" broader patterns
   - Lost ability to generalize across different basin characteristics

3. **Distribution Mismatch**
   - 70/30 split created non-representative subsets
   - Validation set didn't capture full Yakima/Naches variability
   - Model overfit to specific validation HUCs

4. **Regional Complexity**
   - Yakima/Naches may have unique characteristics
   - 81 HUCs insufficient to capture regional patterns
   - Fine-tuning couldn't improve on pre-trained knowledge

**Training Instability:**
Large fluctuations in validation KGE (0.56 → 0.97 → 0.62 → 0.93) indicate:
- High variance due to small dataset
- Insufficient regularization
- Poor train/validation split

---

### 3.4 Comparison with Transfer Learning Literature

**Typical Requirements for Successful Fine-Tuning:**
- Target dataset size: >1000 samples (we had 56 HUCs)
- Similar domain to pre-training (single region vs multi-region reduced diversity)
- Strong regularization (dropout 0.2 may have been insufficient)

**Our Results Align With:**
- Small dataset overfitting is common in transfer learning
- Regional specialization can hurt generalization
- Pre-trained model often better than fine-tuned for small target datasets

---

### 3.5 Recommendation

**Use Pre-trained Exp1B Base Model (0.7603 KGE) for Yakima/Naches predictions.**

**Reasons:**
- 19.6% better performance than fine-tuned version
- Better generalization to unseen HUCs
- Lower risk of overfitting
- Proven performance on 270 diverse deep snow basins

**Alternative Approaches If Further Improvement Needed:**
1. **Ensemble Methods:** Combine predictions from multiple models (Wind, Humidity, Base)
2. **Regional Data Augmentation:** Add similar climate regions to expand training set
3. **K-fold Cross-Validation:** Better validation strategy for small datasets
4. **Layer Freezing:** Freeze encoder, only fine-tune final prediction layer
5. **More Conservative Hyperparameters:**
   - Learning rate: 5e-5 (half of current)
   - Dropout: 0.4 (stronger regularization)
   - Epochs: 5 (reduce overfitting risk)

---

## 4. Generated Visualizations

### 4.1 Snow Type Analysis Graphs

**Location:** `snow_type_graphs/`

#### Figure 1: KGE by Snow Type (Boxplot)
- **File:** `figure1_kge_by_snow_type_boxplot_CORRECTED.png`
- **Shows:** KGE distribution by snow type (Montane Forest, Maritime, Ephemeral)
- **Layout:** 2×2 subplot
  - Top row: Exp1A (3 types including ephemeral)
  - Bottom row: Exp1B (2 types only - no ephemeral)
- **Key Insight:** Exp1B excludes ephemeral basins, achieving higher KGE on deep snow types

#### Figure 2: KGE vs Elevation
- **File:** `figure2_kge_vs_elevation_scatter_CORRECTED.png`
- **Shows:** Test KGE plotted against mean elevation for each HUC
- **Types:** Montane Forest (purple) and Maritime (blue)
- **Key Insight:** No significant correlation between elevation and KGE in multi-HUC models

#### Figure 3: Model Comparison by Snow Type
- **File:** `figure3_model_comparison_by_snow_type_CORRECTED.png`
- **Shows:** Median KGE for each model variation across snow types
- **Layout:** 2×2 subplot (Exp1A/1B × Test A/B)
- **Key Insight:** Humidity and Wind models perform best across all snow types

**Statistical Summary:**
- **File:** `summary_by_snow_type_CORRECTED.csv`
- **Contents:** Median, mean, and standard deviation KGE by model and snow type

---

### 4.2 Performance by Snow Type (Exp1B)

| Model | Montane Forest (Median KGE) | Maritime (Median KGE) |
|-------|----------------------------|----------------------|
| Base | 0.853 | 0.822 |
| Humidity | 0.841 | 0.837 |
| Wind | 0.840 | 0.849 |
| Srad | 0.781 | 0.814 |

**Observations:**
- All models achieve KGE > 0.78 on deep snow types
- Wind performs best on Maritime basins (0.849)
- Base performs best on Montane Forest (0.853)
- Consistent performance across both deep snow types

---

## 5. Accessing MLflow Data

### 5.1 Web Interface

**Steps:**
1. Log into AWS Console
2. Navigate to: SageMaker → MLflow
3. Click "dawgsML" server
4. Click "Open MLflow UI"
5. Browse experiments by name (e.g., `Exp1B_Wind_3e-4_20260826`)

**What You Can View:**
- All 30 epochs for each experiment
- Training curves (KGE vs epoch, loss vs epoch)
- Comparison across different models
- Best performing epoch for each variation
- Download checkpoints and evaluation results

---

### 5.2 Experiment Organization

**Naming Convention:**
```
Exp1A_Base_1e-3_20260901
Exp1A_Base_3e-4_20260901
Exp1A_Wind_1e-3_20260901
Exp1A_Wind_3e-4_20260901
...
Exp1B_Base_1e-3_20260826
Exp1B_Base_3e-4_20260826
Exp1B_Wind_1e-3_20260826
Exp1B_Wind_3e-4_20260826
...
Exp2_FineTune_Base_20260929
```

**Each Experiment Contains:**
- 30 runs (Exp1A/1B) or 10 runs (Exp2)
- Each run = one training epoch with metrics

---

### 5.3 Key Metrics to Examine

**Validation Metrics (Primary):**
- `val_kge` - Kling-Gupta Efficiency (higher is better, range -∞ to 1)
- `val_mse` - Mean Squared Error (lower is better)
- `val_r2` - R² score (higher is better, range 0 to 1)
- `val_mae` - Mean Absolute Error (lower is better)

**Training Metrics:**
- `train_loss` - MSE training loss
- `train_time_seconds` - Time per epoch

**Best Model Tracking:**
- `best_val_kge` - Highest validation KGE achieved so far
- `best_epoch` - Epoch number with best KGE

---

### 5.4 Finding Best Models in MLflow

**To find best performing model:**
1. Navigate to experiment (e.g., `Exp1B_Wind_3e-4_20260826`)
2. Click "Runs" tab
3. Sort by `val_kge` descending
4. Top run shows best epoch (epoch 27 for Wind)
5. Download checkpoint from "Artifacts" tab

**Best Models Summary:**
- **Exp1B Wind 3e-4:** Run at epoch 27, Val KGE 0.8104
- **Exp1B Humidity 3e-4:** Run at epoch 24, Val KGE 0.8097
- **Exp1B Base 3e-4:** Run at epoch 14, Val KGE 0.8084
- **Exp1A Humidity 3e-4:** Best performer for Exp1A, Val KGE 0.7621

---

## 6. Recommendations

### 6.1 Model Selection for Deployment

**For Yakima/Naches Region (Test Set B):**
- **Recommended:** Exp1B Humidity 3e-4 (Test B KGE: 0.7783)
- **Alternative:** Exp1B Wind 3e-4 (Test B KGE: 0.7703)
- **Do NOT use:** Exp2 fine-tuned model (Test B KGE: 0.6110)

**For General Deep Snow Basins:**
- **Recommended:** Exp1B Wind 3e-4 (Val KGE: 0.8104)
- **Alternative:** Exp1B Humidity 3e-4 (Val KGE: 0.8097)

**For Broad Regional Coverage (Including Ephemeral):**
- **Recommended:** Exp1A Humidity 3e-4 (Val KGE: 0.7621)
- Trade-off: Lower performance but broader applicability

---

### 6.2 For Thesis Documentation

**Results to Highlight:**
1. **Multi-HUC Training Success:** Achieved 0.81 validation KGE on 270 HUCs
2. **Spatial Transferability:** 0.78 KGE on unseen Yakima/Naches region
3. **Feature Importance:** Wind and Humidity additions improved performance
4. **Snow Type Analysis:** Deep snow models outperform when ephemeral excluded
5. **Transfer Learning Limits:** Fine-tuning degraded performance due to small dataset

**Key Tables/Figures:**
- Complete results table (16 variations)
- Training curves showing convergence
- Snow type analysis boxplots
- Exp1A vs Exp1B comparison
- Exp2 failure analysis

---

### 6.3 Future Work Suggestions

**Model Improvements:**
1. Test ensemble methods (combine Wind + Humidity predictions)
2. Incorporate snow type as model feature
3. Test attention mechanisms for spatial patterns
4. Explore GNN (Graph Neural Networks) for watershed relationships

**Data Augmentation:**
1. Include more Yakima-similar basins for better fine-tuning
2. Test on additional ungauged regions (Test Set C)
3. Temporal cross-validation (train on 1983-2000, test on 2001-2022)

**Transfer Learning:**
1. Layer-wise fine-tuning (freeze lower layers)
2. Domain adaptation techniques
3. Meta-learning for few-shot adaptation
4. Curriculum learning (gradual fine-tuning)

---

## 7. Summary

### 7.1 Key Achievements

✅ **Successful Multi-HUC Training**
- 270 HUCs trained simultaneously
- Achieved 0.81 validation KGE (Wind 3e-4)
- 30 epochs completed in 6-7 days

✅ **Strong Spatial Generalization**
- 0.78 KGE on unseen Yakima/Naches region
- Comparable to previous single-HUC models
- Demonstrates true spatial transferability

✅ **Comprehensive Evaluation**
- 16 model variations tested
- 4 feature sets × 2 learning rates × 2 experiments
- Complete results logged to MLflow

✅ **Knowledge Transfer**
- Detailed documentation created
- All results backed up to S3
- Future students can reproduce experiments

---

### 7.2 Key Findings

📊 **Best Models:**
- Exp1B Wind 3e-4: Best overall (0.8104 Val KGE)
- Exp1B Humidity 3e-4: Best Test B (0.7783 KGE)

📊 **Feature Importance:**
- Wind and Humidity > Base > Srad
- Learning rate 0.0003 > 0.001

📊 **Dataset Composition:**
- Deep snow only (Exp1B) > All basins (Exp1A)
- +0.05 KGE improvement when ephemeral excluded

❌ **Fine-Tuning Failure:**
- Small dataset (81 HUCs) caused overfitting
- 19.6% performance degradation vs pre-trained
- Recommendation: Use pre-trained model

---

### 7.3 Files for Review

**Essential Documents:**
1. `COMPLETE_EXP1A_EXP1B_RESULTS_TABLE.md` - All 16 variations compared
2. `EXPERIMENT_2_RESULTS_REPORT.md` - Detailed Exp2 analysis
3. `snow_type_graphs/figure1_kge_by_snow_type_boxplot_CORRECTED.png` - Main visualization

**Results Data:**
- `exp1b_corrected_results/summary_4_variations_20260926_033829.csv` - Quick summary
- `exp1b_corrected_results/Humidity_3e-4_Test_B_metrics.csv` - Best Test B performance

**For Questions/Details:**
- `COMPLETE_KNOWLEDGE_TRANSFER_GUIDE.md` - Full technical documentation

---

## Contact Information

**Student:** Simran Dhankar  
**AWS Account:** 677276086662  
**MLflow Server:** dawgsML (ARN: arn:aws:sagemaker:us-west-2:677276086662:mlflow-tracking-server/dawgsML)  
**S3 Backup:** s3://snowml-model-ready/checkpoints/simran_thesis/20260930/

**For Questions:**
- MLflow access: Contact AWS administrator
- Results interpretation: Refer to EXPERIMENT_2_RESULTS_REPORT.md
- Technical details: Refer to COMPLETE_KNOWLEDGE_TRANSFER_GUIDE.md

---

**Document Version:** 1.0  
**Date:** September 30, 2026  
**Purpose:** Professor review of thesis results and findings
