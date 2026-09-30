# Experiment 3: Individual HUC-12 Models - Complete Analysis

**Date:** September 14, 2026  
**What It Is:** 533-534 individual LSTM models, one per HUC-12 basin  
**Purpose:** Baseline comparison for multi-HUC training (Exp1A/1B)

---

## 📍 Where Is Experiment 3?

### In the Repository (Main Branch):
**Location:** `notebooks/Ex2_VarianceByHuc/`

**Key Files:**
- ✅ `single_all_metrics_w_snow_types_and_elev.csv` - **534 HUCs with results**
- ✅ `LSTM_By_Huc.ipynb` - Analysis notebook
- ✅ `charts/` - 39 visualization files
- ✅ `run_id_data/` - MLflow run IDs for each model

**Results Summary:**
- **Total HUCs:** 534 individual models
- **Average Test KGE:** 0.7666
- **Method:** Each HUC trained separately with 0.67/0.33 train/test split
- **Features:** Temperature + Precipitation (baseline)

---

## 🔍 In MLflow Server:

### "Single All" Experiments Found:
- 6 experiments in MLflow (mix of FINISHED, RUNNING, FAILED)
- Range: 24 to 490 HUCs per run
- These appear to be **partial/test runs**, not the complete 534

**Best MLflow Single Run:**
- Status: FINISHED
- HUCs: 144 HUCs (incomplete)
- This is NOT the full experiment

---

## 📊 Experiment 3 Performance Breakdown

### Overall Statistics (from CSV file):
```
Total Models: 534 HUCs
Average Test KGE: 0.7666
Method: Individual training per HUC
Train/Test Split: 0.67 / 0.33 (temporal)
Features: Temperature + Precipitation + Elevation
```

### Performance by Snow Type:

| Snow Type | Average Test KGE | Count | Quality |
|-----------|-----------------|-------|---------|
| **Montane Forest** | **0.8847** | 187 | Excellent |
| **Maritime** | **0.8136** | 155 | Excellent |
| **Boreal Forest** | 0.7997 | 1 | Good |
| **Prairie** | 0.7861 | 11 | Good |
| **Ephemeral** | 0.6022 | 180 | Fair |

### Key Statistics:
- **Min KGE:** -1.16 (some models failed badly)
- **Max KGE:** 0.97 (excellent performance possible)
- **Median KGE:** 0.8466
- **Mean KGE:** 0.7666

### Insights:
✅ **Deep snow (Montane + Maritime) performs MUCH better:** 0.85-0.88 KGE  
⚠️ **Ephemeral snow struggles:** Only 0.60 KGE (poor)  
⚠️ **High variance:** Some models fail completely (KGE < 0)

---

## 🎯 What Your Professor Wants vs What Exists

### Professor's Document Says:

**Experiment 3: Individually Trained HUC-12 Models (Single-Basin Baseline)**
> "In Experiment 3, a separate LSTM model is trained independently for each HUC12 subwatershed in the study domain. We examined the 533 HUC12 sub-basins... Each individual HUC12 unit was trained on data from that unit only and tested on later years of data using a 0.67 train/test split."

### What Actually Exists:

| Aspect | Professor's Doc | Repository Reality | Match? |
|--------|----------------|-------------------|--------|
| **Name** | Experiment 3 | Ex2_VarianceByHuc | ⚠️ Different naming |
| **Number of HUCs** | 533 | **534** | ✅ Close (534 is correct) |
| **Method** | Individual training | Individual training | ✅ Matches |
| **Train/Test Split** | 0.67 / 0.33 | 0.67 / 0.33 | ✅ Matches |
| **Results Available** | Needed | **✅ COMPLETE** | ✅ Ready to use! |
| **In MLflow** | Should be there | Partial only | ⚠️ Not fully logged |

---

## ✅ GOOD NEWS: Experiment 3 Is COMPLETE!

**You CAN use these results!**

### What You Have:
✅ **Complete results file:** 534 HUCs with Test KGE, MSE, etc.  
✅ **Performance by snow type:** Montane, Maritime, Ephemeral, etc.  
✅ **Visualizations:** 39 charts showing spatial patterns  
✅ **Analysis notebooks:** Full analysis pipeline documented

### What's in MLflow:
⚠️ **Incomplete logging:** Only partial runs (24-490 HUCs)  
⚠️ **Not all 534 models logged:** Some missing from MLflow  
✅ **But results CSV has everything:** The file has all 534 HUCs!

---

## 📊 Comparison: Exp3 vs Your Exp1B

### Individual Models (Exp3) vs Multi-HUC (Your Exp1B):

| Metric | Exp3 (Individual) | Your Exp1B Wind_3e-4 | Your Exp1B Humidity_3e-4 | Winner |
|--------|-------------------|---------------------|------------------------|--------|
| **Average KGE** | 0.7666 | - | - | - |
| **Median KGE** | 0.8466 | - | - | - |
| **Test A KGE** | N/A (different split) | **0.8375** | **0.8119** | Similar |
| **Deep snow KGE** | 0.85-0.88 | ~0.84 | ~0.81 | **Exp3 slightly better** |
| **Ephemeral KGE** | 0.60 | N/A (excluded) | N/A (excluded) | - |
| **Training time** | 533× separate | 1× joint | 1× joint | **Exp1B MUCH faster** |
| **New basins** | Cannot transfer | **Can transfer** | **Can transfer** | **Exp1B wins** |

### Key Insights:

1. **Individual models (Exp3) perform well on SEEN data:**
   - Deep snow: KGE 0.85-0.88 (excellent)
   - Each model perfectly fits its own basin

2. **Multi-HUC models (Exp1B) enable transfer learning:**
   - Test A: 0.81-0.84 (excellent, slightly lower)
   - Test B: 0.76-0.79 (good spatial generalization)
   - **Can predict on NEW basins** (Exp3 cannot!)

3. **Trade-off:**
   - Exp3: Better on known basins, cannot generalize
   - Exp1B: Slightly lower on known basins, CAN generalize
   - **For operational use, Exp1B is better** (works on unseen areas)

---

## 🔬 Which Experiment Is This in the Repo?

### Naming Confusion Resolved:

**In Main Branch Notebooks:**
- `Ex1_MoreData/` - Data experiments (not main experiments)
- `Ex2_VarianceByHuc/` - **THIS IS EXPERIMENT 3** (Individual HUC models)
- `Ex3_MultiHucTraining/` - **THIS IS OLD EXPERIMENT 1B** (Multi-HUC deep snow)
- `Ex4_MixedLoss/` - Loss function experiments
- `Ex5_DataIntegration/` - Data integration experiments
- `Ex6_Lag90/` - Lag window experiments

**Professor's New Naming (Updated_experiments_MLSnow.docx):**
- **Experiment 1A** - Multi-HUC with ephemeral (535 HUCs) - NOT DONE
- **Experiment 1B** - Multi-HUC deep snow only (270 HUCs) - YOU DID THIS ✅
- **Experiment 2** - Fine-tuning workflow - NOT DONE
- **Experiment 3** - Individual HUC models (533 HUCs) - **= Ex2_VarianceByHuc/** ✅

---

## 📋 Summary for Your Professor

### Experiment 3 Status: ✅ COMPLETE AND USABLE

**Location:** `notebooks/Ex2_VarianceByHuc/single_all_metrics_w_snow_types_and_elev.csv`

**Results:**
- 534 individual HUC-12 models trained and evaluated
- Average Test KGE: 0.7666 (median: 0.8466)
- Deep snow performance: 0.85-0.88 (excellent)
- Ephemeral snow performance: 0.60 (fair)

**Comparison:**
- Individual models (Exp3): Better on own basins (0.85-0.88)
- Your Multi-HUC (Exp1B): Slightly lower (0.81-0.84) BUT can generalize to new regions
- **Trade-off is expected:** Multi-HUC sacrifices some accuracy for transferability

**MLflow Status:**
- ⚠️ Not all 534 models fully logged in MLflow
- ✅ But complete results file exists in repository
- Can use the CSV file for all comparisons

---

## ✅ What This Means for Your Work

### You Have All Three Baselines Needed:

1. ✅ **Exp3 (Individual models):** 534 HUCs, avg KGE 0.77, in Ex2_VarianceByHuc/
2. ✅ **Exp1B (Your multi-HUC):** 270 HUCs (162/54/55), Test KGE 0.81-0.84
3. ❌ **Exp1A (Multi-HUC + ephemeral):** 535 HUCs - STILL NEED TO RUN

### You Can Compare:
- Exp1B (your multi-HUC deep) vs Exp3 (individual deep) ✅
- Show trade-off: Exp3 better per-basin, Exp1B better for new basins ✅

### You Still Need:
- Exp1A (multi-HUC with ephemeral) to answer: "Does including ephemeral help or hurt?"
- Exp2 (fine-tuning) to answer: "Can we improve on pre-trained models?"

---

**Conclusion:** Experiment 3 is complete and ready to use from `notebooks/Ex2_VarianceByHuc/`. You have 2 out of 4 experiments done (Exp1B and Exp3). You still need Exp1A and Exp2.
