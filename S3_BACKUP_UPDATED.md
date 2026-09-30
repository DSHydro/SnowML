# S3 Backup Script - Updated Version

**Date:** September 30, 2026  
**Update:** Added Exp1A results and Exp1B results directories

---

## What Changed

### Before (Original Script):
✓ Checkpoints (480 files)  
✓ Exp2 fine-tuning results (12 files)  
❌ Exp1A results (missing)  
❌ Exp1B results (missing)  
❌ Evaluation results (missing)  
❌ Training logs (missing)

**Total:** ~492 files, ~53 GB

---

### After (Updated Script):
✅ Checkpoints (480 files)  
✅ Exp1A results (16 files)  
✅ Exp1B results (10 files)  
✅ Exp2 fine-tuning results (12 files)  
✅ Evaluation results (8 files)  
✅ Training logs (4 files)

**Total:** ~526 files, ~53.2 GB

---

## Complete Backup Structure

```
s3://snowml-model-ready/checkpoints/simran_thesis/20260930/
│
├── checkpoints/                    # 480 files, ~52 GB
│   ├── Exp1A_Base_1e-3_epoch1.pth
│   ├── Exp1A_Base_1e-3_epoch2.pth
│   ├── ... (30 epochs × 8 variations)
│   ├── Exp1B_Base_1e-3_epoch1.pth
│   └── ... (30 epochs × 8 variations)
│
├── exp1a_results/                  # 16 files, ~45 MB
│   ├── Base_1e-3_Test_A_metrics.csv
│   ├── Base_1e-3_Test_B_metrics.csv
│   ├── Base_3e-4_Test_A_metrics.csv
│   ├── Base_3e-4_Test_B_metrics.csv
│   ├── Wind_3e-4_Test_A_metrics.csv
│   ├── Wind_3e-4_Test_B_metrics.csv
│   ├── Humidity_3e-4_Test_A_metrics.csv
│   ├── Humidity_3e-4_Test_B_metrics.csv
│   ├── Srad_3e-4_Test_A_metrics.csv
│   ├── Srad_3e-4_Test_B_metrics.csv
│   └── all_variations_summary.csv
│
├── exp1b_corrected_results/        # 10 files, ~29 MB
│   ├── Base_3e-4_Test_A_metrics (1).csv
│   ├── Base_3e-4_Test_B_metrics (1).csv
│   ├── Wind_3e-4_Test_A_metrics (2).csv
│   ├── Wind_3e-4_Test_B_metrics (2).csv
│   ├── Humidity_3e-4_Test_A_metrics (2).csv
│   ├── Humidity_3e-4_Test_B_metrics (2).csv
│   ├── Srad_3e-4_Test_A_metrics (1).csv
│   ├── Srad_3e-4_Test_B_metrics (1).csv
│   ├── all_4_variations_complete_20260926_033829.csv
│   └── summary_4_variations_20260926_033829.csv
│
├── exp2_finetune_results/          # 12 files, ~1.1 GB
│   ├── FineTune_Base_epoch1.pth
│   ├── FineTune_Base_epoch2.pth
│   ├── ... (10 epochs)
│   ├── finetune_summary.json
│   └── finetune_splits.json
│
├── evaluation_results/             # 8 files, ~15 MB
│   └── ... (additional evaluation CSVs if present)
│
└── logs/                           # 4 files, ~11 MB
    ├── exp1a_training.log
    ├── exp1b_training.log
    ├── finetune.log
    └── training_summary.csv
```

---

## How to Use Updated Script

**On SageMaker Studio:**

```bash
# 1. Upload the updated backup_checkpoints_to_s3.py to SageMaker

# 2. Run the script
cd /home/sagemaker-user
python backup_checkpoints_to_s3.py
```

**Expected Progress:**
```
[1/8] Initializing S3 client...
[2/8] Backing up main checkpoints...        (480 files)
[3/8] Backing up Exp1A results...           (16 files)
[4/8] Backing up Exp1B results...           (10 files)
[5/8] Backing up Exp2 fine-tuning results... (12 files)
[6/8] Backing up evaluation results...      (8 files)
[7/8] Backing up training logs...           (4 files)
[8/8] Summary

Total: 526 files successfully uploaded!
```

---

## What Gets Backed Up From Each Directory

### 1. Checkpoints (`/home/sagemaker-user/checkpoints/`)
- **What:** All .pth model files
- **Files:** 480 (240 Exp1A + 240 Exp1B)
- **Size:** ~52 GB
- **Pattern:** `*.pth`, `*.json`

### 2. Exp1A Results (`/home/sagemaker-user/exp1a_results/`)
- **What:** Evaluation metrics CSV files
- **Files:** ~16 CSVs
- **Size:** ~45 MB
- **Contains:** Test A and Test B metrics for all variations
- **Pattern:** `*.csv`, `*.json`

### 3. Exp1B Results (`/home/sagemaker-user/exp1b_corrected_results/`)
- **What:** Evaluation metrics CSV files
- **Files:** ~10 CSVs
- **Size:** ~29 MB
- **Contains:** Test A and Test B metrics for 4 variations at 3e-4 LR
- **Pattern:** `*.csv`, `*.json`

### 4. Exp2 Results (`/home/sagemaker-user/exp2_finetune_results/`)
- **What:** Fine-tuned checkpoints and metadata
- **Files:** 10 .pth + 2 .json = 12 files
- **Size:** ~1.1 GB
- **Contains:** All 10 fine-tuning epochs + summaries
- **Pattern:** `*.pth`, `*.json`

### 5. Evaluation Results (`/home/sagemaker-user/evaluation_results/`)
- **What:** Additional evaluation CSVs (if exists)
- **Files:** Varies
- **Size:** ~15 MB
- **Pattern:** `*.csv`, `*.json`

### 6. Training Logs (`/home/sagemaker-user/`)
- **What:** Full training output logs
- **Files:** 4 logs
- **Size:** ~11 MB
- **Contains:**
  - `exp1a_training.log` (Exp1A full training output)
  - `exp1b_training.log` (Exp1B full training output)
  - `finetune.log` (Exp2 fine-tuning output)
  - `training_summary.csv` (Epoch-by-epoch metrics)

---

## Verification Commands

### Check all files uploaded:
```bash
aws s3 ls s3://snowml-model-ready/checkpoints/simran_thesis/20260930/ --recursive --human-readable | head -50
```

### Count by directory:
```bash
# Count checkpoints
aws s3 ls s3://snowml-model-ready/checkpoints/simran_thesis/20260930/checkpoints/ | wc -l

# Count Exp1A results
aws s3 ls s3://snowml-model-ready/checkpoints/simran_thesis/20260930/exp1a_results/ | wc -l

# Count Exp1B results
aws s3 ls s3://snowml-model-ready/checkpoints/simran_thesis/20260930/exp1b_corrected_results/ | wc -l

# Count Exp2 results
aws s3 ls s3://snowml-model-ready/checkpoints/simran_thesis/20260930/exp2_finetune_results/ | wc -l

# Count logs
aws s3 ls s3://snowml-model-ready/checkpoints/simran_thesis/20260930/logs/ | wc -l
```

### Total size:
```bash
aws s3 ls s3://snowml-model-ready/checkpoints/simran_thesis/20260930/ --recursive --summarize --human-readable
```

---

## Download Everything Later

### Download complete backup:
```bash
# Download all files maintaining structure
aws s3 sync s3://snowml-model-ready/checkpoints/simran_thesis/20260930/ ./complete_backup/
```

### Download specific directories:
```bash
# Only Exp1A results
aws s3 sync s3://snowml-model-ready/checkpoints/simran_thesis/20260930/exp1a_results/ ./exp1a_results/

# Only Exp1B results
aws s3 sync s3://snowml-model-ready/checkpoints/simran_thesis/20260930/exp1b_corrected_results/ ./exp1b_results/

# Only best checkpoints (e.g., epoch 14 for all models)
aws s3 sync s3://snowml-model-ready/checkpoints/simran_thesis/20260930/checkpoints/ ./ \
  --exclude "*" \
  --include "*epoch14.pth"

# Only training logs
aws s3 sync s3://snowml-model-ready/checkpoints/simran_thesis/20260930/logs/ ./logs/
```

---

## Why These Additions Matter

### Exp1A Results
- **Needed for:** Comparing Exp1A vs Exp1B performance
- **Contains:** 533 HUCs (with ephemeral) results
- **Critical for thesis:** Shows impact of including ephemeral basins

### Exp1B Results
- **Needed for:** Deep snow only analysis
- **Contains:** 270 HUCs (no ephemeral) results
- **Critical for thesis:** Best performing models

### Training Logs
- **Needed for:** Debugging, reproducibility
- **Contains:** Complete training output with all errors/warnings
- **Critical for:** Future students to understand what happened

---

## Summary of Changes

| Component | Before | After | Change |
|-----------|--------|-------|--------|
| **Directories backed up** | 2 | 6 | +4 |
| **Total files** | 492 | 526 | +34 |
| **Total size** | ~53 GB | ~53.2 GB | +0.2 GB |
| **Script steps** | [1-4] | [1-8] | +4 steps |
| **File types** | .pth, .json | .pth, .csv, .json, .log | +2 types |

---

## Cost Impact

**Additional storage:** 0.2 GB (results + logs)  
**Additional cost:** $0.0046/month (negligible)  
**Total monthly cost:** Still ~$1.24/month

**The additional files are tiny compared to checkpoints!**

---

**Updated by:** Simran Dhankar  
**Date:** September 30, 2026  
**Status:** Complete - All experiment results now included
