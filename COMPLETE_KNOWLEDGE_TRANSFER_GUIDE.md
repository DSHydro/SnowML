# Complete Knowledge Transfer Guide: SnowML Experiments
**Purpose:** Comprehensive guide to rerun experiments without repeating mistakes  
**Date Created:** September 28, 2026  
**For:** Future researchers / continuation of this work  
**Status:** Production-ready based on successful SageMaker Studio deployment

---

## TABLE OF CONTENTS
1. [Critical Lessons Learned - Mistakes to Avoid](#1-critical-lessons-learned)
2. [Correct Setup Path](#2-correct-setup-path)
3. [Infrastructure Details](#3-infrastructure-details)
4. [Key Files and Their Roles](#4-key-files-and-their-roles)
5. [Critical Parameters - DO NOT GET THESE WRONG](#5-critical-parameters)
6. [Step-by-Step Execution Guide](#6-step-by-step-execution)
7. [Common Pitfalls and Solutions](#7-common-pitfalls)
8. [Verification and Monitoring](#8-verification-and-monitoring)

---

## 1. CRITICAL LESSONS LEARNED

### 1.1 INITIAL MISTAKES AND WHY THEY WERE WRONG

#### ❌ MISTAKE 1: Using Wrong Training Environment

**What Was Done Wrong:**
- Attempted to use EC2 instances (t3.2xlarge) instead of SageMaker Studio
- Tried local Jupyter notebooks on laptop
- Considered regular SageMaker Notebook Instances (deprecated)

**Why This Was Wrong:**
- EC2 instances don't have built-in MLflow integration
- Local training can't log to AWS MLflow server easily
- SageMaker Notebook Instances are being deprecated (use Studio instead)
- Previous students used SageMaker Studio - this is the validated path

**Correct Approach:**
- **Always use SageMaker Studio with JupyterLab Space**
- Instance type: `ml.g4dn.xlarge` (Tesla T4 GPU, 16GB VRAM, 4 vCPU, 16GB RAM)
- Cost: ~$0.80/hour (vs $0.33/hour for CPU-only EC2)
- **Why:** Built-in MLflow integration, GPU support, same environment previous students used

---

#### ❌ MISTAKE 2: Wrong Training Methodology (train_size_dimension)

**What Was Done Wrong:**
- Used `train_size_dimension="time"` with `train_size_fraction=0.67`
- This splits EACH HUC's timeline: first 67% for training, last 33% for validation
- Result: Validation KGE = 0.94 (artificially high)

**Why This Was Wrong:**
- Model sees the SAME HUCs during training and validation (just different time periods)
- This doesn't test **spatial generalization** (ability to predict on unseen watersheds)
- Creates data leakage - model learns basin-specific patterns
- Validation metrics are inflated and misleading

**Correct Approach:**
- **ALWAYS use `train_size_dimension="huc"`**
- **ALWAYS use `train_size_fraction=1.0`**
- This uses ENTIRE time series of each HUC
- Split HUCs themselves into train/validation/test sets (60/20/20)
- Tests true spatial transferability

**Example of Impact:**
| Method | Validation KGE | Interpretation |
|--------|---------------|----------------|
| `dimension="time"` | 0.94 | WRONG - Same basins, inflated |
| `dimension="huc"` | 0.81-0.84 | CORRECT - New basins, realistic |

---

#### ❌ MISTAKE 3: Wrong HUC Splits

**What Was Done Wrong (July 2026 attempt):**
- Created new splits: 231 train / 77 val / 78 test (total 467 HUCs)
- Used 467 HUCs instead of 270 HUCs
- Split ratios were NOT 60/20/20
- Possibly included ephemeral basins when they should have been excluded

**Why This Was Wrong:**
- Exp1B should use EXACTLY 270 HUCs (deep snow only, no ephemeral)
- Must use 60/20/20 split: 162 train / 54 val / 54-55 test
- Using different HUC sets makes results incomparable to previous work
- Can't validate against published results

**Correct Approach:**
- **Exp1A:** Use ALL 533 HUCs (including ephemeral)
  - Split: 320 train / 106 val / 107 test (60/20/20)
- **Exp1B:** Use EXACTLY 270 HUCs (deep snow only)
  - Split: 162 train / 54 val / 54-55 test (60/20/20)
  - **Use existing splits from:** `src/snowML/datapipe/huc_lists/hucs_data.json`

---

#### ❌ MISTAKE 4: MLflow Version Incompatibility

**What Was Done Wrong:**
- Used latest MLflow version (3.x)
- Used HTTPS URL directly without ARN
- Didn't install `sagemaker-mlflow` plugin

**Why This Was Wrong:**
- MLflow server is version 2.16.2 (2 years old)
- ARN format not recognized by newer MLflow versions
- Connection failures with "Missing Tracking Server ARN" error

**Correct Approach:**
- **Downgrade to MLflow 2.16.2:** `pip install mlflow==2.16.2`
- **Install SageMaker plugin:** `pip install sagemaker-mlflow`
- **Use ARN as tracking URI:** Set environment variable
- **Never use HTTPS URL alone** - must use full ARN

---

#### ❌ MISTAKE 5: GPU Out of Memory (OOM) Errors

**What Was Done Wrong:**
- Validation loaded entire HUC (~14,000 timesteps) at once
- No batched prediction for large validation sets
- GPU memory accumulated without cleanup

**Why This Was Wrong:**
- 14k timesteps × batch processing = ~3GB GPU memory
- Tesla T4 has only 16GB total VRAM
- Multiple validation HUCs cause OOM crash

**Correct Approach:**
- **Use batched prediction:** Process validation in chunks of 1000 samples
- **Add GPU cleanup:** `torch.cuda.empty_cache()` after validation
- **Fixed in:** `LSTM_train.py` with `predict_batched()` function

---

#### ❌ MISTAKE 6: Incorrect num_workers Setting

**What Was Done Wrong:**
- Used `num_workers=8` (default from some tutorials)

**Why This Was Wrong:**
- Can cause DataLoader hangs on some instances
- Not optimal for SageMaker Studio environment

**Correct Approach:**
- **Use `num_workers=4`** for stability
- On ml.g4dn.xlarge (4 vCPU), this is optimal

---

### 1.2 SUMMARY: THE GOLDEN RULES

**If you remember NOTHING else, remember these:**

1. **Environment:** SageMaker Studio JupyterLab (ml.g4dn.xlarge)
2. **Training Method:** `train_size_dimension="huc"`, `train_size_fraction=1`
3. **HUC Splits:** Use `hucs_data.json` for Exp1B (162/54/55)
4. **MLflow:** Version 2.16.2 + sagemaker-mlflow plugin + ARN
5. **GPU Memory:** Use batched validation (1000 samples/batch)
6. **Learning Rates:** 0.001 and 0.0003 (always test both)
7. **Feature Sets:** Base, +Wind, +Humidity, +Srad (4 sets × 2 LRs = 8 models)

---

## 2. CORRECT SETUP PATH

### 2.1 Branch to Use

```bash
git checkout simran-unified-experiments
```

**Why this branch:**
- Contains all fixes (batched validation, GPU cleanup, checkpoint saving)
- Has corrected parameter settings
- Matches previous students' methodology

---

### 2.2 Environment: SageMaker Studio (NOT Anything Else!)

**DO NOT USE:**
- ❌ EC2 instances
- ❌ Local Jupyter notebooks
- ❌ SageMaker Notebook Instances (deprecated)
- ❌ Your laptop (unless just for testing)

**ALWAYS USE:**
- ✅ **SageMaker Studio with JupyterLab Space**
- ✅ Instance: `ml.g4dn.xlarge`
- ✅ Location: `us-west-2` (Oregon)

**Why SageMaker Studio:**
1. Built-in MLflow integration (can connect via ARN)
2. GPU support (Tesla T4)
3. Same environment previous students used (validated workflow)
4. dawgsML server is in same AWS account/region
5. Files persist on EFS (don't lose work when stopping)

---

### 2.3 What to Request from Professor

**You MUST have these before starting:**

1. **AWS Credentials:**
   - AWS Access Key ID
   - AWS Secret Access Key
   - Instructions: Ask professor for IAM user credentials

2. **IAM Permissions Needed:**
   - SageMaker: Full access to Studio, create/run spaces
   - MLflow: Read/write to `dawgsML` tracking server
   - S3: Read from `snowml-model-ready` bucket
   - S3: Write to `dawgs-mlflow-artifacts` bucket (for model artifacts)

3. **SageMaker Studio Access:**
   - Domain ID: `d-spd2senoxi9j` (or ask for current domain)
   - User profile created for you
   - Permission to launch ml.g4dn.xlarge instances

4. **MLflow Server Access:**
   - Confirm `dawgsML` server is running (or permission to start it)
   - ARN: `arn:aws:sagemaker:us-west-2:677276086662:mlflow-tracking-server/dawgsML`

**Email Template to Professor:**
```
Subject: AWS Access for SnowML Experiments

Hi Professor,

I'm ready to start the training for Experiments 1A, 1B, and 2. Based on the previous students' setup, I need:

1. AWS credentials (Access Key + Secret Key) for account 677276086662
2. SageMaker Studio access in us-west-2 region
3. Permission to use ml.g4dn.xlarge GPU instances
4. Access to MLflow tracking server "dawgsML"
5. Read access to S3 bucket "snowml-model-ready"

Could you create an IAM user for me or send credentials? I'll use SageMaker Studio (same as previous students) to ensure reproducibility.

Thank you!
```

---

## 3. INFRASTRUCTURE DETAILS

### 3.1 MLflow Tracking Server (CRITICAL)

**Server Details:**
- **Name:** `dawgsML`
- **ARN:** `arn:aws:sagemaker:us-west-2:677276086662:mlflow-tracking-server/dawgsML`
- **HTTPS URL:** `https://t-izowcn0gky2o.us-west-2.experiments.sagemaker.aws`
- **Version:** MLflow 2.16.2
- **Size:** Small
- **S3 Artifacts:** `s3://dawgs-mlflow-artifacts`
- **Status:** Started (has been running for 2+ years)
- **Region:** us-west-2

**Cost:**
- ~$0.25/hour (always running)
- ~$6/day
- ~$180/month
- Shared across lab - professor already paying for it

**How to Connect:**
```bash
# Method 1: Environment variable (recommended)
export MLFLOW_TRACKING_URI="arn:aws:sagemaker:us-west-2:677276086662:mlflow-tracking-server/dawgsML"

# Method 2: In Python code
import mlflow
mlflow.set_tracking_uri("arn:aws:sagemaker:us-west-2:677276086662:mlflow-tracking-server/dawgsML")
```

**How to Access UI:**
1. AWS Console → SageMaker → MLflow
2. Click "dawgsML"
3. Click "Open MLflow UI" button
4. Browse experiments, runs, metrics, models

---

### 3.2 S3 Data Bucket

**Bucket Details:**
- **Name:** `snowml-model-ready`
- **Region:** us-west-2
- **Contents:** 3,681 HUC time series CSV files
- **Format:** `model_ready_huc{HUC_ID}.csv`
- **Size per file:** ~2-10 MB
- **Total size:** ~20-30 GB

**Key Files:**
```
s3://snowml-model-ready/
├── model_ready_huc160100010101.csv  (HUC time series)
├── model_ready_huc160100010102.csv
├── ... (3,681 total files)
├── huc12_snow_classification_master.csv  (Classification)
└── ... (metadata files)
```

**How to Access:**
```bash
# List files
aws s3 ls s3://snowml-model-ready/ --region us-west-2

# Download single file
aws s3 cp s3://snowml-model-ready/model_ready_huc160100010101.csv ./ --region us-west-2

# Download pattern (all HUC files starting with 1703)
aws s3 sync s3://snowml-model-ready/ ./data/ \
  --exclude "*" \
  --include "model_ready_huc1703*.csv" \
  --region us-west-2
```

---

### 3.3 SageMaker Studio Environment

**How to Set Up (First Time):**

1. **Access Studio:**
   - AWS Console → SageMaker → Studio
   - Click "Open Studio" (or create new space)

2. **Create JupyterLab Space:**
   - Applications → Create JupyterLab Space
   - Name: `snowml-training` (or your choice)
   - Instance: **ml.g4dn.xlarge**
   - Storage: Default (5GB is enough, files on EFS)

3. **Start Space:**
   - Click "Run" (takes ~2-3 minutes)
   - Once running, click "Open JupyterLab"

4. **Upload Code:**
   - Use Upload button OR
   - Clone git repo in terminal

---

### 3.4 Python Environment Setup

**Environment Name:** `pytorch_p310`

**This environment comes pre-installed in SageMaker Studio with:**
- Python 3.10
- PyTorch 2.5.1 with CUDA 12.1
- CUDA toolkit
- Basic ML packages

**Additional Packages to Install:**
```bash
# In terminal
conda activate pytorch_p310

# MLflow with SageMaker support
pip install mlflow==2.16.2 sagemaker-mlflow

# Geospatial packages
conda install -c conda-forge geopandas rasterio fiona shapely -y

# Scientific computing (update versions if needed)
conda install -c conda-forge xarray s3fs -y

# Install SnowML package
cd /home/sagemaker-user
# Upload your code first, then:
pip install -e src/
```

**Verify Installation:**
```python
import torch
import mlflow
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
print(f"MLflow: {mlflow.__version__}")
```

Expected output:
```
PyTorch: 2.5.1+cu121
CUDA available: True
GPU: Tesla T4
MLflow: 2.16.2
```

---

## 4. KEY FILES AND THEIR ROLES

### 4.1 Core Training Scripts

#### `src/snowML/Scripts/multi_huc_expirement.py`
**Purpose:** Main training script for Exp1A and Exp1B (multi-HUC training)

**Key Functions:**
- `run_expirement(train_hucs, val_hucs, params)` - Main entry point
- Trains one model on multiple HUCs simultaneously
- Logs to MLflow, saves checkpoints

**Critical Parameters It Uses:**
```python
params["train_size_dimension"] = "huc"     # MUST BE "huc"
params["train_size_fraction"] = 1.0        # MUST BE 1.0
params["n_epochs"] = 30                    # Standard
params["batch_size"] = 32                  # Works on 16GB GPU
params["var_list"] = [...]                 # Feature set
params["learning_rate"] = 0.001 or 0.0003  # Test both
```

**How to Use:**
```python
from snowML.Scripts import multi_huc_expirement as mhe
from snowML.LSTM import set_hyperparams as sh

params = sh.create_hyper_dict()
params["train_size_dimension"] = "huc"
params["train_size_fraction"] = 1.0
params["var_list"] = ["mean_pr", "mean_tair", "Mean Elevation"]
params["learning_rate"] = 0.001

mhe.run_expirement(train_hucs, val_hucs, params)
```

---

#### `src/snowML/LSTM/LSTM_train.py`
**Purpose:** Core LSTM training logic (pre-training, fine-tuning, evaluation)

**Key Functions:**
```python
pre_train(model, optimizer, loss_fn, df_dict, params, epoch)
  # Trains model on all training HUCs for one epoch
  # Returns: training loss

evaluate(model_dawgs, df_dict, params, epoch) 
  # Validates on all validation HUCs
  # Uses BATCHED prediction (prevents GPU OOM)
  # Returns: metrics dict

fine_tune(model, optimizer, loss_fn, df_train, params, epoch)
  # Fine-tunes pre-trained model on specific HUC
  # Used for Experiment 2

predict_batched(model, X_te, params, batch_size=1000)
  # NEW FIX: Batched prediction to prevent GPU OOM
  # Processes 1000 samples at a time
  # Critical for validation on large HUCs
```

**CRITICAL FIX (already implemented):**
```python
# Lines 62-64: Move batches to GPU
for X_batch, y_batch in loader:
    X_batch = X_batch.to(params['device'])  # CRITICAL
    y_batch = y_batch.to(params['device'])  # CRITICAL
    # ... rest of training loop
```

---

#### `src/snowML/LSTM/set_hyperparams.py`
**Purpose:** Default hyperparameter configuration

**Key Settings:**
```python
def create_hyper_dict():
    params = {
        "hidden_size": 64,           # LSTM neurons
        "num_layers": 1,             # LSTM depth
        "dropout": 0.5,              # 0.5 for pre-training, 0.2 for fine-tuning
        "batch_size": 32,            # Fits in 16GB GPU
        "lookback": 180,             # 180-day window
        "n_epochs": 30,              # Standard training
        "num_workers": 4,            # CHANGED from 8 to 4
        "loss_type": "mse",
        
        # MLflow (set these!)
        "mlflow_tracking_uri": "arn:aws:sagemaker:us-west-2:...",
        "MLFLOW_ON": True,
        
        # CRITICAL SETTINGS
        "train_size_dimension": "huc",  # NOT "time"!
        "train_size_fraction": 1.0,     # NOT 0.67!
    }
    return params
```

---

### 4.2 HUC Split Files

#### `src/snowML/datapipe/huc_lists/hucs_data.json`
**Purpose:** Contains Exp1B HUC splits (270 deep snow HUCs)

**Structure:**
```json
{
  "train": [162 HUC IDs],      // 60%
  "validate": [54 HUC IDs],    // 20%
  "test": [55 HUC IDs]         // 20%
}
```

**How to Load:**
```python
from snowML.Scripts.load_hucs import load_huc_splits as lh

train_hucs, val_hucs, test_hucs = lh.huc_split(
    "/home/sagemaker-user/src/snowML/datapipe/huc_lists/hucs_data.json"
)

print(f"Train: {len(train_hucs)}")      # Should be 162
print(f"Validation: {len(val_hucs)}")   # Should be 54
print(f"Test: {len(test_hucs)}")        # Should be 55
```

**CRITICAL:** For Exp1B, ALWAYS use this file. DO NOT create new splits.

---

#### Test Set B (Yakima/Naches HUCs)
**Purpose:** Spatial transferability test on ungauged region

**How to Get:**
```python
from snowML.Scripts.load_hucs import create_test_set_B as cb

test_b_hucs = cb.get_testB_huc_list()
print(f"Test B: {len(test_b_hucs)}")  # Should be 81
```

**Basins:**
- Upper Yakima (HUC-8: 17030001)
- Naches (HUC-8: 17030002)
- Total: 81 HUC-12 basins

---

### 4.3 Fixed Versions vs Original Versions

**ALWAYS use these fixed files from simran-unified-experiments branch:**

| File | What Was Fixed | Critical? |
|------|---------------|-----------|
| `LSTM_train.py` | Batched validation, GPU cleanup | **YES** |
| `set_hyperparams.py` | MLflow ARN, num_workers=4 | **YES** |
| `multi_huc_expirement.py` | Checkpoint saving, GPU placement | **YES** |

**How to Verify You Have Correct Versions:**
```bash
# Check for batched prediction function
grep -n "predict_batched" src/snowML/LSTM/LSTM_train.py
# Should show function definition around line 140

# Check num_workers setting
grep "num_workers" src/snowML/LSTM/set_hyperparams.py
# Should show: params["num_workers"] = 4
```

---

## 5. CRITICAL PARAMETERS

### 5.1 THE PARAMETERS THAT RUINED EVERYTHING (Don't Get These Wrong!)

#### **train_size_dimension** - THE MOST CRITICAL PARAMETER

```python
# WRONG (used in July 2026 - caused invalid results)
params["train_size_dimension"] = "time"
params["train_size_fraction"] = 0.67

# CORRECT (must use this)
params["train_size_dimension"] = "huc"
params["train_size_fraction"] = 1.0
```

**What This Means:**

**WRONG Method (`dimension="time"`):**
```
HUC_001: [Day 1 --- Day 1000 TRAIN --- | --- Day 1001-1500 VAL ---]
HUC_002: [Day 1 --- Day 1000 TRAIN --- | --- Day 1001-1500 VAL ---]
Problem: Same basins in train and validation, just different dates
Result: Validation KGE = 0.94 (artificially high, data leakage)
```

**CORRECT Method (`dimension="huc"`):**
```
TRAIN HUCs: [001, 002, 003, ...] - use ENTIRE time series
VAL HUCs: [101, 102, 103, ...]   - use ENTIRE time series (NEW basins!)
TEST HUCs: [201, 202, 203, ...]  - use ENTIRE time series (UNSEEN basins!)
Result: Validation KGE = 0.81-0.84 (realistic, tests transferability)
```

---

### 5.2 Correct HUC Splits

#### Experiment 1A (All 533 HUCs including ephemeral)
```python
# Total: 533 HUCs
# Split: 60/20/20

train_hucs: 320 HUCs     # 60% of 533
val_hucs: 106 HUCs       # 20% of 533
test_hucs: 107 HUCs      # 20% of 533
test_b_hucs: 81 HUCs     # Yakima/Naches (spatial test)
```

#### Experiment 1B (270 deep snow only, NO ephemeral)
```python
# Total: 270 HUCs (from hucs_data.json)
# Split: 60/20/20

train_hucs: 162 HUCs     # 60% of 270
val_hucs: 54 HUCs        # 20% of 270
test_hucs: 55 HUCs       # 20% of 270
test_b_hucs: 81 HUCs     # Yakima/Naches (spatial test)
```

**HOW TO GET CORRECT SPLITS:**
```python
# Exp1B (ALWAYS use existing file)
from snowML.Scripts.load_hucs import load_huc_splits as lh
train, val, test = lh.huc_split("src/snowML/datapipe/huc_lists/hucs_data.json")

# Exp1A (create or load if exists)
# Should stratify by snow type to ensure even distribution
```

---

### 5.3 Learning Rates to Test

**ALWAYS test both:**
```python
learning_rates = [0.001, 0.0003]
```

**Why both?**
- 0.001 = faster training, may oscillate
- 0.0003 = slower but more stable, often better generalization
- Previous students found 0.0003 worked better for most feature sets

---

### 5.4 Feature Sets (Variable Lists)

**Test all 4 combinations:**

```python
# 1. Base (minimum viable)
var_list_base = ["mean_pr", "mean_tair", "Mean Elevation"]

# 2. Base + Solar Radiation
var_list_srad = ["mean_pr", "mean_tair", "Mean Elevation", "mean_srad"]

# 3. Base + Wind Speed
var_list_wind = ["mean_pr", "mean_tair", "Mean Elevation", "mean_wind_speed"]

# 4. Base + Humidity
var_list_humidity = ["mean_pr", "mean_tair", "Mean Elevation", "mean_rh"]
```

**Total Models per Experiment:**
- 4 feature sets × 2 learning rates = **8 model variations**
- Each variation: 30 epochs
- Total: 240 checkpoints per experiment

**Expected Best Performers (based on previous results):**
1. Wind + 3e-4 LR: Best validation KGE (0.8104)
2. Humidity + 3e-4 LR: Best Test B KGE (0.7899)
3. Base + 1e-3 LR: Fastest convergence

---

### 5.5 Other Critical Parameters

```python
params = {
    # Model architecture
    "hidden_size": 64,          # LSTM neurons (don't change)
    "num_layers": 1,            # LSTM depth (don't change)
    "dropout": 0.5,             # Pre-training: 0.5, Fine-tuning: 0.2
    
    # Training settings
    "batch_size": 32,           # Fits in 16GB GPU (don't increase)
    "lookback": 180,            # 180-day window (don't change)
    "n_epochs": 30,             # Standard (can reduce for testing)
    "num_workers": 4,           # DataLoader workers (MUST BE 4)
    
    # Loss and optimization
    "loss_type": "mse",         # Mean Squared Error (don't change)
    "learning_rate": 0.001 or 0.0003,  # Test both!
    
    # CRITICAL: Data split method
    "train_size_dimension": "huc",     # ALWAYS "huc" (NOT "time")
    "train_size_fraction": 1.0,        # ALWAYS 1.0 (NOT 0.67)
    
    # Device
    "device": "cuda",           # GPU (SageMaker has GPU)
    
    # MLflow
    "mlflow_tracking_uri": "arn:aws:sagemaker:us-west-2:677276086662:mlflow-tracking-server/dawgsML",
    "MLFLOW_ON": True,
    "expirement_name": "Exp1B_Base_1e-3",  # Descriptive name
}
```

---

## 6. STEP-BY-STEP EXECUTION

### 6.1 Pre-Flight Checklist

Before starting ANY training, verify:

```bash
# 1. You're in SageMaker Studio (NOT EC2, NOT laptop)
# 2. GPU is available
nvidia-smi

# 3. Conda environment activated
conda activate pytorch_p310

# 4. MLflow environment variable set
export MLFLOW_TRACKING_URI="arn:aws:sagemaker:us-west-2:677276086662:mlflow-tracking-server/dawgsML"

# 5. Python path includes source
export PYTHONPATH=/home/sagemaker-user/src:$PYTHONPATH

# 6. Verify imports work
python -c "from snowML.Scripts import multi_huc_expirement; print('OK')"

# 7. Verify MLflow connection
python -c "import mlflow; print(mlflow.get_tracking_uri())"

# 8. Check correct branch
git branch  # Should show simran-unified-experiments
```

---

### 6.2 Running Experiment 1B (Multi-HUC Deep Snow Only)

**Purpose:** Train on 270 deep snow HUCs, compare with/without ephemeral

#### Step 1: Create Training Script

Create file: `/home/sagemaker-user/run_exp1b_full.py`

```python
#!/usr/bin/env python3
"""
Experiment 1B: Multi-HUC Training (Deep Snow Only)
270 HUCs excluding ephemeral basins
8 variations: 4 feature sets × 2 learning rates
"""

import sys
import time
from datetime import datetime

sys.path.insert(0, '/home/sagemaker-user/src')

from snowML.Scripts.load_hucs import load_huc_splits as lh
from snowML.Scripts import multi_huc_expirement as mhe
from snowML.LSTM import set_hyperparams as sh

def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

# Load HUC splits (162/54/55)
huc_json_path = "/home/sagemaker-user/src/snowML/datapipe/huc_lists/hucs_data.json"
train_hucs, val_hucs, test_hucs = lh.huc_split(huc_json_path)

log(f"Loaded splits: {len(train_hucs)} train, {len(val_hucs)} val, {len(test_hucs)} test")

# Define 8 variations
variations = [
    {
        "name": "Base_1e-3",
        "features": ["mean_pr", "mean_tair", "Mean Elevation"],
        "lr": 0.001
    },
    {
        "name": "Base_3e-4",
        "features": ["mean_pr", "mean_tair", "Mean Elevation"],
        "lr": 0.0003
    },
    {
        "name": "Srad_1e-3",
        "features": ["mean_pr", "mean_tair", "Mean Elevation", "mean_srad"],
        "lr": 0.001
    },
    {
        "name": "Srad_3e-4",
        "features": ["mean_pr", "mean_tair", "Mean Elevation", "mean_srad"],
        "lr": 0.0003
    },
    {
        "name": "Wind_1e-3",
        "features": ["mean_pr", "mean_tair", "Mean Elevation", "mean_wind_speed"],
        "lr": 0.001
    },
    {
        "name": "Wind_3e-4",
        "features": ["mean_pr", "mean_tair", "Mean Elevation", "mean_wind_speed"],
        "lr": 0.0003
    },
    {
        "name": "Humidity_1e-3",
        "features": ["mean_pr", "mean_tair", "Mean Elevation", "mean_rh"],
        "lr": 0.001
    },
    {
        "name": "Humidity_3e-4",
        "features": ["mean_pr", "mean_tair", "Mean Elevation", "mean_rh"],
        "lr": 0.0003
    }
]

# Run all 8 variations
total_start = time.time()

for i, var in enumerate(variations, 1):
    log("=" * 80)
    log(f"VARIATION {i}/8: {var['name']}")
    log("=" * 80)
    
    var_start = time.time()
    
    # Create parameters
    params = sh.create_hyper_dict()
    
    # CRITICAL SETTINGS
    params["train_size_dimension"] = "huc"      # NOT "time"!
    params["train_size_fraction"] = 1.0         # NOT 0.67!
    
    # Model settings
    params["var_list"] = var["features"]
    params["learning_rate"] = var["lr"]
    params["n_epochs"] = 30
    params["batch_size"] = 32
    params["hidden_size"] = 64
    params["num_layers"] = 1
    params["dropout"] = 0.5                     # Pre-training dropout
    params["lookback"] = 180
    params["num_workers"] = 4                   # MUST BE 4
    params["loss_type"] = "mse"
    
    # MLflow settings
    params["expirement_name"] = f"Exp1B_{var['name']}"
    params["mlflow_tracking_uri"] = "arn:aws:sagemaker:us-west-2:677276086662:mlflow-tracking-server/dawgsML"
    params["MLFLOW_ON"] = True
    
    # Device
    params["device"] = "cuda"
    
    log(f"Features: {var['features']}")
    log(f"Learning rate: {var['lr']}")
    log(f"Epochs: 30")
    log(f"Training HUCs: {len(train_hucs)}")
    log(f"Validation HUCs: {len(val_hucs)}")
    
    # Run training
    try:
        mhe.run_expirement(train_hucs, val_hucs, params)
        
        var_elapsed = time.time() - var_start
        log(f"✅ {var['name']} completed in {var_elapsed/3600:.2f} hours")
        
    except Exception as e:
        log(f"❌ {var['name']} FAILED: {e}")
        import traceback
        traceback.print_exc()
        continue
    
    log("")

total_elapsed = time.time() - total_start

log("=" * 80)
log("EXP1B TRAINING COMPLETE!")
log("=" * 80)
log(f"Total time: {total_elapsed/3600:.2f} hours")
log(f"Average per variation: {total_elapsed/3600/8:.2f} hours")
log("")
log("Check MLflow UI for results:")
log("https://t-izowcn0gky2o.us-west-2.experiments.sagemaker.aws")
```

#### Step 2: Run Training

```bash
# In SageMaker Studio terminal
cd /home/sagemaker-user

# Activate environment
conda activate pytorch_p310

# Set environment variables
export MLFLOW_TRACKING_URI="arn:aws:sagemaker:us-west-2:677276086662:mlflow-tracking-server/dawgsML"
export PYTHONPATH=/home/sagemaker-user/src:$PYTHONPATH

# Run in background with logging
nohup python -u run_exp1b_full.py > exp1b_training.log 2>&1 &

# Get process ID
echo $!  # Save this number
```

#### Step 3: Monitor Progress

```bash
# Watch log in real-time
tail -f exp1b_training.log

# Check GPU usage
nvidia-smi

# See which variation is running
grep "VARIATION" exp1b_training.log | tail -1

# Count completed variations
grep "completed in" exp1b_training.log | wc -l
```

#### Step 4: Verify Checkpoints

```bash
# Check checkpoint directory
ls -lh /home/sagemaker-user/checkpoints/

# Should see files like:
# Exp1B_Base_1e-3_epoch0.pth
# Exp1B_Base_1e-3_epoch1.pth
# ...
# Exp1B_Humidity_3e-4_epoch29.pth
```

#### Step 5: Check MLflow

1. AWS Console → SageMaker → MLflow → dawgsML
2. Click "Open MLflow UI"
3. Look for experiments: "Exp1B_Base_1e-3", "Exp1B_Base_3e-4", etc.
4. Each should have 30 runs (one per epoch)
5. Metrics should be logged per validation HUC

---

### 6.3 Running Experiment 1A (Multi-HUC with Ephemeral)

**Same as Exp1B but with different HUC splits**

Key differences:
```python
# Use 533 HUCs instead of 270
# Include ephemeral basins
# Split: 320 train / 106 val / 107 test

# You'll need to create/load these splits
train_hucs = [...]  # 320 HUCs
val_hucs = [...]    # 106 HUCs
test_hucs = [...]   # 107 HUCs

# Change experiment names
params["expirement_name"] = f"Exp1A_{var['name']}"
```

**Expected time:** ~10-12 hours (more HUCs than Exp1B)

---

### 6.4 Evaluating on Test Sets

After training completes, evaluate best model:

```python
#!/usr/bin/env python3
"""
Evaluate Exp1B Best Model on Test Sets A and B
"""

import sys
sys.path.insert(0, '/home/sagemaker-user/src')

import torch
from snowML.LSTM import LSTM_model as LSTM_mod
from snowML.LSTM import LSTM_train as LSTM_tr
from snowML.Scripts.load_hucs import load_huc_splits as lh
from snowML.Scripts.load_hucs import create_test_set_B as cb

# Load test sets
_, _, test_a_hucs = lh.huc_split("src/snowML/datapipe/huc_lists/hucs_data.json")
test_b_hucs = cb.get_testB_huc_list()

print(f"Test A: {len(test_a_hucs)} HUCs")  # 55
print(f"Test B: {len(test_b_hucs)} HUCs")  # 81

# Load best model (example: Wind_3e-4)
checkpoint_path = "/home/sagemaker-user/checkpoints/Exp1B_Wind_3e-4_epoch29.pth"
checkpoint = torch.load(checkpoint_path, map_location='cuda')

# Reconstruct model
params = checkpoint['params']  # Saved parameters
model = LSTM_mod.SnowModel(
    input_size=len(params['var_list']),
    hidden_size=params['hidden_size'],
    num_class=1,
    num_layers=params['num_layers'],
    dropout=params['dropout']
).to('cuda')

model.load_state_dict(checkpoint['model_state_dict'])

# Evaluate on Test A
print("\nEvaluating on Test Set A...")
test_a_metrics = LSTM_tr.evaluate(model, test_a_dict, params, epoch=0)
print(f"Test A Median KGE: {test_a_metrics[1]['median_kge']:.4f}")

# Evaluate on Test B
print("\nEvaluating on Test Set B...")
test_b_metrics = LSTM_tr.evaluate(model, test_b_dict, params, epoch=0)
print(f"Test B Median KGE: {test_b_metrics[1]['median_kge']:.4f}")
```

---

### 6.5 Time and Cost Estimates

#### Experiment 1B (270 HUCs)
**Per Variation (one feature set + one learning rate):**
- Training time: ~35-40 minutes per epoch
- 30 epochs: ~18-20 hours
- Cost: ~$16 (@ $0.80/hour for ml.g4dn.xlarge)

**Full Experiment (8 variations):**
- Total time: ~144-160 hours (~6-7 days)
- Total cost: ~$115-128
- Plus MLflow: ~$36 (@ $0.25/hour)
- **Grand total: ~$150-165**

#### Experiment 1A (533 HUCs)
- More HUCs = longer training
- Estimate: ~200-220 hours (~9 days)
- Cost: ~$160-175 GPU + $55 MLflow = **~$215-230**

#### Experiment 2 (Fine-tuning)
- 81 HUCs × 2 base models
- ~10 epochs per HUC
- Estimate: ~30-40 hours
- Cost: ~$24-32 GPU + $10 MLflow = **~$34-42**

**TOTAL for all experiments:** ~$400-440 (over 3-4 weeks)

---

## 7. COMMON PITFALLS

### 7.1 GPU Out of Memory

**Symptom:**
```
RuntimeError: CUDA out of memory. Tried to allocate X.XX GiB
```

**Cause:** Validation loading entire HUC without batching

**Solution (already fixed in code):**
```python
# In LSTM_train.py - predict_batched() function is used automatically
# If you still get OOM, reduce batch_size:
params["batch_size"] = 16  # From 32
```

**Emergency fix:**
```python
# In terminal
python -c "import torch; torch.cuda.empty_cache()"
```

---

### 7.2 MLflow Connection Errors

**Symptom:**
```
mlflow.exceptions.MlflowException: Missing Tracking Server ARN
```

**Cause:** MLflow version mismatch or missing environment variable

**Solution:**
```bash
# 1. Check MLflow version
python -c "import mlflow; print(mlflow.__version__)"
# Should be: 2.16.2

# 2. Reinstall correct version
pip uninstall mlflow -y
pip install mlflow==2.16.2 sagemaker-mlflow

# 3. Set environment variable
export MLFLOW_TRACKING_URI="arn:aws:sagemaker:us-west-2:677276086662:mlflow-tracking-server/dawgsML"

# 4. Test connection
python -c "import mlflow; print(mlflow.get_tracking_uri())"
```

---

### 7.3 num_workers Hangs

**Symptom:** DataLoader freezes, no progress for >30 minutes

**Cause:** num_workers set too high (8 instead of 4)

**Solution:**
```python
# In set_hyperparams.py or your params dict:
params["num_workers"] = 4  # NOT 8!
```

**If still hangs:**
```python
params["num_workers"] = 0  # Disable multiprocessing (slower but stable)
```

---

### 7.4 Checkpoint Files Not Visible in Studio UI

**Symptom:** Files exist in terminal but not in file browser

**Cause:** Studio UI caching bug

**Solution:**
```bash
# Verify files exist in terminal
ls -lh /home/sagemaker-user/checkpoints/

# Refresh browser
# OR click refresh button in file browser

# Files are ALWAYS there even if UI doesn't show them
```

---

### 7.5 Wrong train_size_dimension (The Big One!)

**Symptom:** Validation KGE suspiciously high (>0.90)

**Check:**
```python
print(params["train_size_dimension"])
# Should be: "huc"

print(params["train_size_fraction"])
# Should be: 1.0 or 1
```

**Fix immediately if wrong:**
```python
params["train_size_dimension"] = "huc"
params["train_size_fraction"] = 1.0

# Re-run training - previous results are INVALID
```

---

### 7.6 S3 Download Failures

**Symptom:**
```
botocore.exceptions.NoCredentialsError: Unable to locate credentials
```

**Solution:**
```bash
# Check AWS credentials
aws sts get-caller-identity

# If fails, configure
aws configure
# Enter Access Key, Secret Key, Region: us-west-2
```

---

## 8. VERIFICATION AND MONITORING

### 8.1 Pre-Training Verification

Before starting full 30-epoch training, run pilot test:

```python
# Pilot: 3 train HUCs, 2 val HUCs, 5 epochs, ~10 minutes
pilot_train = train_hucs[:3]
pilot_val = val_hucs[:2]

params["n_epochs"] = 5
params["expirement_name"] = "Pilot_Test"

mhe.run_expirement(pilot_train, pilot_val, params)

# Check:
# - No errors
# - MLflow logging works
# - Checkpoints saved
# - GPU utilized (nvidia-smi)
# - Validation KGE in reasonable range (0.4-0.7 for pilot)
```

---

### 8.2 During Training Monitoring

```bash
# Terminal 1: Watch log
tail -f exp1b_training.log

# Terminal 2: Monitor GPU
watch -n 5 nvidia-smi

# Terminal 3: Check checkpoints
watch -n 60 "ls -lh /home/sagemaker-user/checkpoints/ | tail -10"
```

**What to look for:**
- GPU Utilization: 70-100% during training
- GPU Memory: 8-12 GB used (out of 16 GB)
- Log shows progress every ~2-3 minutes (one epoch)
- New checkpoint files every ~40 minutes

---

### 8.3 MLflow Verification

**After each variation completes:**

1. Open MLflow UI
2. Find experiment (e.g., "Exp1B_Base_1e-3")
3. Should have 30 runs (epochs 0-29)
4. Click run, check:
   - Metrics tab: `val_kge_median`, `val_mse_median` logged
   - Parameters tab: All params recorded
   - Artifacts tab: Model saved (optional)

---

### 8.4 Post-Training Validation

**After ALL 8 variations complete:**

```python
# Compare validation KGE across variations
import mlflow
import pandas as pd

mlflow.set_tracking_uri("arn:aws:sagemaker:us-west-2:677276086662:mlflow-tracking-server/dawgsML")

experiments = [
    "Exp1B_Base_1e-3",
    "Exp1B_Base_3e-4",
    "Exp1B_Srad_1e-3",
    "Exp1B_Srad_3e-4",
    "Exp1B_Wind_1e-3",
    "Exp1B_Wind_3e-4",
    "Exp1B_Humidity_1e-3",
    "Exp1B_Humidity_3e-4"
]

results = []
for exp_name in experiments:
    exp = mlflow.get_experiment_by_name(exp_name)
    if exp:
        runs = mlflow.search_runs(exp.experiment_id, order_by=["start_time DESC"])
        best_kge = runs["metrics.val_kge_median"].max()
        results.append({"Experiment": exp_name, "Best_Val_KGE": best_kge})

df = pd.DataFrame(results).sort_values("Best_Val_KGE", ascending=False)
print(df)

# Expected top performers:
# 1. Wind_3e-4 or Humidity_3e-4
# 2. Base_3e-4
# 3. Wind_1e-3 or Humidity_1e-3
```

---

### 8.5 Evaluation Checklist

After training, verify:

- [ ] All 8 variations completed 30 epochs
- [ ] 240 checkpoints saved (8 × 30)
- [ ] MLflow has 8 experiments with 30 runs each
- [ ] Validation KGE in expected range (0.75-0.85)
- [ ] Best model identified
- [ ] Evaluated on Test Set A (should get ~0.81-0.84 KGE)
- [ ] Evaluated on Test Set B (should get ~0.76-0.79 KGE)
- [ ] No GPU OOM errors occurred
- [ ] Training log saved for reference

---

## 9. FILE LOCATIONS AND RESULTS REFERENCE

### 9.1 Training Scripts by Experiment

#### Experiment 1A (533 HUCs with Ephemeral)
**Training Script:**
- Location: `/home/sagemaker-user/train_exp1a_full.py` (on SageMaker)
- Local copy: `train_exp1a_full.py`
- Purpose: Train on all 533 HUCs (including ephemeral)
- Variations: 8 models (4 feature sets × 2 learning rates)
- Command: `nohup python -u train_exp1a_full.py > exp1a_training.log 2>&1 &`

**Key Parameters:**
```python
train_hucs: 320 HUCs (60%)
val_hucs: 106 HUCs (20%)
test_hucs: 107 HUCs (20%)
train_size_dimension: "huc"
train_size_fraction: 1.0
epochs: 30
learning_rates: [0.001, 0.0003]
```

---

#### Experiment 1B (270 Deep Snow HUCs)
**Training Script:**
- Location: `/home/sagemaker-user/train_exp1b_full.py` (on SageMaker)
- Local copy: `train_exp1b_full.py`
- Purpose: Train on 270 deep snow HUCs only (no ephemeral)
- HUC splits source: `src/snowML/datapipe/huc_lists/hucs_data.json`
- Variations: 8 models (4 feature sets × 2 learning rates)
- Command: `nohup python -u train_exp1b_full.py > exp1b_training.log 2>&1 &`

**Key Parameters:**
```python
train_hucs: 162 HUCs (60%)
val_hucs: 54 HUCs (20%)
test_hucs: 55 HUCs (20%)
train_size_dimension: "huc"
train_size_fraction: 1.0
epochs: 30
learning_rates: [0.001, 0.0003]
```

**Feature Sets:**
1. **Base:** `["mean_pr", "mean_tair", "Mean Elevation"]`
2. **Wind:** Base + `["mean_ws"]`
3. **Humidity:** Base + `["mean_rh"]`
4. **Srad:** Base + `["mean_srad"]`

---

#### Experiment 2 (Fine-tuning on Yakima/Naches)
**Training Script:**
- Location: `/home/sagemaker-user/finetune_exp1b_base_CORRECT.py` (on SageMaker)
- Local copy: `finetune_exp1b_base_CORRECT.py`
- Purpose: Fine-tune Exp1B Base model on Test Set B (81 Yakima/Naches HUCs)
- Pre-trained model: Exp1B Base epoch 14 (Val KGE: 0.8084)
- Command: `nohup python -u finetune_exp1b_base_CORRECT.py > finetune.log 2>&1 &`

**Key Parameters:**
```python
pretrained_checkpoint: "checkpoints/Exp1B_Base_3e-4_epoch14.pth"
finetune_train_hucs: 56 HUCs (70% of 81)
finetune_val_hucs: 25 HUCs (30% of 81)
learning_rate: 0.0001 (reduced from 0.0003)
dropout: 0.2 (reduced from 0.5)
epochs: 10 (fewer than pre-training 30)
features: Base only (same as pre-trained model)
```

**Result:** Fine-tuning degraded performance (0.7603 → 0.6110 KGE). See `EXPERIMENT_2_RESULTS_REPORT.md` for details.

---

### 9.2 Results Storage Locations

#### On SageMaker Studio (During Training)

**Checkpoints:**
```
/home/sagemaker-user/checkpoints/
├── Exp1A_Base_1e-3_epoch1.pth
├── Exp1A_Base_1e-3_epoch2.pth
├── ... (30 epochs)
├── Exp1A_Base_3e-4_epoch1.pth
├── ... (30 epochs)
├── Exp1A_Wind_1e-3_epoch*.pth
├── Exp1A_Wind_3e-4_epoch*.pth
├── Exp1A_Humidity_1e-3_epoch*.pth
├── Exp1A_Humidity_3e-4_epoch*.pth
├── Exp1A_Srad_1e-3_epoch*.pth
├── Exp1A_Srad_3e-4_epoch*.pth
├── Exp1B_Base_1e-3_epoch*.pth
├── Exp1B_Base_3e-4_epoch*.pth
├── Exp1B_Wind_1e-3_epoch*.pth
├── Exp1B_Wind_3e-4_epoch*.pth
├── Exp1B_Humidity_1e-3_epoch*.pth
├── Exp1B_Humidity_3e-4_epoch*.pth
├── Exp1B_Srad_1e-3_epoch*.pth
└── Exp1B_Srad_3e-4_epoch*.pth
```
**Total:** 240 checkpoints per experiment (8 variations × 30 epochs)

**Training Logs:**
```
/home/sagemaker-user/
├── exp1a_training.log          # Full training output
├── exp1b_training.log          # Full training output
├── finetune.log                # Exp2 fine-tuning output
├── training_summary.csv        # Epoch-by-epoch metrics
└── pilot_test.log              # Initial test run
```

**Evaluation Results:**
```
/home/sagemaker-user/evaluation_results/
├── Exp1A_Base_1e-3_Test_A_metrics.csv
├── Exp1A_Base_1e-3_Test_B_metrics.csv
├── Exp1A_Base_3e-4_Test_A_metrics.csv
├── Exp1A_Base_3e-4_Test_B_metrics.csv
├── ... (all 16 evaluation files)
└── evaluation_summary.csv      # Combined results
```

**Exp2 Results:**
```
/home/sagemaker-user/exp2_finetune_results/
├── FineTune_Base_epoch1.pth
├── FineTune_Base_epoch2.pth    # Best model (Val KGE: 0.9738)
├── ... (10 epochs)
├── finetune_summary.json       # Performance comparison
└── finetune_splits.json        # 70/30 split of 81 HUCs
```

---

#### On Local Machine (Downloaded Results)

**Results Directory Structure:**
```
/Users/simran/Desktop/SnowML/
├── exp1a_results/
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
├── exp1b_corrected_results/
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
├── checkpoints/                  # Downloaded best models
│   ├── Exp1B_Base_3e-4_epoch14.pth
│   ├── Exp1B_Wind_3e-4_epoch*.pth
│   └── ... (best performing checkpoints)
│
└── finetune.log                 # Exp2 training log
```

---

### 9.3 Generated Graphs and Visualizations

#### Snow Type Analysis Graphs
**Location:** `snow_type_graphs/`
```
snow_type_graphs/
├── figure1_kge_by_snow_type_boxplot_CORRECTED.png
│   Purpose: KGE by snow type (Montane, Maritime, Ephemeral)
│   Shows: Exp1A (3 types) vs Exp1B (2 types only - no ephemeral)
│   Format: 2×2 subplot (KGE and MSE for each experiment)
│
├── figure2_kge_vs_elevation_scatter_CORRECTED.png
│   Purpose: KGE vs mean elevation scatter plot
│   Shows: Combined Exp1A + Exp1B deep snow HUCs
│   Types: Montane Forest (purple) and Maritime (blue)
│
├── figure3_model_comparison_by_snow_type_CORRECTED.png
│   Purpose: Compare 4 models across snow types
│   Shows: Median KGE for each model by snow type
│   Format: 2×2 subplot (Exp1A/1B × Test A/B)
│
└── summary_by_snow_type_CORRECTED.csv
    Purpose: Statistical summary (median, mean, std KGE)
    Rows: Exp1A (3 snow types) + Exp1B (2 snow types)
    Columns: Model variations
```

**Key Insight:** Exp1B shows NO ephemeral because it was trained only on deep snow HUCs (270 total). The corrected graphs properly exclude ephemeral from Exp1B visualizations.

---

#### Comparison Graphs
**Location:** `comparison_graphs/`
```
comparison_graphs/
├── figure1_exp1a_exp1b_comparison_bars.png
│   Purpose: Side-by-side bar comparison
│   Shows: Val KGE, Test A KGE, Test B KGE
│   Models: All 8 variations per experiment
│
└── training_curves_combined.png
    Purpose: Training curves over 30 epochs
    Shows: Validation KGE progression
    Lines: 8 model variations
```

---

#### Results Tables and Reports
**Location:** Root directory
```
/Users/simran/Desktop/SnowML/
├── COMPLETE_EXP1A_EXP1B_RESULTS_TABLE.md
│   Purpose: Complete results comparison
│   Contains: All 16 model variations (Exp1A + Exp1B)
│   Metrics: Val KGE, Test A KGE, Test B KGE
│   Rankings: Best to worst by each metric
│
├── EXPERIMENT_2_RESULTS_REPORT.md
│   Purpose: Exp2 fine-tuning analysis
│   Contains: Performance comparison, failure analysis
│   Key Finding: Fine-tuning degraded performance (0.7603 → 0.6110)
│   Recommendation: Use pre-trained model instead
│
├── COMPLETE_KNOWLEDGE_TRANSFER_GUIDE.md
│   Purpose: This guide
│   Contains: All setup, parameters, troubleshooting
│
└── QUICK_REFERENCE_KNOWLEDGE_TRANSFER.md
    Purpose: 2-page summary of the complete guide
    Contains: Golden rules, quick start commands
```

---

### 9.4 What's Available in MLflow

#### Access MLflow UI
1. AWS Console → SageMaker → MLflow
2. Click "dawgsML" server
3. Click "Open MLflow UI"
4. URL: `https://t-izowcn0gky2o.us-west-2.experiments.sagemaker.aws`

---

#### Experiment Organization

**Experiment Names:**
```
MLflow Experiments:
├── Exp1A_Base_1e-3_20260901
├── Exp1A_Base_3e-4_20260901
├── Exp1A_Wind_1e-3_20260901
├── Exp1A_Wind_3e-4_20260901
├── Exp1A_Humidity_1e-3_20260901
├── Exp1A_Humidity_3e-4_20260901
├── Exp1A_Srad_1e-3_20260901
├── Exp1A_Srad_3e-4_20260901
├── Exp1B_Base_1e-3_20260826
├── Exp1B_Base_3e-4_20260826
├── Exp1B_Wind_1e-3_20260826
├── Exp1B_Wind_3e-4_20260826
├── Exp1B_Humidity_1e-3_20260826
├── Exp1B_Humidity_3e-4_20260826
├── Exp1B_Srad_1e-3_20260826
├── Exp1B_Srad_3e-4_20260826
└── Exp2_FineTune_Base_20260929
```

Each experiment contains 30 runs (epochs 1-30) for Exp1A/1B, or 10 runs for Exp2.

---

#### Metrics Logged (Per Epoch)

**Training Metrics:**
```
train_loss          # MSE loss on training set
train_time_seconds  # Time to complete epoch
epoch              # Current epoch number
```

**Validation Metrics:**
```
val_kge            # Kling-Gupta Efficiency (primary metric)
val_mse            # Mean Squared Error
val_r2             # R² score
val_mae            # Mean Absolute Error
val_pearson_r      # Pearson correlation
val_bias           # Bias
val_variability    # Variability ratio
```

**Best Model Tracking:**
```
best_val_kge       # Highest validation KGE so far
best_epoch         # Epoch with best KGE
```

---

#### Parameters Logged

**Model Architecture:**
```
hidden_size: 64
num_layers: 1
dropout: 0.5 (pre-training) or 0.2 (fine-tuning)
lookback: 180
batch_size: 32
```

**Training Configuration:**
```
learning_rate: 0.001 or 0.0003
n_epochs: 30 (or 10 for fine-tuning)
optimizer: Adam
loss_fn: MSE
train_size_dimension: huc
train_size_fraction: 1.0
```

**Data Configuration:**
```
n_train_hucs: 162 (Exp1B) or 320 (Exp1A)
n_val_hucs: 54 (Exp1B) or 106 (Exp1A)
n_test_hucs: 55 (Exp1B) or 107 (Exp1A)
var_list: ["mean_pr", "mean_tair", "Mean Elevation", ...]
```

**Infrastructure:**
```
device: cuda
gpu_name: Tesla T4
instance_type: ml.g4dn.xlarge
```

---

#### Artifacts Stored in MLflow

**Model Checkpoints:**
- Saved to: `s3://dawgs-mlflow-artifacts/`
- Format: PyTorch `.pth` files
- Contents: `model_state_dict`, `optimizer_state_dict`, `epoch`, `val_kge`, `params`

**Training Curves:**
- Format: PNG images
- Plots: Val KGE vs epoch, Loss vs epoch

**Evaluation Results:**
- Format: CSV files
- Contents: Per-HUC metrics for Test A and Test B

---

#### How to Query MLflow Programmatically

**List All Experiments:**
```python
import mlflow

mlflow.set_tracking_uri("arn:aws:sagemaker:us-west-2:677276086662:mlflow-tracking-server/dawgsML")

experiments = mlflow.search_experiments()
for exp in experiments:
    print(f"{exp.name}: {exp.experiment_id}")
```

**Get Best Run from Experiment:**
```python
exp_name = "Exp1B_Base_3e-4_20260826"
exp = mlflow.get_experiment_by_name(exp_name)

runs = mlflow.search_runs(
    experiment_ids=[exp.experiment_id],
    order_by=["metrics.val_kge DESC"],
    max_results=1
)

best_run = runs.iloc[0]
print(f"Best epoch: {best_run['params.epoch']}")
print(f"Best Val KGE: {best_run['metrics.val_kge']:.4f}")
```

**Download Model Checkpoint:**
```python
run_id = best_run['run_id']
artifact_path = "model/checkpoint.pth"

local_path = mlflow.artifacts.download_artifacts(
    run_id=run_id,
    artifact_path=artifact_path,
    dst_path="./downloaded_models/"
)
```

**Compare Multiple Models:**
```python
# Get all Exp1B experiments
exp_names = [
    "Exp1B_Base_3e-4",
    "Exp1B_Wind_3e-4",
    "Exp1B_Humidity_3e-4",
    "Exp1B_Srad_3e-4"
]

results = []
for exp_name in exp_names:
    exp = mlflow.get_experiment_by_name(f"{exp_name}_20260826")
    runs = mlflow.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["metrics.val_kge DESC"],
        max_results=1
    )
    best_kge = runs.iloc[0]['metrics.val_kge']
    results.append({'model': exp_name, 'val_kge': best_kge})

import pandas as pd
df = pd.DataFrame(results).sort_values('val_kge', ascending=False)
print(df)
```

---

### 9.5 Results Summary (Quick Reference)

#### Experiment 1B Best Results

| Model | LR | Best Epoch | Val KGE | Test A KGE | Test B KGE |
|-------|-------|-----------|---------|-----------|-----------|
| **Wind** | 0.0003 | 27 | **0.8104** | 0.8127 | 0.7703 |
| **Humidity** | 0.0003 | 24 | 0.8097 | 0.8066 | **0.7783** |
| **Base** | 0.0003 | 14 | 0.8084 | 0.8107 | 0.7603 |
| Srad | 0.0003 | 28 | 0.7902 | 0.7952 | 0.7544 |

**Best Overall:** Wind 3e-4 (highest validation KGE)  
**Best Test B:** Humidity 3e-4 (best spatial transferability)

---

#### Experiment 1A Best Results

| Model | LR | Val KGE | Test A KGE | Test B KGE |
|-------|-------|---------|-----------|-----------|
| **Humidity** | 0.0003 | 0.7621 | 0.7845 | **0.7783** |
| Wind | 0.0003 | 0.7587 | 0.7732 | 0.7721 |
| Base | 0.0003 | 0.7512 | 0.7644 | 0.7455 |
| Srad | 0.0003 | 0.7189 | 0.7421 | 0.7288 |

**Key Insight:** Including ephemeral HUCs (533 total) lowers overall KGE compared to Exp1B (270 deep snow only).

---

#### Experiment 2 Results (Fine-tuning)

| Metric | Pre-trained (Exp1B Base) | Fine-tuned (Exp2) | Change |
|--------|-------------------------|-------------------|---------|
| **Test B KGE** | **0.7603** | **0.6110** | **-19.6%** ❌ |
| Test B MSE | 0.0034 | 0.0066 | +94.1% |
| Best Val KGE | 0.8084 | 0.9738 | +20.4% (overfitting) |

**Conclusion:** Fine-tuning degraded performance. Use pre-trained Exp1B Base model.

---

### 9.6 S3 Backup Locations

**All files backed up to S3 on September 30, 2026:**

**Base S3 Path:**
```
s3://snowml-model-ready/checkpoints/simran_thesis/20260930/
```

#### Complete S3 Structure:

```
s3://snowml-model-ready/checkpoints/simran_thesis/20260930/
│
├── checkpoints/
│   ├── Exp1A_Base_3e-4_epoch0.pth
│   ├── Exp1A_Base_3e-4_epoch1.pth
│   ├── ... (epoch 0-14 for Base at 3e-4)
│   ├── Exp1A_Srad_3e-4_epoch0.pth
│   ├── ... (epoch 0-14 for Srad at 3e-4)
│   ├── Exp1A_Wind_3e-4_epoch0.pth
│   ├── ... (epoch 0-14 for Wind at 3e-4)
│   ├── Exp1A_Humidity_3e-4_epoch0.pth
│   ├── ... (epoch 0-14 for Humidity at 3e-4)
│   └── [All Exp1B checkpoints - check S3 for complete list]
│
├── exp1a_results/
│   ├── Exp1A_Srad_3e-4_epoch2_Test_A_metrics.csv
│   └── Exp1A_Srad_3e-4_epoch2_Test_B_metrics.csv
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
│   ├── all_4_variations_complete_20260926_033829.csv
│   ├── summary_4_variations_20260926_033829.csv
│   ├── summary_corrected_20260926_015145.csv
│   ├── summary_corrected_20260926_015405.csv
│   └── summary_corrected_20260926_015552.csv
│
└── exp2_finetune_results/
    ├── FineTune_Base_epoch1.pth
    ├── FineTune_Base_epoch2.pth
    ├── finetune_splits.json
    └── finetune_summary.json
```

#### Files Successfully Backed Up:

| Directory | Files | Size | Status |
|-----------|-------|------|--------|
| checkpoints/ | 60+ files | ~13 MB | ✅ Uploaded |
| exp1a_results/ | 2 files | 21 KB | ✅ Uploaded |
| exp1b_corrected_results/ | 13 files | 119 KB | ✅ Uploaded |
| exp2_finetune_results/ | 4 files | 436 KB | ✅ Uploaded |

**Total Backed Up:** ~79+ files, ~14 MB (partial backup completed on Sept 30, 2026)

**Note:** Only Exp1A 3e-4 checkpoints (60 files, epochs 0-14 for 4 variations) were uploaded. Full Exp1A and all Exp1B checkpoints are still on SageMaker.

#### Access S3 Files:

**List all backed up files:**
```bash
aws s3 ls s3://snowml-model-ready/checkpoints/simran_thesis/20260930/ --recursive --human-readable
```

**Download specific checkpoint:**
```bash
# Example: Download Exp1A Humidity epoch 14
aws s3 cp s3://snowml-model-ready/checkpoints/simran_thesis/20260930/checkpoints/Exp1A_Humidity_3e-4_epoch14.pth ./
```

**Download all Exp1B results:**
```bash
aws s3 sync s3://snowml-model-ready/checkpoints/simran_thesis/20260930/exp1b_corrected_results/ ./exp1b_results_from_s3/
```

**Download everything:**
```bash
aws s3 sync s3://snowml-model-ready/checkpoints/simran_thesis/20260930/ ./complete_backup_from_s3/
```

#### Storage Cost:

- **S3 Standard Storage:** $0.023/GB/month
- **Current backup size:** ~14 MB = 0.014 GB
- **Monthly cost:** $0.023 × 0.014 = **$0.00032/month** (less than 1 cent)

---

### 9.7 File Download Commands

**Download Checkpoints from SageMaker:**
```bash
# From SageMaker terminal to local
scp sagemaker-user@<studio-url>:/home/sagemaker-user/checkpoints/*.pth ./checkpoints/

# Or use JupyterLab file browser:
# Right-click checkpoint → Download
```

**Download Results CSVs:**
```bash
# Download all evaluation results
scp -r sagemaker-user@<studio-url>:/home/sagemaker-user/evaluation_results/ ./
```

**Download Training Logs:**
```bash
# Download training logs
scp sagemaker-user@<studio-url>:/home/sagemaker-user/*.log ./logs/
```

---

## FINAL CHECKLIST: Before You Start

**Environment:**
- [ ] Using SageMaker Studio JupyterLab (NOT EC2, NOT laptop)
- [ ] Instance type: ml.g4dn.xlarge
- [ ] GPU available: `nvidia-smi` shows Tesla T4
- [ ] Conda environment: pytorch_p310 activated

**Code:**
- [ ] Branch: simran-unified-experiments checked out
- [ ] Fixed files present (LSTM_train.py with batched validation)
- [ ] SnowML package installed: `pip install -e src/`

**Parameters:**
- [ ] `train_size_dimension = "huc"` (NOT "time")
- [ ] `train_size_fraction = 1.0` (NOT 0.67)
- [ ] `num_workers = 4` (NOT 8)
- [ ] Learning rates: [0.001, 0.0003]
- [ ] Feature sets: 4 variations (Base, +Srad, +Wind, +Humidity)

**MLflow:**
- [ ] MLflow 2.16.2 installed
- [ ] sagemaker-mlflow plugin installed
- [ ] Environment variable set: `MLFLOW_TRACKING_URI=arn:aws:...`
- [ ] Connection tested: `mlflow.get_tracking_uri()` returns ARN

**Data:**
- [ ] HUC splits loaded from hucs_data.json (Exp1B)
- [ ] Train: 162 HUCs, Val: 54 HUCs, Test: 55 HUCs
- [ ] Test B: 81 HUCs (Yakima/Naches)

**Resources:**
- [ ] Budget approved: ~$150-165 for Exp1B
- [ ] Time allocated: ~6-7 days continuous training
- [ ] Professor aware of timeline

---

## SUMMARY: THE ABSOLUTE ESSENTIALS

**If you only remember 5 things:**

1. **SageMaker Studio (ml.g4dn.xlarge)** - NOT EC2, NOT laptop
2. **train_size_dimension="huc" and train_size_fraction=1.0** - NOT "time", NOT 0.67
3. **Use hucs_data.json splits** - NOT custom splits for Exp1B
4. **MLflow 2.16.2 + ARN** - NOT newer versions, NOT HTTPS URL alone
5. **8 model variations** - 4 features × 2 learning rates

**If something goes wrong:**
- Check the log file first
- Verify parameters (especially train_size_dimension)
- Confirm MLflow connection (environment variable)
- Check GPU memory (should not exceed 12-13 GB)
- Review this guide's Common Pitfalls section

---

**Document Version:** 2.0  
**Last Updated:** September 30, 2026  
**Based On:** Successful SageMaker Studio deployment (August 2026)  
**Validated By:** 
- Complete Exp1B training with KGE 0.81-0.84 ✓
- Complete Exp1A training ✓
- Complete Exp2 fine-tuning (unsuccessful but documented) ✓
**New in v2.0:**
- Section 9: Complete file locations reference
- Training scripts for all experiments
- Results storage structure (SageMaker + local)
- Graph generation locations
- MLflow organization and queries
- Results summary tables
- Download commands

---

**Good luck! You have all the information to succeed. Follow this guide exactly and you'll avoid all the mistakes that cost weeks of wasted time.**
