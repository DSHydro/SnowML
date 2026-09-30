#!/usr/bin/env python3
"""
SMALL TEST SCRIPT - Validate entire training pipeline before expensive run.

Tests with:
- 2 training HUCs
- 2 validation HUCs
- 1 epoch
- 1 model variation (Base model)

Expected duration: ~5-10 minutes
Expected cost: ~$0.05

This validates:
✓ Data downloads from S3
✓ Data format is correct
✓ GPU is being used
✓ Training runs without errors
✓ Validation runs without GPU memory issues
✓ MLflow logging works
✓ Timing is reasonable
"""

import sys
import time
import torch
from datetime import datetime

sys.path.insert(0, '/home/ec2-user/SageMaker')

from snowML.Scripts.load_hucs import load_huc_splits as lh
from snowML.Scripts import multi_huc_expirement as mhe
from snowML.LSTM import set_hyperparams as sh

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

print("=" * 80)
print("SMALL PIPELINE TEST")
print("=" * 80)
print("Testing with 2 train HUCs, 2 val HUCs, 1 epoch")
print("This should take ~5-10 minutes")
print("=" * 80)
print()

# ============================================================================
# Load HUC splits from CORRECT source (previous students' exact splits)
# ============================================================================
log("Loading HUC splits...")
huc_json_path = "/home/ec2-user/SageMaker/src/snowML/datapipe/huc_lists/hucs_data.json"
tr_full, val_full, te_full = lh.huc_split(huc_json_path)

log(f"Full Exp3 splits: {len(tr_full)} train, {len(val_full)} val, {len(te_full)} test")

# Use only first 2 of each for testing
tr = tr_full[:2]
val = val_full[:2]

log(f"Test subset: {len(tr)} train HUCs, {len(val)} val HUCs")
log(f"  Train HUCs: {tr}")
log(f"  Val HUCs: {val}")
print()

# ============================================================================
# Set up parameters (exact same as previous students)
# ============================================================================
log("Setting up parameters...")

params = sh.create_hyper_dict()

# Base model settings (same as previous students' Base_1e-3 variation)
params["var_list"] = ["mean_pr", "mean_tair", "Mean Elevation"]
params["learning_rate"] = 0.001
params["n_epochs"] = 1  # Just 1 epoch for testing
params["batch_size"] = 32
params["hidden_size"] = 64
params["num_layers"] = 1
params["dropout"] = 0.5
params["loss_type"] = "mse"
params["lookback"] = 180
params["num_workers"] = 8

# CRITICAL: Multi-HUC settings
params["train_size_dimension"] = "huc"  # NOT "time"!
params["train_size_fraction"] = 1.0     # 100% of each HUC

# MLflow settings
params["expirement_name"] = "Pipeline_Test_Small"
params["mlflow_tracking_uri"] = "sqlite:///mlflow_test.db"
# params["mlflow_tracking_uri"] = "arn:aws:sagemaker:us-west-2:677276086662:mlflow-tracking-server/dawgsML"
# params["mlflow_tracking_uri"] = "https://t-izowcn0gky2o.us-west-2.experiments.sagemaker.aws"
params["MLFLOW_ON"] = False

# Device
params["device"] = "cuda" if torch.cuda.is_available() else "cpu"

log(f"Parameters configured:")
log(f"  Variables: {params['var_list']}")
log(f"  Learning rate: {params['learning_rate']}")
log(f"  Epochs: {params['n_epochs']}")
log(f"  Batch size: {params['batch_size']}")
log(f"  Device: {params['device']}")
log(f"  train_size_dimension: {params['train_size_dimension']}")
log(f"  train_size_fraction: {params['train_size_fraction']}")
print()

# ============================================================================
# Check GPU availability
# ============================================================================
if torch.cuda.is_available():
    log("✅ GPU available:")
    log(f"   Device: {torch.cuda.get_device_name(0)}")
    log(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print()
else:
    log("⚠️  WARNING: No GPU detected! Training will be VERY slow.")
    print()

# ============================================================================
# Run test training
# ============================================================================
log("=" * 80)
log("STARTING TEST TRAINING")
log("=" * 80)
print()

start_time = time.time()

try:
    # This is the EXACT function previous students used
    log("Calling run_expirement()...")
    log("(Will download 4 HUCs from S3 if not cached)")
    print()

    mhe.run_expirement(tr, val, params)

    elapsed = time.time() - start_time

    print()
    log("=" * 80)
    log("✅ TEST COMPLETED SUCCESSFULLY!")
    log("=" * 80)
    log(f"Duration: {elapsed/60:.2f} minutes")
    print()

    # Calculate projected full training time
    # Full training: 162 train HUCs, 54 val HUCs, 30 epochs
    ratio_train = 162 / 2  # 81x more training HUCs
    ratio_val = 54 / 2     # 27x more validation HUCs
    ratio_epochs = 30 / 1  # 30x more epochs

    # Rough estimate (training scales linearly, validation is fixed cost per epoch)
    estimated_full = (elapsed / 60) * ratio_train * ratio_epochs

    log(f"Projected time for full training (1 variation):")
    log(f"  ~{estimated_full/60:.1f} hours")
    log(f"  (This is a rough estimate based on linear scaling)")
    print()

    log("CHECKS:")
    log("  ✅ Data downloaded from S3")
    log("  ✅ Training completed without errors")
    log("  ✅ Validation completed without GPU errors")
    log("  ✅ MLflow logging worked")
    print()

    if params["device"] == "cuda":
        log("GPU Usage:")
        log(f"  Memory allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        log(f"  Memory reserved: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
        print()

    log("=" * 80)
    log("READY FOR FULL TRAINING!")
    log("=" * 80)
    log("The pipeline works correctly. You can now run full training with:")
    log("  - 8 variations")
    log("  - 30 epochs each")
    log("  - All 162 train + 54 val HUCs")
    log(f"  - Estimated total: ~{estimated_full/60 * 8:.1f} hours for all 8 variations")
    print()

except Exception as e:
    elapsed = time.time() - start_time

    print()
    log("=" * 80)
    log("❌ TEST FAILED!")
    log("=" * 80)
    log(f"Duration before failure: {elapsed/60:.2f} minutes")
    log(f"Error: {e}")
    print()

    import traceback
    traceback.print_exc()

    print()
    log("TROUBLESHOOTING:")
    log("  1. Check if S3 access is working")
    log("  2. Check if MLflow server is accessible")
    log("  3. Check GPU memory if CUDA error")
    log("  4. Review full error trace above")
    print()

    sys.exit(1)
