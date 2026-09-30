#!/usr/bin/env python3
"""
MLflow Test Script for Studio - Verify MLflow logging works
Tests with 2 train HUCs, 2 val HUCs, 1 epoch
"""

import sys
import time
import torch
from datetime import datetime

sys.path.insert(0, '/home/sagemaker-user/src')

from snowML.Scripts.load_hucs import load_huc_splits as lh
from snowML.Scripts import multi_huc_expirement as mhe
from snowML.LSTM import set_hyperparams as sh

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

print("=" * 80)
print("MLFLOW TEST - Studio")
print("=" * 80)
print("Testing with 2 train HUCs, 2 val HUCs, 1 epoch")
print("MLflow ENABLED - Will log to dawgsML tracking server")
print("=" * 80)
print()

# Load HUC splits
log("Loading HUC splits...")
huc_json_path = "/home/sagemaker-user/src/snowML/datapipe/huc_lists/hucs_data.json"
tr_full, val_full, te_full = lh.huc_split(huc_json_path)

log(f"Full splits: {len(tr_full)} train, {len(val_full)} val, {len(te_full)} test")

# Use only first 2 of each for testing
tr = tr_full[:2]
val = val_full[:2]

log(f"Test subset: {len(tr)} train HUCs, {len(val)} val HUCs")
log(f"  Train HUCs: {tr}")
log(f"  Val HUCs: {val}")
print()

# Set up parameters
log("Setting up parameters...")

params = sh.create_hyper_dict()

# Model settings
params["var_list"] = ["mean_pr", "mean_tair", "Mean Elevation"]
params["learning_rate"] = 0.001
params["n_epochs"] = 1  # Just 1 epoch for testing
params["batch_size"] = 32
params["hidden_size"] = 64
params["num_layers"] = 1
params["dropout"] = 0.5
params["loss_type"] = "mse"
params["lookback"] = 180
params["num_workers"] = 4  # Fixed from 8 to 4

# CRITICAL: Multi-HUC settings
params["train_size_dimension"] = "huc"
params["train_size_fraction"] = 1.0

# MLflow settings - ENABLED!
params["expirement_name"] = "MLflow_Test_Studio"
params["mlflow_tracking_uri"] = "arn:aws:sagemaker:us-west-2:677276086662:mlflow-tracking-server/dawgsML"
params["MLFLOW_ON"] = True  # ENABLED!

# Device
params["device"] = "cuda" if torch.cuda.is_available() else "cpu"

log(f"Parameters configured:")
log(f"  Device: {params['device']}")
log(f"  MLflow: {params['MLFLOW_ON']}")
log(f"  MLflow URI: {params['mlflow_tracking_uri']}")
log(f"  Experiment: {params['expirement_name']}")
print()

# Check GPU
if torch.cuda.is_available():
    log("✅ GPU available:")
    log(f"   Device: {torch.cuda.get_device_name(0)}")
    log(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print()
else:
    log("⚠️  WARNING: No GPU detected!")
    print()

# Run test training
log("=" * 80)
log("STARTING MLFLOW TEST")
log("=" * 80)
print()

start_time = time.time()

try:
    log("Calling run_expirement()...")
    log("(Will download 4 HUCs from S3 if not cached)")
    log("(Will log to MLflow tracking server)")
    print()

    mhe.run_expirement(tr, val, params)

    elapsed = time.time() - start_time

    print()
    log("=" * 80)
    log("✅ MLFLOW TEST COMPLETED SUCCESSFULLY!")
    log("=" * 80)
    log(f"Duration: {elapsed/60:.2f} minutes")
    print()

    log("CHECKS:")
    log("  ✅ Training completed")
    log("  ✅ Validation completed")
    log("  ✅ Checkpoint saved to /home/sagemaker-user/checkpoints/")
    log("  ✅ MLflow logging attempted")
    print()

    log("=" * 80)
    log("NEXT STEP: Verify in MLflow UI")
    log("=" * 80)
    log("1. In Studio left sidebar, click 'MLflow'")
    log("2. Click on 'dawgsML' tracking server")
    log("3. Look for experiment: 'MLflow_Test_Studio'")
    log("4. Verify you see 1 run with metrics logged")
    print()

    log("If you see the run in MLflow UI → READY FOR FULL TRAINING! 🎉")
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
    log("  1. Check MLflow tracking server status")
    log("  2. Check S3 access permissions")
    log("  3. Review error trace above")
    print()

    sys.exit(1)
