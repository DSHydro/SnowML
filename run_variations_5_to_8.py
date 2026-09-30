#!/usr/bin/env python3
"""
Complete Exp1B Training - Variations 5-8 ONLY
Wind and Humidity models (4 variations × 30 epochs each)
Total time: ~64-72 hours
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
print("EXP1B TRAINING - VARIATIONS 5-8 (Wind + Humidity)")
print("4 variations × 30 epochs each")
print("Expected time: 64-72 hours (~3 days)")
print("=" * 80)
print()

# Load HUC splits
log("Loading HUC splits...")
huc_json_path = "/home/sagemaker-user/src/snowML/datapipe/huc_lists/hucs_data.json"
tr, val, te = lh.huc_split(huc_json_path)

log(f"Splits loaded: {len(tr)} train, {len(val)} val, {len(te)} test")
log(f"Expected: 162 train, 54 val, 54 test")
print()

# Verify correct splits
if len(tr) != 162 or len(val) != 54:
    log("⚠️  WARNING: Unexpected HUC counts!")
    log(f"Got {len(tr)} train (expected 162), {len(val)} val (expected 54)")
    response = input("Continue anyway? (y/n): ")
    if response.lower() != 'y':
        log("Aborted.")
        sys.exit(1)
print()

# Define variations 5-8
variations = [
    {"name": "Wind_1e-3", "var_list": ["mean_pr", "mean_tair", "Mean Elevation", "mean_vs"],
     "learning_rate": 0.001},
    {"name": "Wind_3e-4", "var_list": ["mean_pr", "mean_tair", "Mean Elevation", "mean_vs"],
     "learning_rate": 0.0003},
    {"name": "Humidity_1e-3", "var_list": ["mean_pr", "mean_tair", "Mean Elevation", "mean_hum"],
     "learning_rate": 0.001},
    {"name": "Humidity_3e-4", "var_list": ["mean_pr", "mean_tair", "Mean Elevation", "mean_hum"],
     "learning_rate": 0.0003}
]

total_start_time = time.time()

# Train each variation
for i, var in enumerate(variations, 1):
    log("=" * 80)
    log(f"VARIATION {i+4}/8: {var['name']}")
    log(f"Progress: {i}/4 remaining variations")
    log("=" * 80)

    var_start_time = time.time()

    # Set up parameters
    params = sh.create_hyper_dict()

    # Model settings
    params["var_list"] = var["var_list"]
    params["learning_rate"] = var["learning_rate"]
    params["n_epochs"] = 30
    params["batch_size"] = 32
    params["hidden_size"] = 64
    params["num_layers"] = 1
    params["dropout"] = 0.5
    params["loss_type"] = "mse"
    params["lookback"] = 180
    params["num_workers"] = 4

    # Multi-HUC settings (CRITICAL!)
    params["train_size_dimension"] = "huc"  # ← Entire HUC time series
    params["train_size_fraction"] = 1.0      # ← Use 100% of each HUC

    # MLflow settings
    params["expirement_name"] = f"Exp1B_{var['name']}"
    params["mlflow_tracking_uri"] = "arn:aws:sagemaker:us-west-2:677276086662:mlflow-tracking-server/dawgsML"
    params["MLFLOW_ON"] = True

    # Device
    params["device"] = "cuda" if torch.cuda.is_available() else "cpu"

    log(f"Configuration:")
    log(f"  Variables: {params['var_list']}")
    log(f"  Learning rate: {params['learning_rate']}")
    log(f"  Epochs: {params['n_epochs']}")
    log(f"  Train method: {params['train_size_dimension']} (spatial transfer)")
    log(f"  Device: {params['device']}")
    print()

    # Check GPU
    if torch.cuda.is_available():
        log("✅ GPU detected:")
        log(f"   Name: {torch.cuda.get_device_name(0)}")
        log(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        torch.cuda.empty_cache()
    else:
        log("⚠️  WARNING: No GPU detected! Training will be VERY slow.")
    print()

    log(f"Starting training for {var['name']}...")
    log(f"Estimated completion: {16*(i)} hours from start")
    print()

    try:
        # Run training
        mhe.run_expirement(tr, val, params)

        var_elapsed = time.time() - var_start_time
        total_elapsed = time.time() - total_start_time
        remaining_variations = 4 - i
        est_remaining = (total_elapsed / i) * remaining_variations

        log(f"✅ {var['name']} completed in {var_elapsed/3600:.2f} hours")
        log(f"⏱️  Total elapsed: {total_elapsed/3600:.2f} hours")
        log(f"📊 Estimated time remaining: {est_remaining/3600:.2f} hours")
        print()

    except Exception as e:
        log(f"❌ ERROR in {var['name']}: {str(e)}")
        import traceback
        traceback.print_exc()
        log("Continuing to next variation...")
        print()
        continue

total_elapsed = time.time() - total_start_time
log("=" * 80)
log(f"✅ ALL 4 VARIATIONS COMPLETED!")
log(f"⏱️  Total time: {total_elapsed/3600:.2f} hours")
log(f"💰 Estimated cost: ${total_elapsed/3600 * 0.53:.2f}")
log("=" * 80)
log("")
log("Next steps:")
log("1. Download mlflow.db from AWS MLflow server")
log("2. Analyze results for all 8 variations")
log("3. Stop this Studio app to save costs!")
