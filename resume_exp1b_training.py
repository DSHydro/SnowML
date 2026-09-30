#!/usr/bin/env python3
"""
Resume Exp1B Training - Complete remaining variations
- Variation 4 (Srad_3e-4): Epochs 28-29 (resume from checkpoint)
- Variations 5-8: Full 30 epochs each
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
print("RESUME EXP1B TRAINING - Complete Variations 4-8")
print("=" * 80)
print()

# Load HUC splits
log("Loading HUC splits...")
huc_json_path = "/home/sagemaker-user/src/snowML/datapipe/huc_lists/hucs_data.json"
tr, val, te = lh.huc_split(huc_json_path)

log(f"Full splits: {len(tr)} train, {len(val)} val, {len(te)} test")
print()

# Define remaining variations
variations = [
    # Variation 4: Resume from epoch 27 (checkpoint exists), need epochs 28-29
    {"name": "Srad_3e-4", "var_list": ["mean_pr", "mean_tair", "Mean Elevation", "mean_srad"],
     "learning_rate": 0.0003, "start_epoch": 28, "resume": True},

    # Variations 5-8: Start from scratch
    {"name": "Wind_1e-3", "var_list": ["mean_pr", "mean_tair", "Mean Elevation", "mean_wind_speed"],
     "learning_rate": 0.001, "start_epoch": 0, "resume": False},
    {"name": "Wind_3e-4", "var_list": ["mean_pr", "mean_tair", "Mean Elevation", "mean_wind_speed"],
     "learning_rate": 0.0003, "start_epoch": 0, "resume": False},
    {"name": "Humidity_1e-3", "var_list": ["mean_pr", "mean_tair", "Mean Elevation", "mean_rh"],
     "learning_rate": 0.001, "start_epoch": 0, "resume": False},
    {"name": "Humidity_3e-4", "var_list": ["mean_pr", "mean_tair", "Mean Elevation", "mean_rh"],
     "learning_rate": 0.0003, "start_epoch": 0, "resume": False}
]

total_start_time = time.time()

# Train each variation
for i, var in enumerate(variations, 1):
    log("=" * 80)
    if var["resume"]:
        log(f"VARIATION 4/8 (RESUME): {var['name']} - Epochs {var['start_epoch']}-29")
    else:
        log(f"VARIATION {i+3}/8: {var['name']}")
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

    # Multi-HUC settings
    params["train_size_dimension"] = "huc"
    params["train_size_fraction"] = 1.0

    # MLflow settings
    params["expirement_name"] = f"Exp1B_{var['name']}"
    params["mlflow_tracking_uri"] = "arn:aws:sagemaker:us-west-2:677276086662:mlflow-tracking-server/dawgsML"
    params["MLFLOW_ON"] = True

    # Device
    params["device"] = "cuda" if torch.cuda.is_available() else "cpu"

    log(f"Parameters configured:")
    log(f"  Variables: {params['var_list']}")
    log(f"  Learning rate: {params['learning_rate']}")
    if var["resume"]:
        log(f"  Resume from: epoch {var['start_epoch']-1} checkpoint")
        log(f"  Remaining epochs: {30 - var['start_epoch']}")
    else:
        log(f"  Epochs: {params['n_epochs']}")
    log(f"  Device: {params['device']}")
    print()

    # Check GPU
    if torch.cuda.is_available():
        log("✅ GPU available:")
        log(f"   Device: {torch.cuda.get_device_name(0)}")
        log(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print()

    log(f"Starting training for {var['name']}...")
    print()

    try:
        # Run training
        mhe.run_expirement(tr, val, params)

        var_elapsed = time.time() - var_start_time
        log(f"✅ {var['name']} completed in {var_elapsed/3600:.2f} hours")
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
log(f"ALL VARIATIONS COMPLETED in {total_elapsed/3600:.2f} hours")
log("=" * 80)
