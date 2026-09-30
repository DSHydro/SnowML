#!/usr/bin/env python3
"""
8 Variations Training - EXACT same structure as test_mlflow_studio.py
Full Training: 162 train HUCs, 54 val HUCs, 30 epochs × 8 variations
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
print("FULL EXP1B TRAINING - 8 VARIATIONS × 30 EPOCHS")
print("=" * 80)
print()

# Load HUC splits
log("Loading HUC splits...")
huc_json_path = "/home/sagemaker-user/src/snowML/datapipe/huc_lists/hucs_data.json"
tr, val, te = lh.huc_split(huc_json_path)

log(f"Full splits: {len(tr)} train, {len(val)} val, {len(te)} test")
print()

# Define 8 variations (EXACT same format as test)
variations = [
    {"name": "Base_1e-3", "var_list": ["mean_pr", "mean_tair", "Mean Elevation"], "learning_rate": 0.001},
    {"name": "Base_3e-4", "var_list": ["mean_pr", "mean_tair", "Mean Elevation"], "learning_rate": 0.0003},
    {"name": "Srad_1e-3", "var_list": ["mean_pr", "mean_tair", "Mean Elevation", "mean_srad"], "learning_rate": 0.001},
    {"name": "Srad_3e-4", "var_list": ["mean_pr", "mean_tair", "Mean Elevation", "mean_srad"], "learning_rate": 0.0003},
    {"name": "Wind_1e-3", "var_list": ["mean_pr", "mean_tair", "Mean Elevation", "mean_wind_speed"], "learning_rate": 0.001},
    {"name": "Wind_3e-4", "var_list": ["mean_pr", "mean_tair", "Mean Elevation", "mean_wind_speed"], "learning_rate": 0.0003},
    {"name": "Humidity_1e-3", "var_list": ["mean_pr", "mean_tair", "Mean Elevation", "mean_rh"], "learning_rate": 0.001},
    {"name": "Humidity_3e-4", "var_list": ["mean_pr", "mean_tair", "Mean Elevation", "mean_rh"], "learning_rate": 0.0003}
]

total_start_time = time.time()

# Train each variation
for i, var in enumerate(variations, 1):
    log("=" * 80)
    log(f"VARIATION {i}/8: {var['name']}")
    log("=" * 80)

    var_start_time = time.time()

    # Set up parameters (EXACT same as test_mlflow_studio.py)
    params = sh.create_hyper_dict()

    # Model settings
    params["var_list"] = var["var_list"]
    params["learning_rate"] = var["learning_rate"]
    params["n_epochs"] = 30  # Full 30 epochs (not 1 like test)
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
        # EXACT same function call as test
        mhe.run_expirement(tr, val, params)

        var_elapsed = time.time() - var_start_time

        print()
        log("=" * 80)
        log(f"✅ {var['name']} COMPLETED!")
        log("=" * 80)
        log(f"Duration: {var_elapsed/60:.2f} minutes ({var_elapsed/3600:.2f} hours)")
        print()

    except Exception as e:
        var_elapsed = time.time() - var_start_time

        print()
        log("=" * 80)
        log(f"❌ {var['name']} FAILED!")
        log("=" * 80)
        log(f"Duration before failure: {var_elapsed/60:.2f} minutes")
        log(f"Error: {e}")
        print()

        import traceback
        traceback.print_exc()

        print()
        log(f"Continuing to next variation...")
        print()

total_elapsed = time.time() - total_start_time

print()
log("=" * 80)
log("FULL TRAINING COMPLETE!")
log("=" * 80)
log(f"Total time: {total_elapsed/60:.2f} minutes ({total_elapsed/3600:.2f} hours / {total_elapsed/86400:.2f} days)")
print()
log("Next steps:")
log("1. Check MLflow UI for all experiments")
log("2. Verify checkpoints in /home/sagemaker-user/checkpoints/")
log("3. Analyze results and select best models")
print()
