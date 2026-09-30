#!/usr/bin/env python3
"""
EXACT REPLICA of what previous students did for Exp1B (Multi-HUC Training)
========================================================================

This script runs the EXACT same code from the repo with NO modifications.
The previous students successfully trained 8 variations using this approach.

Variables:
- Base (LR 1e-3, 3e-4): mean_pr, mean_tair, Mean Elevation
- Solar (LR 1e-3, 3e-4): Base + mean_srad
- Wind (LR 1e-3, 3e-4): Base + mean_vs
- Humidity (LR 1e-3, 3e-4): Base + mean_hum

Data: 270 HUCs (162 train / 54 val / 55 test) from hucs_data.json
Method: train_size_dimension="huc", train_size_fraction=1.0
Expected: ~3-4 hours per variation, ~28-32 hours total
"""

import sys
sys.path.insert(0, '/home/ec2-user/SageMaker')

from snowML.Scripts.load_hucs import load_huc_splits as lh
from snowML.Scripts import multi_huc_expirement as mhe
from snowML.LSTM import set_hyperparams as sh
from datetime import datetime

def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

# ============================================================================
# STEP 1: Load HUC splits (exact same as previous students)
# ============================================================================
log("=" * 80)
log("EXPERIMENT 1B - EXACT REPLICA OF PREVIOUS STUDENTS' TRAINING")
log("=" * 80)

# Load splits from the EXACT file they used
huc_json_path = "/home/ec2-user/SageMaker/src/snowML/datapipe/huc_lists/hucs_data.json"
tr, val, te = lh.huc_split(huc_json_path)

log(f"Loaded HUC splits:")
log(f"  Train: {len(tr)} HUCs")
log(f"  Val: {len(val)} HUCs")
log(f"  Test: {len(te)} HUCs")
log("")

# ============================================================================
# STEP 2: Define 8 variations (exact same as previous students)
# ============================================================================
variations = [
    # Base model (temp + precip + elevation)
    {"name": "Base", "lr": 0.001, "vars": ["mean_pr", "mean_tair", "Mean Elevation"]},
    {"name": "Base", "lr": 0.0003, "vars": ["mean_pr", "mean_tair", "Mean Elevation"]},

    # Solar radiation
    {"name": "Base_Solar", "lr": 0.001, "vars": ["mean_pr", "mean_tair", "Mean Elevation", "mean_srad"]},
    {"name": "Base_Solar", "lr": 0.0003, "vars": ["mean_pr", "mean_tair", "Mean Elevation", "mean_srad"]},

    # Wind speed
    {"name": "Base_Wind", "lr": 0.001, "vars": ["mean_pr", "mean_tair", "Mean Elevation", "mean_vs"]},
    {"name": "Base_Wind", "lr": 0.0003, "vars": ["mean_pr", "mean_tair", "Mean Elevation", "mean_vs"]},

    # Humidity
    {"name": "Base_Humidity", "lr": 0.001, "vars": ["mean_pr", "mean_tair", "Mean Elevation", "mean_hum"]},
    {"name": "Base_Humidity", "lr": 0.0003, "vars": ["mean_pr", "mean_tair", "Mean Elevation", "mean_hum"]},
]

log(f"Will train {len(variations)} variations")
log("")

# ============================================================================
# STEP 3: Train each variation
# ============================================================================
start_all = datetime.now()

for i, var in enumerate(variations, 1):
    start_var = datetime.now()

    log("=" * 80)
    log(f"VARIATION {i}/8: {var['name']}_LR{var['lr']}")
    log("=" * 80)
    log(f"Variables: {var['vars']}")
    log(f"Learning Rate: {var['lr']}")
    log("")

    # Create params using the EXACT default structure
    params = sh.create_hyper_dict()

    # Override only what's needed for multi-HUC training
    params["var_list"] = var["vars"]
    params["learning_rate"] = var["lr"]
    params["n_epochs"] = 30
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

    # MLflow settings - use AWS server (same as previous students)
    params["expirement_name"] = f"Exp1B_Simran_{var['name']}_LR{var['lr']}"
    params["mlflow_tracking_uri"] = "arn:aws:sagemaker:us-west-2:677276086662:mlflow-tracking-server/dawgsML"
    params["MLFLOW_ON"] = True

    # Device settings (use GPU if available)
    import torch
    params["device"] = "cuda" if torch.cuda.is_available() else "cpu"

    log(f"Parameters:")
    log(f"  train_size_dimension: {params['train_size_dimension']}")
    log(f"  train_size_fraction: {params['train_size_fraction']}")
    log(f"  n_epochs: {params['n_epochs']}")
    log(f"  batch_size: {params['batch_size']}")
    log(f"  device: {params['device']}")
    log(f"  mlflow_uri: {params['mlflow_tracking_uri']}")
    log("")

    # Run the EXACT same function previous students used
    log(f"Starting training (est. ~3-4 hours)...")
    log("")

    try:
        # This is the EXACT function call from multi_huc_expirement.py
        # It does everything:
        # - Downloads data from S3
        # - Trains on all training HUCs
        # - Validates on all validation HUCs each epoch
        # - Logs to MLflow
        # - Saves model checkpoints
        mhe.run_expirement(tr, val, params)

        elapsed = (datetime.now() - start_var).total_seconds() / 3600
        log("")
        log(f"✅ Variation {i}/8 COMPLETE in {elapsed:.2f} hours")
        log("")

    except Exception as e:
        log(f"❌ ERROR in variation {i}: {e}")
        import traceback
        traceback.print_exc()
        log("Continuing to next variation...")
        log("")

# ============================================================================
# SUMMARY
# ============================================================================
elapsed_total = (datetime.now() - start_all).total_seconds() / 3600

log("=" * 80)
log("ALL 8 VARIATIONS COMPLETE!")
log("=" * 80)
log(f"Total time: {elapsed_total:.2f} hours")
log(f"Average per variation: {elapsed_total/8:.2f} hours")
log("")
log("Results are logged in MLflow server:")
log("  arn:aws:sagemaker:us-west-2:677276086662:mlflow-tracking-server/dawgsML")
log("")
log("⚠️  REMEMBER TO STOP YOUR AWS INSTANCE!")
log("=" * 80)
