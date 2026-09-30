#!/usr/bin/env python3
"""
Complete Exp1B Training - Final 2 Variations (0.0003 LR only)
Based on results: LR=0.001 fails, only LR=0.0003 works well

Variations to run:
- Wind_3e-4: Base + Wind Speed (LR=0.0003)
- Humidity_3e-4: Base + Humidity (LR=0.0003) ← BEST model in original Exp3

Expected time: ~32-36 hours (2 variations × 16 hours each)
Expected cost: ~$17-19 @ $0.53/hour
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
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

print("=" * 80)
print("EXP1B TRAINING - FINAL 2 VARIATIONS")
print("Wind_3e-4 + Humidity_3e-4 (LR=0.0003 only)")
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
    log("Continuing anyway...")
print()

# Define final 2 variations (0.0003 LR only - proven to work)
variations = [
    {
        "name": "Wind_3e-4",
        "var_list": ["mean_pr", "mean_tair", "Mean Elevation", "mean_vs"],
        "learning_rate": 0.0003,
        "description": "Base + Wind Speed"
    },
    {
        "name": "Humidity_3e-4",
        "var_list": ["mean_pr", "mean_tair", "Mean Elevation", "mean_hum"],
        "learning_rate": 0.0003,
        "description": "Base + Humidity (BEST in original Exp3)"
    }
]

log("Variations to train:")
for i, var in enumerate(variations, 1):
    log(f"  {i}. {var['name']}: {var['description']}")
print()

total_start_time = time.time()

# Train each variation
for i, var in enumerate(variations, 1):
    log("=" * 80)
    log(f"VARIATION {i+5}/8: {var['name']}")
    log(f"Description: {var['description']}")
    log(f"Progress: {i}/2 final variations")
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

    # Multi-HUC settings (CRITICAL - spatial transfer learning)
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
    log(f"Estimated single variation time: ~16 hours")
    print()

    try:
        # Run training
        mhe.run_expirement(tr, val, params)

        var_elapsed = time.time() - var_start_time
        total_elapsed = time.time() - total_start_time
        remaining_variations = 2 - i
        est_remaining = (total_elapsed / i) * remaining_variations

        log("=" * 80)
        log(f"✅ {var['name']} COMPLETED!")
        log(f"⏱️  Variation time: {var_elapsed/3600:.2f} hours")
        log(f"⏱️  Total elapsed: {total_elapsed/3600:.2f} hours")
        if remaining_variations > 0:
            log(f"📊 Estimated time remaining: {est_remaining/3600:.2f} hours")
        log("=" * 80)
        print()

    except Exception as e:
        log("=" * 80)
        log(f"❌ ERROR in {var['name']}: {str(e)}")
        log("=" * 80)
        import traceback
        traceback.print_exc()
        log("Continuing to next variation...")
        print()
        continue

total_elapsed = time.time() - total_start_time
log("=" * 80)
log("🎉 ALL TRAINING COMPLETED!")
log("=" * 80)
log(f"⏱️  Total time: {total_elapsed/3600:.2f} hours")
log(f"💰 Estimated cost: ${total_elapsed/3600 * 0.53:.2f}")
log("")
log("✅ Completed variations (6/8 total):")
log("  1. Base_1e-3 (failed - LR too high)")
log("  2. Base_3e-4 ✓")
log("  3. Srad_1e-3 (unstable - LR too high)")
log("  4. Srad_3e-4 (27/30 epochs)")
log("  5. Wind_3e-4 ✓")
log("  6. Humidity_3e-4 ✓ (BEST model from original Exp3)")
log("")
log("Next steps:")
log("1. Check MLflow results: python check_mlflow_results.py")
log("2. Compare with original Exp3 (expected median KGE ~0.82)")
log("3. Stop Studio app to save costs!")
log("4. Download results and analyze")
log("=" * 80)
