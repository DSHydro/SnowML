#!/usr/bin/env python3
"""
Experiment 1A Training - All 8 Variations (15 epochs)
Based on successful Exp1B code with modifications:
- 15 epochs (not 30) - saves 50% time
- 272 train HUCs (vs 162 in Exp1B) - includes ephemeral
- All 8 variations (4 features × 2 learning rates)

Expected time: ~100 hours (8 variations × ~12 hours each)
Expected cost: ~$16-17 @ $0.16/hour (Spot)
"""

import sys
import time
import torch
from datetime import datetime

sys.path.insert(0, '/home/sagemaker-user/src')

from snowML.Scripts import multi_huc_expirement as mhe
from snowML.LSTM import set_hyperparams as sh

def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

print("=" * 80)
print("EXPERIMENT 1A TRAINING - 8 VARIATIONS")
print("All HUCs including ephemeral (272 train / 90 val / 92 test)")
print("15 epochs per variation (optimized from 30)")
print("=" * 80)
print()

# Load Exp1A HUC splits from files
log("Loading Exp1A HUC splits from files...")
with open('/home/sagemaker-user/src/data/exp1a_train_hucs.txt', 'r') as f:
    train_hucs = [line.strip() for line in f if line.strip()]
with open('/home/sagemaker-user/src/data/exp1a_validation_hucs.txt', 'r') as f:
    val_hucs = [line.strip() for line in f if line.strip()]
with open('/home/sagemaker-user/src/data/exp1a_test_a_hucs.txt', 'r') as f:
    test_a_hucs = [line.strip() for line in f if line.strip()]

log(f"Splits loaded: {len(train_hucs)} train, {len(val_hucs)} val, {len(test_a_hucs)} test")
log(f"Expected: 272 train, 90 val, 92 test")
print()

# Verify correct splits
if len(train_hucs) != 272 or len(val_hucs) != 90:
    log("⚠️  WARNING: Unexpected HUC counts!")
    log(f"Got {len(train_hucs)} train (expected 272), {len(val_hucs)} val (expected 90)")
    log("Continuing anyway...")
print()

# Define all 8 variations (4 features × 2 learning rates)
variations = [
    # Learning Rate = 0.001
    {
        "name": "Base_1e-3",
        "var_list": ["mean_pr", "mean_tair", "Mean Elevation"],
        "learning_rate": 0.001,
        "description": "Baseline (Temp + Precip + Elevation)"
    },
    {
        "name": "Srad_1e-3",
        "var_list": ["mean_pr", "mean_tair", "Mean Elevation", "mean_srad"],
        "learning_rate": 0.001,
        "description": "Base + Solar Radiation"
    },
    {
        "name": "Wind_1e-3",
        "var_list": ["mean_pr", "mean_tair", "Mean Elevation", "mean_vs"],
        "learning_rate": 0.001,
        "description": "Base + Wind Speed"
    },
    {
        "name": "Humidity_1e-3",
        "var_list": ["mean_pr", "mean_tair", "Mean Elevation", "mean_hum"],
        "learning_rate": 0.001,
        "description": "Base + Humidity"
    },
    # Learning Rate = 0.0003
    {
        "name": "Base_3e-4",
        "var_list": ["mean_pr", "mean_tair", "Mean Elevation"],
        "learning_rate": 0.0003,
        "description": "Baseline (Temp + Precip + Elevation)"
    },
    {
        "name": "Srad_3e-4",
        "var_list": ["mean_pr", "mean_tair", "Mean Elevation", "mean_srad"],
        "learning_rate": 0.0003,
        "description": "Base + Solar Radiation"
    },
    {
        "name": "Wind_3e-4",
        "var_list": ["mean_pr", "mean_tair", "Mean Elevation", "mean_vs"],
        "learning_rate": 0.0003,
        "description": "Base + Wind Speed (BEST in Exp1B)"
    },
    {
        "name": "Humidity_3e-4",
        "var_list": ["mean_pr", "mean_tair", "Mean Elevation", "mean_hum"],
        "learning_rate": 0.0003,
        "description": "Base + Humidity"
    }
]

log("Variations to train:")
for i, var in enumerate(variations, 1):
    log(f"  {i}. {var['name']}: {var['description']}")
print()

log("⚠️  NOTE: Based on Exp1B, LR=0.001 variations may fail/underperform")
log("         LR=0.0003 variations are proven to work well")
print()

total_start_time = time.time()

# Train each variation
for i, var in enumerate(variations, 1):
    log("=" * 80)
    log(f"VARIATION {i}/8: {var['name']}")
    log(f"Description: {var['description']}")
    log(f"Progress: {i}/8 variations")
    log("=" * 80)

    var_start_time = time.time()

    # Set up parameters
    params = sh.create_hyper_dict()

    # Model settings
    params["var_list"] = var["var_list"]
    params["learning_rate"] = var["learning_rate"]
    params["n_epochs"] = 15  # ← CHANGED FROM 30 TO 15 (saves 50% time)
    params["batch_size"] = 32
    params["hidden_size"] = 64
    params["num_layers"] = 1
    params["dropout"] = 0.5
    params["loss_type"] = "mse"
    params["lookback"] = 180
    params["num_workers"] = 4

    # Multi-HUC settings (CRITICAL - spatial transfer learning)
    params["train_size_dimension"] = "huc"  # ← SPATIAL splits (not temporal!)
    params["train_size_fraction"] = 1.0      # ← Use 100% of each HUC

    # MLflow settings
    params["expirement_name"] = f"Exp1A_{var['name']}"
    params["mlflow_tracking_uri"] = "arn:aws:sagemaker:us-west-2:677276086662:mlflow-tracking-server/dawgsML"
    params["MLFLOW_ON"] = True

    # Device
    params["device"] = "cuda" if torch.cuda.is_available() else "cpu"

    log(f"Configuration:")
    log(f"  Experiment: Exp1A (with ephemeral basins)")
    log(f"  Train HUCs: {len(train_hucs)} (vs 162 in Exp1B)")
    log(f"  Variables: {params['var_list']}")
    log(f"  Learning rate: {params['learning_rate']}")
    log(f"  Epochs: {params['n_epochs']} ← OPTIMIZED (was 30)")
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
    log(f"Estimated time: ~10-12 hours per variation")
    print()

    try:
        # Run training
        mhe.run_expirement(train_hucs, val_hucs, params)

        var_elapsed = time.time() - var_start_time
        total_elapsed = time.time() - total_start_time
        remaining_variations = 8 - i
        est_remaining = (total_elapsed / i) * remaining_variations

        log("=" * 80)
        log(f"✅ {var['name']} COMPLETED!")
        log(f"⏱️  Variation time: {var_elapsed/3600:.2f} hours")
        log(f"⏱️  Total elapsed: {total_elapsed/3600:.2f} hours")
        if remaining_variations > 0:
            log(f"📊 Progress: {i}/8 complete")
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
log("🎉 EXPERIMENT 1A TRAINING COMPLETED!")
log("=" * 80)
log(f"⏱️  Total time: {total_elapsed/3600:.2f} hours")
log(f"💰 Estimated cost: ${total_elapsed/3600 * 0.16:.2f} (Spot)")
log("")
log("Completed variations:")
for i, var in enumerate(variations, 1):
    log(f"  {i}. {var['name']}: {var['description']}")
log("")
log("Next steps:")
log("1. Check MLflow for best model by median validation KGE")
log("2. Compare Exp1A vs Exp1B:")
log("   - Does ephemeral help or hurt?")
log("   - Which learning rate/features work best?")
log("3. Select BEST Exp1A model for Experiment 2 fine-tuning")
log("4. Download results and evaluate on Test Sets A and B")
log("5. Stop Studio app to save costs!")
log("=" * 80)
