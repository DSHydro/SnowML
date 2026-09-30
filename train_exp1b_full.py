#!/usr/bin/env python3
"""
Experiment 1B - FULL TRAINING
==============================
Purpose: Train 8 model variations on all deep snow HUCs
Duration: ~4-6 hours on GPU
Branch: simran-unified-experiments

Scientific Question:
"Can we train ONE LSTM on multiple deep snow watersheds and have it generalize?"

This is Experiment 1B (Deep Snow Only):
- Training: 231 HUCs (deep snow, no ephemeral)
- Validation: 77 HUCs (20% held-out)
- Test Set A: 78 HUCs (20% random held-out)
- Test Set B: 81 HUCs (Yakima/Naches - unseen region)

Model Variations (8 total):
- 4 variable sets × 2 learning rates = 8 models
- After all finish, select best by median validation KGE

Variable Sets:
1. Base: [temperature, precipitation]
2. Base + Wind: [temperature, precipitation, wind_speed]
3. Base + Solar: [temperature, precipitation, solar_radiation]
4. Base + Solar + Wind: [temperature, precipitation, solar_radiation, wind_speed]

Learning Rates:
- 0.001 (standard)
- 0.0003 (slower, more stable)
"""

import os
import json
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd

print("=" * 80)
print("EXPERIMENT 1B - FULL TRAINING")
print("Multi-HUC Deep Snow Only - Production Scale")
print("=" * 80)
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ============================================================================
# IMPORTS
# ============================================================================
print("\n[Step 1/9] Importing libraries...")

import torch
from torch import optim
import mlflow
import numpy as np

from snowML.LSTM import LSTM_train as LSTM_tr
from snowML.LSTM import LSTM_model as LSTM_mod
from snowML.LSTM import set_hyperparams as sh
from snowML.LSTM import LSTM_pre_process as pp

print("   ✅ All imports successful")
print(f"   📍 PyTorch version: {torch.__version__}")
print(f"   📍 MLflow version: {mlflow.__version__}")

device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"   📍 Using device: {device}")
if device == 'cuda':
    print(f"   📍 GPU: {torch.cuda.get_device_name(0)}")

# ============================================================================
# CONFIGURATION - Define 8 Model Variations
# ============================================================================
print("\n[Step 2/9] Configuring model variations...")

# Base hyperparameters (same for all variations)
BASE_PARAMS = {
    "hidden_size": 64,
    "num_layers": 1,
    "dropout": 0.3,
    "batch_size": 32,
    "lookback": 180,
    "n_epochs": 30,
    "num_workers": 0,  # Single-threaded for stability
    "num_class": 1,
    "loss_type": "mse",
    "recursive_predict": False,
    "lag_days": 30,
    "lag_swe_var_idx": 3,
    "filter_dates": ["1984-10-01", "2021-09-30"],
    "train_size_dimension": "time",
    "train_size_fraction": 0.67,
    "custom_delta": 0.04,
    "UCLA": False,
    "Stop_Loss": False,
    "KGE_target": 0.9,
    "mse_lambda_start": 1,
    "mse_lambda_end": 0.5,
    "expirement_name": "Exp1B_Full_DeepSnowOnly",
    "mlflow_tracking_uri": "sqlite:///mlflow.db",
    "MLFLOW_ON": True
}

# Define 4 variable sets
VARIABLE_SETS = [
    {
        "name": "Base",
        "vars": ["mean_pr", "mean_tair"],
        "description": "Temperature + Precipitation only"
    },
    {
        "name": "Base_Wind",
        "vars": ["mean_pr", "mean_tair", "mean_vs"],
        "description": "Base + Wind Speed"
    },
    {
        "name": "Base_Solar",
        "vars": ["mean_pr", "mean_tair", "mean_srad"],
        "description": "Base + Solar Radiation"
    },
    {
        "name": "Base_Solar_Wind",
        "vars": ["mean_pr", "mean_tair", "mean_srad", "mean_vs"],
        "description": "Base + Solar + Wind"
    }
]

# Define 2 learning rates
LEARNING_RATES = [0.001, 0.0003]

# Generate all 8 variations
variations = []
for var_set in VARIABLE_SETS:
    for lr in LEARNING_RATES:
        variation = {
            "name": f"{var_set['name']}_LR{lr}",
            "var_list": var_set['vars'],
            "learning_rate": lr,
            "description": f"{var_set['description']}, LR={lr}"
        }
        variations.append(variation)

print(f"   📊 Total variations to train: {len(variations)}")
for i, var in enumerate(variations, 1):
    print(f"   {i}. {var['name']}: {var['description']}")

# ============================================================================
# LOAD TRAINING AND VALIDATION HUCs
# ============================================================================
print("\n[Step 3/9] Loading HUC splits...")

data_dir = Path("/Users/simran/Desktop/SnowML/data")

# Load training HUCs (231 deep snow HUCs)
with open(data_dir / "exp1b_train_hucs.txt", "r") as f:
    train_hucs = [line.strip() for line in f if line.strip()]

# Load validation HUCs (77 deep snow HUCs)
with open(data_dir / "exp1b_validation_hucs.txt", "r") as f:
    val_hucs = [line.strip() for line in f if line.strip()]

print(f"   📂 Training HUCs: {len(train_hucs)}")
print(f"   📂 Validation HUCs: {len(val_hucs)}")
print(f"   📂 Total: {len(train_hucs) + len(val_hucs)} HUCs")

# ============================================================================
# DATA PREPARATION
# ============================================================================
print("\n[Step 4/9] Downloading and pre-processing data...")
print("   ⏳ This will take 5-10 minutes (downloading ~1GB from S3)...")
print(f"   📍 Started: {datetime.now().strftime('%H:%M:%S')}")

# We need to prepare data for the most complex variable set first (all 4 variables)
# This way we can subset later for simpler models
all_vars = ["mean_pr", "mean_tair", "mean_srad", "mean_vs"]
all_hucs = train_hucs + val_hucs

try:
    df_dict, global_means, global_stds = pp.pre_process(all_hucs, all_vars)

    print(f"   ✅ Data loaded for {len(df_dict)} HUCs")
    print(f"   📍 Finished: {datetime.now().strftime('%H:%M:%S')}")
    print(f"   📊 Normalization stats computed:")
    print(f"      - Means: {dict(global_means)}")
    print(f"      - Stds: {dict(global_stds)}")

    # Split into train and validation
    df_dict_train = {huc: df_dict[huc] for huc in train_hucs if huc in df_dict}
    df_dict_val = {huc: df_dict[huc] for huc in val_hucs if huc in df_dict}

    print(f"   ✅ Training data: {len(df_dict_train)} HUCs")
    print(f"   ✅ Validation data: {len(df_dict_val)} HUCs")

except Exception as e:
    print(f"   ❌ Error during data loading: {e}")
    sys.exit(1)

# ============================================================================
# TRAINING LOOP - All 8 Variations
# ============================================================================
print("\n[Step 5/9] Starting training for all 8 variations...")
print(f"   🎯 {len(variations)} models × 30 epochs each")
print(f"   ⏱️  Estimated total time: 4-6 hours")
print(f"   📍 Training started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Set up MLflow
mlflow.set_tracking_uri(BASE_PARAMS["mlflow_tracking_uri"])
mlflow.set_experiment(BASE_PARAMS["expirement_name"])

# Store results for model selection
all_results = []

for idx, variation in enumerate(variations, 1):
    print("\n" + "=" * 80)
    print(f"VARIATION {idx}/{len(variations)}: {variation['name']}")
    print("=" * 80)
    print(f"   Variables: {variation['var_list']}")
    print(f"   Learning Rate: {variation['learning_rate']}")
    print(f"   Started: {datetime.now().strftime('%H:%M:%S')}")

    # Create params for this variation
    params = BASE_PARAMS.copy()
    params["var_list"] = variation['var_list']
    params["learning_rate"] = variation['learning_rate']
    params["variation_name"] = variation['name']

    # Validate params
    sh.val_params(params)

    # Initialize model
    input_size = len(params["var_list"])
    model = LSTM_mod.SnowModel(
        input_size,
        params['hidden_size'],
        params['num_class'],
        params['num_layers'],
        params['dropout']
    )

    optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])

    if params["loss_type"] == "mse":
        loss_fn = torch.nn.MSELoss()
    else:
        loss_fn = LSTM_mod.HybridLoss(
            initial_lambda=params.get("mse_lambda_start", 1),
            final_lambda=params.get("mse_lambda_end", 0.5),
            total_epochs=params["n_epochs"]
        )

    print(f"   ✅ Model initialized: {sum(p.numel() for p in model.parameters()):,} parameters")

    # Start MLflow run for this variation
    with mlflow.start_run(run_name=variation['name']):
        # Log parameters
        mlflow.log_params(params)
        mlflow.log_param("train_hucs", train_hucs)
        mlflow.log_param("val_hucs", val_hucs)
        mlflow.log_param("global_means", global_means)
        mlflow.log_param("global_stds", global_stds)
        mlflow.log_param("variation_name", variation['name'])

        print(f"   ✅ MLflow run started")

        # Training loop
        for epoch in range(params["n_epochs"]):
            epoch_start = datetime.now()
            print(f"\n   ━━━ Epoch {epoch + 1}/{params['n_epochs']} ━━━")

            # Training
            print(f"      🏋️  Training on {len(df_dict_train)} HUCs...")
            LSTM_tr.pre_train(
                model,
                optimizer,
                loss_fn,
                df_dict_train,
                params,
                epoch
            )

            # Validation
            print(f"      🔍 Validating on {len(df_dict_val)} HUCs...")
            LSTM_tr.evaluate(
                model,
                df_dict_val,
                params,
                epoch
            )

            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0 or epoch == params["n_epochs"] - 1:
                mlflow.pytorch.log_model(model, artifact_path=f"epoch{epoch}_model")
                print(f"      💾 Checkpoint saved (epoch {epoch})")

            epoch_duration = (datetime.now() - epoch_start).total_seconds()
            print(f"      ⏱️  Epoch time: {epoch_duration:.1f}s")

        # Get final validation metrics
        print(f"\n   ✅ Training complete for {variation['name']}")
        print(f"   📍 Finished: {datetime.now().strftime('%H:%M:%S')}")

        # Store result for model selection
        run_info = mlflow.active_run().info
        all_results.append({
            "variation_name": variation['name'],
            "run_id": run_info.run_id,
            "var_list": variation['var_list'],
            "learning_rate": variation['learning_rate']
        })

print("\n" + "=" * 80)
print("✅ ALL 8 VARIATIONS TRAINED!")
print("=" * 80)

# ============================================================================
# MODEL SELECTION
# ============================================================================
print("\n[Step 6/9] Selecting best model by validation KGE...")

# Load all runs from MLflow
client = mlflow.tracking.MlflowClient()
experiment = client.get_experiment_by_name(BASE_PARAMS["expirement_name"])
runs = client.search_runs(experiment_ids=[experiment.experiment_id])

# Find best model by median validation KGE
best_run = None
best_kge = -999

for run in runs:
    run_name = run.data.params.get("variation_name", "")

    # Get all validation KGE metrics
    kge_metrics = [v for k, v in run.data.metrics.items() if "val_" in k and "_kge" in k]

    if kge_metrics:
        median_kge = np.median(kge_metrics)
        print(f"   {run_name}: Median validation KGE = {median_kge:.4f}")

        if median_kge > best_kge:
            best_kge = median_kge
            best_run = run

if best_run:
    print(f"\n   🏆 BEST MODEL: {best_run.data.params.get('variation_name')}")
    print(f"   📊 Median validation KGE: {best_kge:.4f}")
    print(f"   🆔 Run ID: {best_run.info.run_id}")

    # Save best model info
    best_model_info = {
        "variation_name": best_run.data.params.get("variation_name"),
        "run_id": best_run.info.run_id,
        "median_val_kge": float(best_kge),
        "var_list": best_run.data.params.get("var_list"),
        "learning_rate": best_run.data.params.get("learning_rate"),
        "timestamp": datetime.now().isoformat()
    }

    with open("exp1b_best_model.json", "w") as f:
        json.dump(best_model_info, f, indent=2)

    print(f"   💾 Best model info saved to: exp1b_best_model.json")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("🎉 EXPERIMENT 1B FULL TRAINING COMPLETE!")
print("=" * 80)

print(f"\n📊 Summary:")
print(f"   • Total variations trained: {len(variations)}")
print(f"   • Training HUCs: {len(train_hucs)}")
print(f"   • Validation HUCs: {len(val_hucs)}")
print(f"   • Epochs per variation: 30")
print(f"   • Best model: {best_model_info['variation_name']}")
print(f"   • Best median validation KGE: {best_kge:.4f}")

print(f"\n📁 Outputs:")
print(f"   • MLflow database: sqlite:///mlflow.db")
print(f"   • Experiment name: {BASE_PARAMS['expirement_name']}")
print(f"   • Best model info: exp1b_best_model.json")

print(f"\n🔍 View results:")
print(f"   mlflow ui --backend-store-uri sqlite:///mlflow.db")
print(f"   Then open: http://127.0.0.1:5000")

print(f"\n🚀 Next Steps:")
print(f"   1. Review results in MLflow UI")
print(f"   2. Test best model on Test Set A (78 random held-out HUCs)")
print(f"   3. Test best model on Test Set B (81 Yakima/Naches HUCs)")
print(f"   4. Compare to published results")
print(f"   5. Move to Experiment 1A or Experiment 2")

print(f"\n📍 Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)
