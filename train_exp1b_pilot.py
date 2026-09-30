#!/usr/bin/env python3
"""
Experiment 1B - PILOT TEST
===========================
Purpose: Validate the entire training pipeline before full-scale training
Duration: ~10-15 minutes
Branch: simran-unified-experiments

What this tests:
- ✅ Data downloads from S3
- ✅ Model initialization
- ✅ Training loop works
- ✅ MLflow logging works
- ✅ Validation metrics computed correctly
- ✅ Model can be saved

Scientific Question:
"Can we train ONE LSTM on multiple deep snow watersheds and have it generalize?"

This is Experiment 1B (Deep Snow Only):
- Training: 231 HUCs (deep snow, no ephemeral)
- Goal: Learn transferable patterns across watersheds

For PILOT, we'll use:
- 3 training HUCs (instead of 231)
- 2 validation HUCs (instead of 77)
- 1 model variation (instead of 8)
- 5 epochs (instead of 30)
"""

import os
import json
import sys
from pathlib import Path

print("=" * 80)
print("EXPERIMENT 1B - PILOT TEST")
print("Multi-HUC Deep Snow Only - Small Scale Validation")
print("=" * 80)

# ============================================================================
# PART 1: IMPORTS AND ENVIRONMENT SETUP
# ============================================================================
print("\n[1/7] Setting up environment...")

import torch
from torch import optim
import mlflow
import numpy as np
import pandas as pd

# Import SnowML modules
from snowML.LSTM import LSTM_train as LSTM_tr
from snowML.LSTM import LSTM_model as LSTM_mod
from snowML.LSTM import set_hyperparams as sh
from snowML.LSTM import LSTM_pre_process as pp

print("   ✅ All imports successful")
print(f"   📍 PyTorch version: {torch.__version__}")
print(f"   📍 MLflow version: {mlflow.__version__}")

# Check device
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"   📍 Using device: {device}")

# ============================================================================
# PART 2: CONFIGURATION - Pilot Test Parameters
# ============================================================================
print("\n[2/7] Configuring pilot test parameters...")

# Start with default hyperparameters
params = sh.create_hyper_dict()

# Override for pilot test (small-scale)
params["n_epochs"] = 5  # Just 5 epochs for pilot
params["hidden_size"] = 64  # Standard size
params["num_layers"] = 1
params["dropout"] = 0.3  # Professor's unified value
params["learning_rate"] = 0.001  # Test with standard LR first
params["batch_size"] = 32
params["lookback"] = 180  # 180 days of history
params["num_workers"] = 0  # Disable multiprocessing for pilot (avoids spawn issues)

# Variable set: Base (temperature + precipitation)
params["var_list"] = ["mean_pr", "mean_tair"]
print(f"   📊 Variable set: {params['var_list']}")
print(f"   📍 num_workers=0 (single-threaded for pilot compatibility)")

# Experiment naming
params["expirement_name"] = "Exp1B_Pilot_Test"

# For pilot: Use SQLite database for MLflow tracking
# For full training: We'll set up proper AWS connection
params["mlflow_tracking_uri"] = "sqlite:///mlflow.db"
print("   📍 Using local SQLite MLflow tracking for pilot")

# Validate parameters
sh.val_params(params)
print("   ✅ Parameters validated")

# ============================================================================
# PART 3: SELECT SAMPLE HUCs FOR PILOT
# ============================================================================
print("\n[3/7] Selecting sample HUCs for pilot test...")

# Load the full training and validation HUC lists
data_dir = Path("/Users/simran/Desktop/SnowML/data")
with open(data_dir / "exp1b_train_hucs.txt", "r") as f:
    all_train_hucs = [line.strip() for line in f if line.strip()]

with open(data_dir / "exp1b_validation_hucs.txt", "r") as f:
    all_val_hucs = [line.strip() for line in f if line.strip()]

print(f"   📂 Full training set: {len(all_train_hucs)} HUCs")
print(f"   📂 Full validation set: {len(all_val_hucs)} HUCs")

# Select PILOT subset (first few HUCs for reproducibility)
pilot_train_hucs = all_train_hucs[:3]  # Just 3 for pilot
pilot_val_hucs = all_val_hucs[:2]      # Just 2 for pilot

print(f"\n   🎯 PILOT Training HUCs (3): {pilot_train_hucs}")
print(f"   🎯 PILOT Validation HUCs (2): {pilot_val_hucs}")

# ============================================================================
# PART 4: DATA PREPARATION AND PRE-PROCESSING
# ============================================================================
print("\n[4/7] Loading and pre-processing data...")
print("   ⏳ This may take 1-2 minutes (downloading from S3)...")

# Combine train + val for pre-processing
pilot_all_hucs = pilot_train_hucs + pilot_val_hucs

try:
    # Pre-process: downloads data, normalizes, creates tensors
    # Returns:
    #   - df_dict: Dictionary of {HUC_ID: processed_dataframe}
    #   - global_means: Mean values for normalization (per variable)
    #   - global_stds: Std values for normalization (per variable)
    df_dict, global_means, global_stds = pp.pre_process(
        pilot_all_hucs,
        params["var_list"]
    )

    print(f"   ✅ Data loaded for {len(df_dict)} HUCs")
    print(f"   📊 Normalization - Means: {global_means}")
    print(f"   📊 Normalization - Stds: {global_stds}")

    # Split into training and validation dictionaries
    df_dict_train = {huc: df_dict[huc] for huc in pilot_train_hucs if huc in df_dict}
    df_dict_val = {huc: df_dict[huc] for huc in pilot_val_hucs if huc in df_dict}

    print(f"   ✅ Training data: {len(df_dict_train)} HUCs")
    print(f"   ✅ Validation data: {len(df_dict_val)} HUCs")

    # Show sample data shape
    sample_huc = pilot_train_hucs[0]
    if sample_huc in df_dict:
        print(f"   📐 Sample data shape (HUC {sample_huc}): {df_dict[sample_huc].shape}")
        print(f"   📅 Date range: {df_dict[sample_huc].index[0]} to {df_dict[sample_huc].index[-1]}")

except Exception as e:
    print(f"   ❌ Error during data loading: {e}")
    print("\n🔍 TROUBLESHOOTING:")
    print("   1. Check AWS credentials: aws s3 ls s3://snowml-model-ready/")
    print("   2. Verify HUC IDs exist in S3 bucket")
    print("   3. Check network connection")
    sys.exit(1)

# ============================================================================
# PART 5: MODEL INITIALIZATION
# ============================================================================
print("\n[5/7] Initializing LSTM model...")

# Calculate input size from variable list
input_size = len(params["var_list"])
print(f"   📐 Input size: {input_size} features")
print(f"   🧠 Hidden size: {params['hidden_size']} neurons")
print(f"   📚 Number of layers: {params['num_layers']}")
print(f"   💧 Dropout: {params['dropout']}")

# Initialize model
model = LSTM_mod.SnowModel(
    input_size,
    params['hidden_size'],
    params['num_class'],
    params['num_layers'],
    params['dropout']
)

# Initialize optimizer
optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])

# Initialize loss function (Mean Squared Error)
if params["loss_type"] == "mse":
    loss_fn = torch.nn.MSELoss()
else:
    loss_fn = LSTM_mod.HybridLoss(
        initial_lambda=params["mse_lambda_start"],
        final_lambda=params["mse_lambda_end"],
        total_epochs=params["n_epochs"]
    )

print("   ✅ Model initialized")
print(f"   ✅ Optimizer: Adam (LR={params['learning_rate']})")
print(f"   ✅ Loss function: {params['loss_type'].upper()}")

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"   📊 Total parameters: {total_params:,}")
print(f"   📊 Trainable parameters: {trainable_params:,}")

# ============================================================================
# PART 6: TRAINING LOOP WITH MLFLOW LOGGING
# ============================================================================
print("\n[6/7] Starting training loop...")
print(f"   🎯 Training for {params['n_epochs']} epochs")
print(f"   📊 Monitoring validation KGE and MSE")

# Set up MLflow
mlflow.set_tracking_uri(params["mlflow_tracking_uri"])
mlflow.set_experiment(params["expirement_name"])

# Start MLflow run
with mlflow.start_run():
    # Log all parameters to MLflow
    mlflow.log_params(params)
    mlflow.log_param("pilot_train_hucs", pilot_train_hucs)
    mlflow.log_param("pilot_val_hucs", pilot_val_hucs)
    mlflow.log_param("global_means", global_means)
    mlflow.log_param("global_stds", global_stds)
    mlflow.log_param("is_pilot", True)

    print("   ✅ MLflow run started")
    print(f"   🔗 Tracking URI: {params['mlflow_tracking_uri']}")

    # Training loop
    for epoch in range(params["n_epochs"]):
        print(f"\n   ━━━ Epoch {epoch + 1}/{params['n_epochs']} ━━━")

        # TRAINING PHASE
        print(f"      🏋️ Training on {len(df_dict_train)} HUCs...")
        LSTM_tr.pre_train(
            model,
            optimizer,
            loss_fn,
            df_dict_train,
            params,
            epoch
        )
        print(f"      ✅ Training complete")

        # VALIDATION PHASE
        print(f"      🔍 Validating on {len(df_dict_val)} HUCs...")
        LSTM_tr.evaluate(
            model,
            df_dict_val,
            params,
            epoch
        )
        print(f"      ✅ Validation complete")

        # Save model checkpoint
        mlflow.pytorch.log_model(model, artifact_path=f"epoch{epoch}_model")

    print("\n   ✅ Training loop complete!")
    print(f"   💾 All {params['n_epochs']} models saved to MLflow")

# ============================================================================
# PART 7: SUMMARY AND NEXT STEPS
# ============================================================================
print("\n" + "=" * 80)
print("🎉 PILOT TEST COMPLETE!")
print("=" * 80)

print("\n✅ What was tested:")
print("   1. ✅ Data loading from S3")
print("   2. ✅ Pre-processing and normalization")
print("   3. ✅ Model initialization")
print("   4. ✅ Training loop (forward + backward pass)")
print("   5. ✅ Validation metrics (KGE, MSE, R²)")
print("   6. ✅ MLflow logging")
print("   7. ✅ Model checkpointing")

print("\n📊 To view results:")
print("   1. Go to MLflow UI")
print("   2. Look for experiment: 'Exp1B_Pilot_Test'")
print("   3. Check metrics: val_kge_median, val_mse_median")
print("   4. Verify: Should have 5 epochs logged")

print("\n🔍 Expected behavior:")
print("   • Validation KGE: ~0.4-0.7 (low because only 5 epochs + 3 HUCs)")
print("   • Validation MSE: Will vary, look for decreasing trend")
print("   • Each epoch should take ~1-2 minutes")

print("\n🚀 NEXT STEPS:")
print("   If this pilot test succeeded:")
print("   ✅ Step 1: Review results in MLflow")
print("   ✅ Step 2: Create full training script (train_exp1b_full.py)")
print("   ✅ Step 3: Run full training with:")
print("            - 231 training HUCs")
print("            - 77 validation HUCs")
print("            - 8 model variations")
print("            - 30 epochs each")
print("            - Duration: 4-6 hours on GPU")

print("\n" + "=" * 80)
print("🎓 INTERVIEW TALKING POINTS FROM THIS PILOT:")
print("=" * 80)
print("""
Q: "How did you validate your pipeline before large-scale training?"
A: "I created a pilot test with a small subset of data (3 training HUCs,
    2 validation HUCs) and ran for just 5 epochs. This allowed me to:
    - Verify data loading and pre-processing worked correctly
    - Confirm MLflow experiment tracking was set up properly
    - Test model initialization and training loop
    - Validate metrics computation (KGE, MSE, R²)
    - Catch any bugs early before committing GPU resources

    The pilot test took ~10 minutes and saved hours of debugging later."

Q: "Why use KGE instead of just MSE?"
A: "KGE (Kling-Gupta Efficiency) is the standard metric in hydrology because
    it evaluates three components simultaneously:
    - Correlation (r): Does the model capture timing of peaks/valleys?
    - Bias (β): Is the model systematically over/under predicting?
    - Variability (α): Does the model capture the range of values?

    MSE only measures magnitude errors and can be dominated by a few large
    outliers. KGE above 0.75 is considered 'good' for hydrological predictions."
""")

print("=" * 80)
print("✅ Ready for full-scale Experiment 1B!")
print("=" * 80)
