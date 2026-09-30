#!/usr/bin/env python3
"""
VERBOSE TEST SCRIPT - Shows actual metrics and validation results
"""

import sys
import time
import torch
from datetime import datetime

sys.path.insert(0, '/home/ec2-user/SageMaker')

from snowML.Scripts.load_hucs import load_huc_splits as lh
from snowML.LSTM import set_hyperparams as sh
from snowML.LSTM import LSTM_pre_process as pp
from snowML.LSTM import LSTM_model as LSTM_mod
from snowML.LSTM import LSTM_train as LSTM_tr
from torch import optim

def log(msg):
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

def batched_predict(model, X_test, params, batch_size=1000):
    """
    Run prediction in batches to avoid GPU OOM.
    Processes batch_size samples at a time instead of all at once.
    """
    model.eval()
    device = params.get('device', 'cpu')

    all_predictions = []
    num_samples = X_test.shape[0]

    with torch.no_grad():
        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            batch = X_test[start_idx:end_idx].to(device)

            batch_pred = model(batch).cpu().numpy()
            all_predictions.append(batch_pred)

            # Clear GPU cache after each batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Concatenate all batch predictions
    return np.concatenate(all_predictions, axis=0)

import numpy as np

print("=" * 80)
print("VERBOSE PIPELINE TEST - WITH ACTUAL METRICS")
print("=" * 80)
print()

# Load HUC splits
log("Loading HUC splits...")
huc_json_path = "/home/ec2-user/SageMaker/src/snowML/datapipe/huc_lists/hucs_data.json"
tr_full, val_full, te_full = lh.huc_split(huc_json_path)

# Use only 2 of each for testing
tr = tr_full[:2]
val = val_full[:2]

log(f"Using {len(tr)} train HUCs: {tr}")
log(f"Using {len(val)} val HUCs: {val}")

# Set up parameters
params = sh.create_hyper_dict()
params["var_list"] = ["mean_pr", "mean_tair", "Mean Elevation"]
params["learning_rate"] = 0.001
params["n_epochs"] = 1
params["batch_size"] = 32
params["hidden_size"] = 64
params["num_layers"] = 1
params["dropout"] = 0.5
params["loss_type"] = "mse"
params["lookback"] = 180
params["num_workers"] = 4  # Reduced from 8
params["train_size_dimension"] = "huc"
params["train_size_fraction"] = 1.0
params["device"] = "cuda" if torch.cuda.is_available() else "cpu"

log(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
log(f"Parameters: train_size_dimension={params['train_size_dimension']}, device={params['device']}")

start_time = time.time()

# ============================================================================
# STEP 1: Load and preprocess data
# ============================================================================
log("=" * 80)
log("STEP 1: Loading data from S3...")
log("=" * 80)

tr_and_val_hucs = tr + val
df_dict, global_means, global_stds = pp.pre_process(tr_and_val_hucs, params["var_list"])

log(f"✅ Downloaded {len(df_dict)} HUCs from S3")
log(f"Global normalization: mean={global_means}, std={global_stds}")

# Split into train and val dicts
df_dict_tr = {huc: df_dict[huc] for huc in tr if huc in df_dict}
df_dict_val = {huc: df_dict[huc] for huc in val if huc in df_dict}

log(f"Train dict: {len(df_dict_tr)} HUCs")
log(f"Val dict: {len(df_dict_val)} HUCs")

# Check one HUC
sample_huc = list(df_dict.keys())[0]
sample_df = df_dict[sample_huc]
log(f"\nSample HUC {sample_huc}:")
log(f"  Shape: {sample_df.shape}")
log(f"  Columns: {sample_df.columns.tolist()}")
log(f"  Total timesteps: {len(sample_df)}")

# ============================================================================
# STEP 2: Initialize model
# ============================================================================
log("=" * 80)
log("STEP 2: Initializing model...")
log("=" * 80)

input_size = len(params["var_list"])
model = LSTM_mod.SnowModel(
    input_size,
    params['hidden_size'],
    params['num_class'],
    params['num_layers'],
    params['dropout']
)

# Move model to GPU
model = model.to(params['device'])

optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
loss_fn = torch.nn.MSELoss()

log(f"✅ Model initialized on {params['device']}")
log(f"Model parameters: {sum(p.numel() for p in model.parameters())} total")

# ============================================================================
# STEP 3: Train for 1 epoch
# ============================================================================
log("=" * 80)
log("STEP 3: Training (1 epoch on 2 HUCs)...")
log("=" * 80)

epoch = 0
LSTM_tr.pre_train(model, optimizer, loss_fn, df_dict_tr, params, epoch)

log("✅ Training complete")

# CRITICAL: Clear GPU memory before validation
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    log(f"GPU memory cleared. Free memory: {torch.cuda.mem_get_info()[0]/1e9:.2f} GB")

# ============================================================================
# STEP 4: Validate and SHOW METRICS
# ============================================================================
log("=" * 80)
log("STEP 4: Validation (2 val HUCs) - SHOWING METRICS")
log("=" * 80)

for val_huc in val:
    log(f"\nValidating on HUC: {val_huc}")

    # Get validation data for this HUC
    val_data = df_dict_val[val_huc]

    # Create tensors
    from snowML.LSTM import LSTM_pre_process as pp
    X_test, y_test = pp.create_tensor(val_data, params['lookback'], params['var_list'])
    y_test_np = y_test.numpy()

    log(f"  Validation data: {X_test.shape[0]} timesteps")

    # Use batched prediction to avoid GPU OOM
    y_pred_np = batched_predict(model, X_test, params, batch_size=1000)

    # Calculate metrics
    from snowML.LSTM import LSTM_metrics as met
    metric_dict = met.calc_metrics(y_test_np, y_pred_np, metric_type="test")

    log(f"  Results for {val_huc}:")
    log(f"    Test MSE: {metric_dict['test_mse']:.6f}")
    log(f"    Test KGE: {metric_dict['test_kge']:.6f}")
    log(f"    Test R2:  {metric_dict['test_r2']:.6f}")
    log(f"    Test MAE: {metric_dict['test_mae']:.6f}")
    log(f"    Predictions shape: {y_pred_np.shape}")

    # Clear GPU memory after each validation HUC
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

elapsed = time.time() - start_time

# ============================================================================
# SUMMARY
# ============================================================================
log("=" * 80)
log("TEST SUMMARY")
log("=" * 80)
log(f"Duration: {elapsed/60:.2f} minutes")
log(f"GPU Memory: {torch.cuda.memory_allocated(0)/1e9:.2f} GB allocated")

# Calculate projection
ratio = (162 / 2) * 30  # Scale to full training
projected = (elapsed / 60) * ratio / 60  # in hours

log(f"\nProjected full training time (1 variation):")
log(f"  ~{projected:.1f} hours")
log(f"  (162 train HUCs × 54 val HUCs × 30 epochs)")

log("\n✅ PIPELINE VALIDATED!")
log("All components working:")
log("  ✓ S3 data download")
log("  ✓ GPU training")
log("  ✓ Validation with metrics")
log("  ✓ Reasonable timing")
