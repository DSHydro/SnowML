#!/usr/bin/env python3
"""
Experiment 2: Fine-tune Exp1B Base on Yakima/Naches
CORRECT VERSION - Based on actual working train_exp1b_full.py

Usage on SageMaker Studio:
    cd /home/sagemaker-user
    conda activate pytorch_p310
    nohup python -u finetune_exp1b_base_CORRECT.py > finetune.log 2>&1 &
    tail -f finetune.log
"""

import os
import json
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import torch
from torch import optim
import mlflow

# Add src to Python path so snowML can be imported
sys.path.insert(0, '/home/sagemaker-user/src')
sys.path.insert(0, '/home/sagemaker-user')

# CORRECT imports based on train_exp1b_full.py
from snowML.LSTM import LSTM_train as LSTM_tr
from snowML.LSTM import LSTM_model as LSTM_mod
from snowML.LSTM import set_hyperparams as sh
from snowML.LSTM import LSTM_pre_process as pp

print("=" * 80)
print("EXPERIMENT 2: FINE-TUNING EXP1B BASE ON YAKIMA/NACHES")
print("=" * 80)
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_DIR = Path("/home/sagemaker-user")
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
OUTPUT_DIR = BASE_DIR / "exp2_finetune_results"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

PRETRAINED_CHECKPOINT = CHECKPOINT_DIR / "Exp1B_Base_3e-4_epoch14.pth"

# Yakima/Naches HUCs (Test Set B - 81 HUCs) - VERIFIED CORRECT
TEST_B_HUCS = [
    "170300010101", "170300010102", "170300010103", "170300010104", "170300010105",
    "170300010106", "170300010201", "170300010202", "170300010203", "170300010204",
    "170300010205", "170300010301", "170300010302", "170300010303", "170300010304",
    "170300010305", "170300010306", "170300010307", "170300010401", "170300010402",
    "170300010403", "170300010404", "170300010405", "170300010406", "170300010407",
    "170300010408", "170300010409", "170300010501", "170300010502", "170300010503",
    "170300010504", "170300010505", "170300010506", "170300010507", "170300010508",
    "170300010509", "170300010510", "170300010511", "170300010601", "170300010602",
    "170300010603", "170300010604", "170300010605", "170300010701", "170300010702",
    "170300010703", "170300010704", "170300010705", "170300010706", "170300010707",
    "170300010708", "170300010709", "170300020101", "170300020102", "170300020103",
    "170300020104", "170300020105", "170300020106", "170300020107", "170300020108",
    "170300020109", "170300020201", "170300020202", "170300020203", "170300020204",
    "170300020205", "170300020206", "170300020207", "170300020208", "170300020301",
    "170300020302", "170300020303", "170300020304", "170300020305", "170300020306",
    "170300020307", "170300020308", "170300020309", "170300020310", "170300020311",
    "170300020312"
]

# Fine-tuning hyperparameters
FINETUNE_EPOCHS = 10
FINETUNE_LR = 0.0001  # Lower than pre-training (0.0003)
FINETUNE_DROPOUT = 0.2  # Lower than pre-training (0.5)

# MLflow
MLFLOW_URI = "arn:aws:sagemaker:us-west-2:677276086662:mlflow-tracking-server/dawgsML"
os.environ['MLFLOW_TRACKING_ARN'] = MLFLOW_URI

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"   📍 Device: {device}")

# =============================================================================
# STEP 1: LOAD CHECKPOINT
# =============================================================================

print("\n[1/7] Loading pre-trained checkpoint...")
if not PRETRAINED_CHECKPOINT.exists():
    print(f"   ❌ ERROR: {PRETRAINED_CHECKPOINT} not found!")
    sys.exit(1)

checkpoint = torch.load(PRETRAINED_CHECKPOINT, map_location='cpu')
print(f"   ✅ Loaded from epoch {checkpoint.get('epoch', 14)}")
print(f"   📊 Pre-trained Val KGE: {checkpoint.get('val_kge', 0.8084):.4f}")

# =============================================================================
# STEP 2: CREATE SPLITS
# =============================================================================

print(f"\n[2/7] Creating fine-tune splits from {len(TEST_B_HUCS)} HUCs...")
np.random.seed(42)
shuffled_hucs = np.random.permutation(TEST_B_HUCS)
n_train = int(len(shuffled_hucs) * 0.7)

finetune_train_hucs = shuffled_hucs[:n_train].tolist()  # 57
finetune_val_hucs = shuffled_hucs[n_train:].tolist()    # 24

print(f"   🎯 Train: {len(finetune_train_hucs)} HUCs")
print(f"   🎯 Val:   {len(finetune_val_hucs)} HUCs")

# Save splits
with open(OUTPUT_DIR / "finetune_splits.json", 'w') as f:
    json.dump({
        'train': finetune_train_hucs,
        'val': finetune_val_hucs,
        'all_test_b': TEST_B_HUCS
    }, f, indent=2)

# =============================================================================
# STEP 3: LOAD DATA (using pp.pre_process like train_exp1b_full.py)
# =============================================================================

print("\n[3/7] Loading data from S3...")
print("   (Downloading from S3 - will take a few minutes)")

# Variables for Base model
var_list = ["mean_pr", "mean_tair", "Mean Elevation"]

# Load all Test B HUCs at once (this is how train_exp1b_full.py does it)
df_dict, global_means, global_stds = pp.pre_process(TEST_B_HUCS, var_list)
print(f"   ✅ Loaded {len(df_dict)} HUCs")

# Split into train and val
df_dict_train = {huc: df_dict[huc] for huc in finetune_train_hucs if huc in df_dict}
df_dict_val = {huc: df_dict[huc] for huc in finetune_val_hucs if huc in df_dict}

print(f"   ✅ Training data: {len(df_dict_train)} HUCs")
print(f"   ✅ Validation data: {len(df_dict_val)} HUCs")

if len(df_dict_train) < 10:
    print("   ❌ ERROR: Too few training HUCs loaded!")
    sys.exit(1)

# =============================================================================
# STEP 4: PARAMETERS
# =============================================================================

print("\n[4/7] Setting up parameters...")

# Create base params (like train_exp1b_full.py)
params = sh.create_hyper_dict()

# Override for fine-tuning
params.update({
    "hidden_size": 64,
    "num_layers": 1,
    "dropout": FINETUNE_DROPOUT,
    "batch_size": 32,
    "lookback": 180,
    "n_epochs": FINETUNE_EPOCHS,
    "learning_rate": FINETUNE_LR,
    "num_workers": 4,
    "train_size_dimension": "huc",  # CRITICAL
    "train_size_fraction": 1.0,     # CRITICAL
    "var_list": var_list,
    "device": device,
    "loss_type": "mse",
    "mlflow_tracking_uri": MLFLOW_URI,
    "MLFLOW_ON": True,
    "expirement_name": f"Exp2_FineTune_Base_{datetime.now().strftime('%Y%m%d_%H%M')}",
})

print(f"   ✅ Learning rate: {FINETUNE_LR}")
print(f"   ✅ Dropout: {FINETUNE_DROPOUT}")
print(f"   ✅ Epochs: {FINETUNE_EPOCHS}")

# =============================================================================
# STEP 5: INITIALIZE MODEL
# =============================================================================

print("\n[5/7] Initializing model...")

n_features = len(var_list)
model = LSTM_mod.SnowModel(
    input_size=n_features,
    hidden_size=params["hidden_size"],
    num_class=1,
    num_layers=params["num_layers"],
    dropout=params["dropout"]
)

# Load pre-trained weights
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=params["learning_rate"])
loss_fn = torch.nn.MSELoss()

print(f"   ✅ Model ready: {n_features} → 64 → 1")

# =============================================================================
# STEP 6: FINE-TUNING
# =============================================================================

print(f"\n[6/7] Fine-tuning for {FINETUNE_EPOCHS} epochs...")

mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment(params["expirement_name"])

best_val_kge = -999
best_epoch = 0

with mlflow.start_run(run_name="FineTune_Base_Yakima"):

    mlflow.log_params({
        "pretrained_epoch": 14,
        "pretrained_val_kge": 0.8084,
        "finetune_n_train": len(df_dict_train),
        "finetune_n_val": len(df_dict_val),
        "finetune_lr": FINETUNE_LR,
        "finetune_dropout": FINETUNE_DROPOUT,
        "finetune_epochs": FINETUNE_EPOCHS,
    })

    for epoch in range(1, FINETUNE_EPOCHS + 1):
        epoch_start = datetime.now()
        print(f"\n   ━━━ Epoch {epoch}/{FINETUNE_EPOCHS} ━━━")

        # Train (using existing LSTM_tr.pre_train function)
        # Note: pre_train modifies model in-place, doesn't return it
        print(f"      🏋️  Training on {len(df_dict_train)} HUCs...")
        LSTM_tr.pre_train(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            df_dict=df_dict_train,
            params=params,
            epoch=epoch
        )

        # Validate (using existing LSTM_tr.evaluate function)
        print(f"      🔍 Validating on {len(df_dict_val)} HUCs...")
        # evaluate returns: (kge_tr, metric_dict_test, metric_dict_te_recur, metric_dict_train, data, y_te_true, y_te_pred, y_te_pred_recur, train_size)
        _, metric_dict_test, _, _, _, _, _, _, _ = LSTM_tr.evaluate(
            model_dawgs=model,
            df_dict=df_dict_val,
            params=params,
            epoch=epoch
        )

        val_kge = metric_dict_test['test_kge']
        val_mse = metric_dict_test['test_mse']
        epoch_time = (datetime.now() - epoch_start).total_seconds()

        print(f"      📊 Val KGE: {val_kge:.4f} | MSE: {val_mse:.6f}")
        print(f"      ⏱️  Time: {epoch_time:.0f}s")

        mlflow.log_metrics({
            "val_kge": val_kge,
            "val_mse": val_mse,
            "epoch_time": epoch_time
        }, step=epoch)

        # Save best
        if val_kge > best_val_kge:
            best_val_kge = val_kge
            best_epoch = epoch

            checkpoint_path = OUTPUT_DIR / f"FineTune_Base_epoch{epoch}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_kge': val_kge,
                'val_mse': val_mse,
                'params': params,
            }, checkpoint_path)
            print(f"      ✅ NEW BEST! Saved to epoch{epoch}.pth")

    mlflow.log_metrics({
        "best_val_kge": best_val_kge,
        "best_epoch": best_epoch
    })

print(f"\n   🏆 Fine-tuning complete!")
print(f"   📊 Best: KGE {best_val_kge:.4f} at epoch {best_epoch}")

# =============================================================================
# STEP 7: FINAL EVALUATION ON ALL TEST B
# =============================================================================

print(f"\n[7/7] Final evaluation on all {len(TEST_B_HUCS)} Test B HUCs...")

# Load best checkpoint
best_checkpoint = torch.load(OUTPUT_DIR / f"FineTune_Base_epoch{best_epoch}.pth")
model.load_state_dict(best_checkpoint['model_state_dict'])

# Evaluate on ALL Test B HUCs
_, metric_dict_test, _, _, _, _, _, _, _ = LSTM_tr.evaluate(
    model_dawgs=model,
    df_dict=df_dict,  # All Test B HUCs
    params=params,
    epoch=best_epoch
)

final_test_kge = metric_dict_test['test_kge']
final_test_mse = metric_dict_test['test_mse']

print(f"\n   📊 Final Test B Results:")
print(f"      KGE: {final_test_kge:.4f}")
print(f"      MSE: {final_test_mse:.6f}")

# =============================================================================
# COMPARISON
# =============================================================================

pretrained_kge = 0.7603
improvement = final_test_kge - pretrained_kge
improvement_pct = (improvement / pretrained_kge) * 100

print(f"\n{'='*80}")
print("COMPARISON:")
print(f"   Pre-trained (Exp1B Base):  {pretrained_kge:.4f}")
print(f"   Fine-tuned (Exp2):         {final_test_kge:.4f}")
print(f"   Improvement:               {improvement:+.4f} ({improvement_pct:+.1f}%)")

if improvement > 0.02:
    print(f"   ✅ Fine-tuning SIGNIFICANTLY IMPROVED!")
elif improvement > 0:
    print(f"   ⚠️  Fine-tuning had minimal improvement")
else:
    print(f"   ❌ Fine-tuning did not improve")

print(f"{'='*80}")

# Save summary
summary = {
    'pretrained_test_b_kge': pretrained_kge,
    'finetuned_test_b_kge': final_test_kge,
    'improvement_absolute': improvement,
    'improvement_percent': improvement_pct,
    'best_epoch': best_epoch,
    'best_val_kge': best_val_kge,
    'finetune_epochs': FINETUNE_EPOCHS,
    'finetune_lr': FINETUNE_LR,
    'finetune_dropout': FINETUNE_DROPOUT,
}

with open(OUTPUT_DIR / "finetune_summary.json", 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\n✅ COMPLETE! Results saved to: {OUTPUT_DIR}/")
print(f"   - Checkpoints: FineTune_Base_epoch*.pth")
print(f"   - Summary: finetune_summary.json")
print(f"   - Splits: finetune_splits.json")
print(f"\n⏱️  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
