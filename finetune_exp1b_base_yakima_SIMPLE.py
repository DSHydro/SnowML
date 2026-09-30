#!/usr/bin/env python3
"""
Experiment 2: Fine-tune Exp1B Base on Yakima/Naches - SIMPLIFIED VERSION
Uses existing SnowML functions directly

Usage on SageMaker Studio:
    cd /home/sagemaker-user
    conda activate pytorch_p310
    nohup python -u finetune_exp1b_base_yakima_SIMPLE.py > finetune.log 2>&1 &
    tail -f finetune.log
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
import json
import mlflow
import os
from datetime import datetime
import sys

# Add src to path
sys.path.insert(0, '/home/sagemaker-user/src')

# Import SnowML modules
from snowML.LSTM import LSTM_train
from snowML.LSTM import LSTM_model as LSTM_mod
from snowML.LSTM.set_hyperparams import create_hyper_dict
from snowML.datapipe import load_data

print("="*80)
print("EXPERIMENT 2: FINE-TUNING EXP1B BASE ON YAKIMA/NACHES")
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)

# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_DIR = Path("/home/sagemaker-user")
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
OUTPUT_DIR = BASE_DIR / "exp2_finetune_results"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

PRETRAINED_CHECKPOINT = CHECKPOINT_DIR / "Exp1B_Base_3e-4_epoch14.pth"

# Yakima/Naches HUCs (Test Set B - 81 HUCs) - VERIFIED CORRECT LIST
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
FINETUNE_LR = 0.0001
FINETUNE_DROPOUT = 0.2

# MLflow
MLFLOW_URI = "arn:aws:sagemaker:us-west-2:677276086662:mlflow-tracking-server/dawgsML"
os.environ['MLFLOW_TRACKING_ARN'] = MLFLOW_URI

# =============================================================================
# STEP 1: LOAD CHECKPOINT
# =============================================================================

print("\n[1/6] Loading pre-trained checkpoint...")
if not PRETRAINED_CHECKPOINT.exists():
    print(f"   ❌ ERROR: {PRETRAINED_CHECKPOINT} not found!")
    sys.exit(1)

checkpoint = torch.load(PRETRAINED_CHECKPOINT, map_location='cpu')
print(f"   ✅ Checkpoint loaded from epoch {checkpoint.get('epoch', 14)}")
print(f"   📊 Pre-trained Val KGE: {checkpoint.get('val_kge', 0.8084):.4f}")

# =============================================================================
# STEP 2: CREATE SPLITS
# =============================================================================

print(f"\n[2/6] Creating fine-tune splits...")
np.random.seed(42)
shuffled_hucs = np.random.permutation(TEST_B_HUCS)
n_train = int(len(shuffled_hucs) * 0.7)

finetune_train_hucs = shuffled_hucs[:n_train].tolist()  # 57
finetune_val_hucs = shuffled_hucs[n_train:].tolist()    # 24

print(f"   🎯 Train: {len(finetune_train_hucs)} HUCs")
print(f"   🎯 Val:   {len(finetune_val_hucs)} HUCs")

# Save splits
splits_file = OUTPUT_DIR / "finetune_splits.json"
with open(splits_file, 'w') as f:
    json.dump({'train': finetune_train_hucs, 'val': finetune_val_hucs}, f, indent=2)

# =============================================================================
# STEP 3: LOAD DATA
# =============================================================================

print("\n[3/6] Loading HUC data from S3...")
print("   (This will take a few minutes - downloading from S3)")

# Load training data
train_df_dict = {}
for huc_id in finetune_train_hucs:
    try:
        df = load_data.load_single_huc(huc_id)
        if df is not None and len(df) > 0:
            train_df_dict[huc_id] = df
    except Exception as e:
        print(f"   ⚠️  Failed to load train HUC {huc_id}: {str(e)[:50]}")

# Load validation data
val_df_dict = {}
for huc_id in finetune_val_hucs:
    try:
        df = load_data.load_single_huc(huc_id)
        if df is not None and len(df) > 0:
            val_df_dict[huc_id] = df
    except Exception as e:
        print(f"   ⚠️  Failed to load val HUC {huc_id}: {str(e)[:50]}")

print(f"   ✅ Loaded {len(train_df_dict)} train HUCs")
print(f"   ✅ Loaded {len(val_df_dict)} val HUCs")

if len(train_df_dict) < 10:
    print("   ❌ ERROR: Too few training HUCs loaded!")
    sys.exit(1)

# =============================================================================
# STEP 4: SET UP MODEL
# =============================================================================

print("\n[4/6] Initializing model...")

# Parameters
params = create_hyper_dict()
params.update({
    "hidden_size": 64,
    "num_layers": 1,
    "dropout": FINETUNE_DROPOUT,
    "batch_size": 32,
    "lookback": 180,
    "n_epochs": FINETUNE_EPOCHS,
    "learning_rate": FINETUNE_LR,
    "num_workers": 4,
    "train_size_dimension": "huc",
    "train_size_fraction": 1.0,
    "var_list": ["mean_pr", "mean_tair", "Mean Elevation"],
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "loss_type": "mse",
    "mlflow_tracking_uri": MLFLOW_URI,
    "MLFLOW_ON": True,
    "expirement_name": f"Exp2_FineTune_Base_{datetime.now().strftime('%Y%m%d')}",
})

# Create model
n_features = len(params["var_list"])
model = LSTM_mod.SnowModel(
    input_size=n_features,
    hidden_size=params["hidden_size"],
    num_class=1,
    num_layers=params["num_layers"],
    dropout=params["dropout"]
)

# Load pre-trained weights
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(params['device'])

# Optimizer and loss
optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])
loss_fn = torch.nn.MSELoss()

print(f"   ✅ Model ready on {params['device']}")

# =============================================================================
# STEP 5: FINE-TUNING
# =============================================================================

print(f"\n[5/6] Fine-tuning for {FINETUNE_EPOCHS} epochs...")

mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment(params["expirement_name"])

best_val_kge = -999
best_epoch = 0

with mlflow.start_run(run_name="FineTune_Base_Yakima"):

    mlflow.log_params({
        "pretrained_epoch": 14,
        "finetune_n_train": len(train_df_dict),
        "finetune_n_val": len(val_df_dict),
        "finetune_lr": FINETUNE_LR,
        "finetune_dropout": FINETUNE_DROPOUT,
        "finetune_epochs": FINETUNE_EPOCHS,
    })

    for epoch in range(1, FINETUNE_EPOCHS + 1):
        epoch_start = datetime.now()
        print(f"\n   ━━━ Epoch {epoch}/{FINETUNE_EPOCHS} ━━━")

        # Train using existing pre_train function
        print(f"      🏋️  Training on {len(train_df_dict)} HUCs...")
        model = LSTM_train.pre_train(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            df_dict=train_df_dict,
            params=params,
            epoch=epoch
        )

        # Validate using existing evaluate function
        print(f"      🔍 Validating on {len(val_df_dict)} HUCs...")
        val_metrics = LSTM_train.evaluate(
            model_dawgs=model,
            df_dict=val_df_dict,
            params=params,
            epoch=epoch
        )

        val_kge = val_metrics['test_kge']
        val_mse = val_metrics['test_mse']
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
            print(f"      ✅ NEW BEST! Saved")

    mlflow.log_metrics({"best_val_kge": best_val_kge, "best_epoch": best_epoch})

print(f"\n   🏆 Best: KGE {best_val_kge:.4f} at epoch {best_epoch}")

# =============================================================================
# STEP 6: FINAL EVALUATION
# =============================================================================

print(f"\n[6/6] Final evaluation on all {len(TEST_B_HUCS)} Test B HUCs...")

# Load all Test B data
print("   Loading all Test B HUCs...")
test_df_dict = {}
for huc_id in TEST_B_HUCS:
    try:
        df = load_data.load_single_huc(huc_id)
        if df is not None and len(df) > 0:
            test_df_dict[huc_id] = df
    except:
        pass

print(f"   ✅ Loaded {len(test_df_dict)} test HUCs")

# Load best checkpoint
best_checkpoint = torch.load(OUTPUT_DIR / f"FineTune_Base_epoch{best_epoch}.pth")
model.load_state_dict(best_checkpoint['model_state_dict'])

# Evaluate
test_metrics = LSTM_train.evaluate(
    model_dawgs=model,
    df_dict=test_df_dict,
    params=params,
    epoch=best_epoch
)

final_test_kge = test_metrics['test_kge']
final_test_mse = test_metrics['test_mse']

print(f"\n   📊 Final Test B Results:")
print(f"      KGE: {final_test_kge:.4f}")
print(f"      MSE: {final_test_mse:.6f}")

# Comparison
pretrained_kge = 0.7603
improvement = final_test_kge - pretrained_kge
improvement_pct = (improvement / pretrained_kge) * 100

print(f"\n{'='*80}")
print(f"COMPARISON:")
print(f"   Pre-trained (Exp1B):  {pretrained_kge:.4f}")
print(f"   Fine-tuned (Exp2):    {final_test_kge:.4f}")
print(f"   Improvement:          {improvement:+.4f} ({improvement_pct:+.1f}%)")

if improvement > 0.02:
    print(f"   ✅ Fine-tuning SIGNIFICANTLY IMPROVED performance!")
elif improvement > 0:
    print(f"   ⚠️  Fine-tuning had minimal improvement")
else:
    print(f"   ❌ Fine-tuning did not improve (possible overfit)")

print(f"{'='*80}")

# Save summary
summary = {
    'pretrained_test_b_kge': pretrained_kge,
    'finetuned_test_b_kge': final_test_kge,
    'improvement_absolute': improvement,
    'improvement_percent': improvement_pct,
    'best_epoch': best_epoch,
    'best_val_kge': best_val_kge,
}

with open(OUTPUT_DIR / "finetune_summary.json", 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\n✅ COMPLETE! Results in: {OUTPUT_DIR}/")
print(f"⏱️  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
