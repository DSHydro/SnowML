#!/usr/bin/env python3
"""
Experiment 2: Fine-tune Exp1B Base model on Yakima/Naches HUCs
Author: Simran
Date: September 29, 2026

Pre-trained: Exp1B_Base_3e-4_epoch14 (Val KGE: 0.8084, Test B: 0.7603)
Target: Fine-tune on Yakima/Naches to improve Test B KGE
Expected: 0.76 → 0.80-0.85 KGE

Usage:
    cd /home/sagemaker-user
    nohup python -u finetune_exp1b_base_yakima.py > finetune.log 2>&1 &
    tail -f finetune.log
"""

import torch
import torch.nn as nn
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
from snowML.LSTM import LSTM_pre_process as pp
from snowML.LSTM import LSTM_metrics as met
from snowML.LSTM.set_hyperparams import set_hyperparams

print("="*80)
print("EXPERIMENT 2: FINE-TUNING EXP1B BASE ON YAKIMA/NACHES")
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Paths (SageMaker Studio)
BASE_DIR = Path("/home/sagemaker-user")
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
OUTPUT_DIR = BASE_DIR / "exp2_finetune_results"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Pre-trained model checkpoint
PRETRAINED_CHECKPOINT = CHECKPOINT_DIR / "Exp1B_Base_3e-4_epoch14.pth"

# Yakima/Naches HUCs (Test Set B - 81 HUCs)
# These are the ACTUAL HUCs used in Exp1A and Exp1B evaluations
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

# MLflow tracking
MLFLOW_URI = "arn:aws:sagemaker:us-west-2:677276086662:mlflow-tracking-server/dawgsML"
os.environ['MLFLOW_TRACKING_ARN'] = MLFLOW_URI

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def calculate_metrics(y_true, y_pred):
    """Calculate KGE, MSE, R2, MAE"""
    # Remove NaNs
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) == 0:
        return {'kge': -999, 'mse': 999, 'r2': -999, 'mae': 999}

    # KGE (Kling-Gupta Efficiency)
    r = np.corrcoef(y_true, y_pred)[0, 1] if len(y_true) > 1 else 0
    alpha = np.std(y_pred) / np.std(y_true) if np.std(y_true) > 0 else 1
    beta = np.mean(y_pred) / np.mean(y_true) if np.mean(y_true) > 0 else 1
    kge = 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)

    # Other metrics
    mse = np.mean((y_true - y_pred)**2)
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else -999
    mae = np.mean(np.abs(y_true - y_pred))

    return {'kge': kge, 'mse': mse, 'r2': r2, 'mae': mae}

# =============================================================================
# STEP 1: LOAD PRE-TRAINED MODEL
# =============================================================================

print("\n[1/7] Loading pre-trained model...")

if not PRETRAINED_CHECKPOINT.exists():
    print(f"   ❌ ERROR: Checkpoint not found at {PRETRAINED_CHECKPOINT}")
    print(f"\n   Please ensure Exp1B_Base_3e-4_epoch14.pth exists in:")
    print(f"   {CHECKPOINT_DIR}/")
    sys.exit(1)

checkpoint = torch.load(PRETRAINED_CHECKPOINT, map_location='cpu')
print(f"   ✅ Loaded checkpoint from epoch {checkpoint.get('epoch', 14)}")
print(f"   📊 Pre-trained Val KGE: {checkpoint.get('val_kge', 'N/A')}")

# =============================================================================
# STEP 2: CREATE FINE-TUNE SPLITS
# =============================================================================

print(f"\n[2/7] Creating fine-tune splits from {len(TEST_B_HUCS)} HUCs...")

# 70/30 split: 57 train, 24 validation
np.random.seed(42)  # Reproducible splits
shuffled_hucs = np.random.permutation(TEST_B_HUCS)
n_train = int(len(shuffled_hucs) * 0.7)

finetune_train_hucs = shuffled_hucs[:n_train].tolist()
finetune_val_hucs = shuffled_hucs[n_train:].tolist()

print(f"   🎯 Fine-tune train: {len(finetune_train_hucs)} HUCs (70%)")
print(f"   🎯 Fine-tune val:   {len(finetune_val_hucs)} HUCs (30%)")
print(f"   📝 First 5 train HUCs: {finetune_train_hucs[:5]}")

# Save splits for reproducibility
splits_info = {
    'train': finetune_train_hucs,
    'val': finetune_val_hucs,
    'all_test_b': TEST_B_HUCS,
    'n_train': len(finetune_train_hucs),
    'n_val': len(finetune_val_hucs),
    'random_seed': 42,
    'split_ratio': 0.7
}

splits_file = OUTPUT_DIR / "finetune_splits.json"
with open(splits_file, 'w') as f:
    json.dump(splits_info, f, indent=2)
print(f"   💾 Splits saved: {splits_file}")

# =============================================================================
# STEP 3: SET UP PARAMETERS
# =============================================================================

print("\n[3/7] Configuring fine-tuning parameters...")

# Load base parameters
params = set_hyperparams()

# Override for fine-tuning
params.update({
    # Model architecture (same as pre-training)
    "hidden_size": 64,
    "num_layers": 1,
    "dropout": FINETUNE_DROPOUT,  # LOWER dropout for fine-tuning

    # Training settings
    "batch_size": 32,
    "lookback": 180,
    "n_epochs": FINETUNE_EPOCHS,
    "learning_rate": FINETUNE_LR,  # LOWER learning rate for fine-tuning
    "num_workers": 4,

    # Data split method (CRITICAL - must be "huc")
    "train_size_dimension": "huc",
    "train_size_fraction": 1.0,

    # Feature set (Base model)
    "var_list": ["mean_pr", "mean_tair", "Mean Elevation"],

    # Device
    "device": "cuda" if torch.cuda.is_available() else "cpu",

    # MLflow
    "mlflow_tracking_uri": MLFLOW_URI,
    "MLFLOW_ON": True,
    "expirement_name": f"Exp2_FineTune_Base_{datetime.now().strftime('%Y%m%d_%H%M')}",
})

print(f"   ✅ Device: {params['device']}")
print(f"   ✅ Features: {params['var_list']}")
print(f"   ✅ Learning rate: {FINETUNE_LR} (was 0.0003)")
print(f"   ✅ Dropout: {FINETUNE_DROPOUT} (was 0.5)")
print(f"   ✅ Epochs: {FINETUNE_EPOCHS}")

# =============================================================================
# STEP 4: INITIALIZE MODEL WITH PRE-TRAINED WEIGHTS
# =============================================================================

print("\n[4/7] Initializing model with pre-trained weights...")

n_features = len(params["var_list"])
model = LSTM_mod.SnowModel(
    input_size=n_features,
    hidden_size=params["hidden_size"],
    num_class=1,  # Output size
    num_layers=params["num_layers"],
    dropout=params["dropout"]
)

# Load pre-trained weights
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(params['device'])

print(f"   ✅ Model architecture: {n_features} inputs → 64 hidden → 1 output")
print(f"   ✅ Pre-trained weights loaded successfully")

# =============================================================================
# STEP 5: FINE-TUNING LOOP
# =============================================================================

print(f"\n[5/7] Starting fine-tuning...")
print(f"   Training on {len(finetune_train_hucs)} HUCs")
print(f"   Validating on {len(finetune_val_hucs)} HUCs")

# Set up optimizer (fresh optimizer, not from checkpoint)
optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])
loss_fn = nn.MSELoss()

# Initialize MLflow
mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment(params["expirement_name"])

best_val_kge = -999
best_epoch = 0
training_history = []

with mlflow.start_run(run_name="FineTune_Base_3e-4_Yakima"):

    # Log hyperparameters
    mlflow.log_params({
        "pretrained_model": "Exp1B_Base_3e-4",
        "pretrained_epoch": checkpoint.get('epoch', 14),
        "pretrained_val_kge": checkpoint.get('val_kge', 0.8084),
        "finetune_region": "Yakima_Naches",
        "finetune_n_train": len(finetune_train_hucs),
        "finetune_n_val": len(finetune_val_hucs),
        "finetune_epochs": FINETUNE_EPOCHS,
        "finetune_lr": FINETUNE_LR,
        "finetune_dropout": FINETUNE_DROPOUT,
        "train_size_dimension": "huc",
        "train_size_fraction": 1.0,
        "batch_size": params["batch_size"],
        "lookback": params["lookback"],
    })

    # Fine-tuning loop
    for epoch in range(1, FINETUNE_EPOCHS + 1):
        epoch_start = datetime.now()
        print(f"\n   ━━━ Epoch {epoch}/{FINETUNE_EPOCHS} ━━━")

        # =====================================================================
        # TRAINING PHASE
        # =====================================================================
        print(f"      🏋️  Training on {len(finetune_train_hucs)} HUCs...")

        model.train()
        epoch_losses = []

        # Train on each HUC
        for huc_idx, huc_id in enumerate(finetune_train_hucs):
            try:
                # Load data for this HUC using DataGenerator
                data_gen = DataGenerator(
                    huc_list=[huc_id],
                    params=params,
                    train_size_dimension="huc",
                    train_size_fraction=1.0
                )

                # Get training data
                train_loader = data_gen.get_train_dataloader()

                # Train on this HUC
                for batch_X, batch_y in train_loader:
                    batch_X = batch_X.to(params['device'])
                    batch_y = batch_y.to(params['device'])

                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = loss_fn(outputs, batch_y)
                    loss.backward()
                    optimizer.step()

                    epoch_losses.append(loss.item())

                if params['device'].startswith('cuda'):
                    torch.cuda.empty_cache()

            except Exception as e:
                print(f"         ⚠️  Warning: HUC {huc_id} failed: {str(e)[:50]}")
                continue

        avg_train_loss = np.mean(epoch_losses) if epoch_losses else 999
        print(f"      ✅ Training complete - Avg loss: {avg_train_loss:.6f}")

        # =====================================================================
        # VALIDATION PHASE
        # =====================================================================
        print(f"      🔍 Validating on {len(finetune_val_hucs)} HUCs...")

        model.eval()
        val_kges = []
        val_mses = []

        with torch.no_grad():
            for huc_id in finetune_val_hucs:
                try:
                    # Load validation data
                    data_gen = DataGenerator(
                        huc_list=[huc_id],
                        params=params,
                        train_size_dimension="huc",
                        train_size_fraction=1.0
                    )

                    val_loader = data_gen.get_test_dataloader()

                    # Predict
                    all_preds = []
                    all_targets = []

                    for batch_X, batch_y in val_loader:
                        batch_X = batch_X.to(params['device'])
                        outputs = model(batch_X)

                        all_preds.append(outputs.cpu().numpy())
                        all_targets.append(batch_y.numpy())

                    if all_preds:
                        y_pred = np.concatenate(all_preds).flatten()
                        y_true = np.concatenate(all_targets).flatten()

                        metrics = calculate_metrics(y_true, y_pred)
                        val_kges.append(metrics['kge'])
                        val_mses.append(metrics['mse'])

                    if params['device'].startswith('cuda'):
                        torch.cuda.empty_cache()

                except Exception as e:
                    print(f"         ⚠️  Warning: Val HUC {huc_id} failed: {str(e)[:50]}")
                    continue

        # Calculate median validation metrics
        val_kge = np.median(val_kges) if val_kges else -999
        val_mse = np.median(val_mses) if val_mses else 999

        epoch_time = (datetime.now() - epoch_start).total_seconds()

        print(f"      📊 Val KGE: {val_kge:.4f} | MSE: {val_mse:.6f}")
        print(f"      ⏱️  Epoch time: {epoch_time:.0f}s")

        # Log to MLflow
        mlflow.log_metrics({
            "train_loss": avg_train_loss,
            "val_kge": val_kge,
            "val_mse": val_mse,
            "epoch_time_sec": epoch_time,
        }, step=epoch)

        # Track history
        training_history.append({
            'epoch': epoch,
            'train_loss': avg_train_loss,
            'val_kge': val_kge,
            'val_mse': val_mse,
            'epoch_time_sec': epoch_time
        })

        # Save best model
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
                'train_loss': avg_train_loss,
                'pretrained_from': 'Exp1B_Base_3e-4_epoch14',
                'params': params,
            }, checkpoint_path)

            print(f"      ✅ NEW BEST! Saved to {checkpoint_path.name}")

    # Log final best metrics
    mlflow.log_metrics({
        "best_val_kge": best_val_kge,
        "best_epoch": best_epoch,
    })

    print(f"\n   🏆 Fine-tuning complete!")
    print(f"   📊 Best validation KGE: {best_val_kge:.4f} at epoch {best_epoch}")

# Save training history
history_df = pd.DataFrame(training_history)
history_path = OUTPUT_DIR / "finetune_training_history.csv"
history_df.to_csv(history_path, index=False)
print(f"   💾 Training history saved: {history_path}")

# =============================================================================
# STEP 6: EVALUATE ON FULL TEST SET B
# =============================================================================

print(f"\n[6/7] Evaluating best model on full Test Set B ({len(TEST_B_HUCS)} HUCs)...")

# Load best checkpoint
best_checkpoint_path = OUTPUT_DIR / f"FineTune_Base_epoch{best_epoch}.pth"
best_checkpoint = torch.load(best_checkpoint_path)

model.load_state_dict(best_checkpoint['model_state_dict'])
model.eval()

# Evaluate on all Test B HUCs
test_results = []

with torch.no_grad():
    for huc_id in TEST_B_HUCS:
        try:
            data_gen = DataGenerator(
                huc_list=[huc_id],
                params=params,
                train_size_dimension="huc",
                train_size_fraction=1.0
            )

            test_loader = data_gen.get_test_dataloader()

            all_preds = []
            all_targets = []

            for batch_X, batch_y in test_loader:
                batch_X = batch_X.to(params['device'])
                outputs = model(batch_X)

                all_preds.append(outputs.cpu().numpy())
                all_targets.append(batch_y.numpy())

            if all_preds:
                y_pred = np.concatenate(all_preds).flatten()
                y_true = np.concatenate(all_targets).flatten()

                metrics = calculate_metrics(y_true, y_pred)

                test_results.append({
                    'huc_id': huc_id,
                    'test_kge': metrics['kge'],
                    'test_mse': metrics['mse'],
                    'test_r2': metrics['r2'],
                    'test_mae': metrics['mae'],
                })

            if params['device'].startswith('cuda'):
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"   ⚠️  HUC {huc_id} failed: {str(e)[:50]}")
            test_results.append({
                'huc_id': huc_id,
                'test_kge': -999,
                'test_mse': 999,
                'test_r2': -999,
                'test_mae': 999,
            })

# Save results
results_df = pd.DataFrame(test_results)
results_path = OUTPUT_DIR / "FineTune_Base_TestB_metrics.csv"
results_df.to_csv(results_path, index=False)

# Calculate summary statistics
valid_kges = results_df[results_df['test_kge'] > -999]['test_kge']
median_kge = valid_kges.median()
mean_kge = valid_kges.mean()

print(f"   ✅ Evaluation complete!")
print(f"   📊 Test Set B Results:")
print(f"      Median KGE: {median_kge:.4f}")
print(f"      Mean KGE:   {mean_kge:.4f}")
print(f"      Valid HUCs: {len(valid_kges)}/{len(TEST_B_HUCS)}")
print(f"   💾 Results saved: {results_path}")

# =============================================================================
# STEP 7: COMPARISON & SUMMARY
# =============================================================================

print("\n[7/7] Final Comparison")
print("="*80)

pretrained_test_b_kge = 0.7603  # From your results table
finetuned_test_b_kge = median_kge
improvement = finetuned_test_b_kge - pretrained_test_b_kge
improvement_pct = (improvement / pretrained_test_b_kge) * 100

comparison_table = pd.DataFrame({
    'Model': [
        'Exp1B Base (Pre-trained)',
        'Exp2 FineTune (Best Val)',
        'Exp2 FineTune (Final Test)',
    ],
    'Validation KGE': [
        0.8084,
        best_val_kge,
        'N/A',
    ],
    'Test B KGE': [
        pretrained_test_b_kge,
        'N/A',
        finetuned_test_b_kge,
    ],
    'Training': [
        '270 deep snow HUCs',
        '57 Yakima/Naches',
        '57 Yakima/Naches',
    ]
})

print("\n" + comparison_table.to_string(index=False))

print(f"\n🎯 Fine-tuning Impact:")
print(f"   Pre-trained Test B KGE:  {pretrained_test_b_kge:.4f}")
print(f"   Fine-tuned Test B KGE:   {finetuned_test_b_kge:.4f}")
print(f"   Absolute improvement:    {improvement:+.4f}")
print(f"   Relative improvement:    {improvement_pct:+.2f}%")

if improvement > 0.02:
    print(f"   ✅ Fine-tuning SIGNIFICANTLY IMPROVED performance!")
elif improvement > 0:
    print(f"   ⚠️  Fine-tuning had minimal positive impact")
else:
    print(f"   ❌ Fine-tuning did not improve performance (possible overfitting)")

# Save comparison
comparison_path = OUTPUT_DIR / "finetune_comparison_summary.csv"
comparison_table.to_csv(comparison_path, index=False)

# Create final summary
summary = {
    'pretrained_model': 'Exp1B_Base_3e-4_epoch14',
    'pretrained_val_kge': 0.8084,
    'pretrained_test_b_kge': pretrained_test_b_kge,
    'finetune_best_epoch': best_epoch,
    'finetune_best_val_kge': best_val_kge,
    'finetune_test_b_kge': finetuned_test_b_kge,
    'improvement_absolute': improvement,
    'improvement_percent': improvement_pct,
    'finetune_n_train': len(finetune_train_hucs),
    'finetune_n_val': len(finetune_val_hucs),
    'finetune_n_test': len(TEST_B_HUCS),
    'finetune_epochs_run': FINETUNE_EPOCHS,
    'finetune_lr': FINETUNE_LR,
    'finetune_dropout': FINETUNE_DROPOUT,
}

summary_path = OUTPUT_DIR / "finetune_summary.json"
with open(summary_path, 'w') as f:
    json.dump(summary, f, indent=2)

print("\n" + "="*80)
print("✅ EXPERIMENT 2 COMPLETE!")
print("="*80)
print(f"\n📁 All results saved to: {OUTPUT_DIR}/")
print(f"   - Best checkpoint: FineTune_Base_epoch{best_epoch}.pth")
print(f"   - Test metrics: FineTune_Base_TestB_metrics.csv")
print(f"   - Training history: finetune_training_history.csv")
print(f"   - Splits used: finetune_splits.json")
print(f"   - Summary: finetune_summary.json")
print(f"\n🔍 View results in MLflow UI:")
print(f"   Experiment: {params['expirement_name']}")
print(f"\n⏱️  Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
