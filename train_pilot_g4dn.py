#!/usr/bin/env python3
"""
PILOT TEST - g4dn.2xlarge (1 GPU)
==================================
Hardware: 1× NVIDIA T4 GPU (16GB)
Cost: ~$0.75/hour
Duration: ~30-45 minutes
Purpose: Test setup before running full experiments

Tests:
- AWS S3 data access
- PyTorch + CUDA setup
- MLflow logging
- Single model variation training
- Validates everything works before spending $20-25 on full run
"""

import os
import json
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import torch
from torch import optim
import mlflow

print("=" * 80)
print("PILOT TEST - Experiment 1B")
print("Testing on g4dn.2xlarge (1 GPU)")
print("=" * 80)
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Check CUDA availability
if not torch.cuda.is_available():
    print("❌ ERROR: CUDA not available! This script requires GPU.")
    print(f"   PyTorch version: {torch.__version__}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    sys.exit(1)

print(f"\n✅ GPU Setup:")
print(f"   PyTorch version: {torch.__version__}")
print(f"   CUDA version: {torch.version.cuda}")
print(f"   Available GPUs: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")

# Import SnowML modules
from snowML.LSTM import LSTM_train as LSTM_tr
from snowML.LSTM import LSTM_model as LSTM_mod

# ============================================================================
# PILOT CONFIGURATION (minimal for testing)
# ============================================================================

PARAMS = {
    "hidden_size": 64,
    "num_layers": 1,
    "dropout": 0.3,
    "batch_size": 32,
    "lookback": 180,
    "n_epochs": 5,  # Just 5 epochs for pilot
    "num_workers": 2,
    "num_class": 1,
    "loss_type": "mse",
    "recursive_predict": False,
    "lag_days": 30,
    "lag_swe_var_idx": 3,
    "filter_dates": ["1984-10-01", "2021-09-30"],
    "train_size_dimension": "time",
    "train_size_fraction": 0.67,
    "learning_rate": 0.001,
    "input_vars": ["mean_tair", "mean_pr"],  # Base model only
    "run_name": "Exp1B_Pilot_Test",
    "expirement_name": "Exp1B_Pilot",
    "mlflow_tracking_uri": "sqlite:///mlflow.db",
    "MLFLOW_ON": True,
    "device": torch.device("cuda:0")
}

# Load HUC splits (use subset for pilot)
DATA_DIR = Path("data")
with open(DATA_DIR / "exp1b_train_hucs.txt") as f:
    ALL_TRAIN_HUCS = [line.strip() for line in f]
with open(DATA_DIR / "exp1b_validation_hucs.txt") as f:
    ALL_VAL_HUCS = [line.strip() for line in f]

# Use only first 20 HUCs for pilot test
TRAIN_HUCS = ALL_TRAIN_HUCS[:20]
VAL_HUCS = ALL_VAL_HUCS[:10]

print(f"\n📊 Pilot Configuration:")
print(f"   Training HUCs: {len(TRAIN_HUCS)} (subset of {len(ALL_TRAIN_HUCS)})")
print(f"   Validation HUCs: {len(VAL_HUCS)} (subset of {len(ALL_VAL_HUCS)})")
print(f"   Epochs: {PARAMS['n_epochs']} (pilot only)")
print(f"   Features: {PARAMS['input_vars']}")

# ============================================================================
# LOAD DATA
# ============================================================================

print(f"\n{'='*80}")
print("STEP 1: LOADING DATA FROM S3")
print(f"{'='*80}")

print("Loading training data...")
train_dfs = []
for i, huc in enumerate(TRAIN_HUCS):
    try:
        df = pd.read_parquet(f"s3://snowml-model-ready/{huc}.parquet")
        train_dfs.append(df)
        if (i + 1) % 5 == 0:
            print(f"   Loaded {i+1}/{len(TRAIN_HUCS)} training HUCs...")
    except Exception as e:
        print(f"   ⚠️  Warning - couldn't load {huc}: {e}")
        continue

print("Loading validation data...")
val_dfs = []
for i, huc in enumerate(VAL_HUCS):
    try:
        df = pd.read_parquet(f"s3://snowml-model-ready/{huc}.parquet")
        val_dfs.append(df)
        if (i + 1) % 5 == 0:
            print(f"   Loaded {i+1}/{len(VAL_HUCS)} validation HUCs...")
    except Exception as e:
        print(f"   ⚠️  Warning - couldn't load {huc}: {e}")
        continue

if not train_dfs or not val_dfs:
    print("❌ ERROR: No data loaded! Check S3 access.")
    sys.exit(1)

print(f"\n✅ Data loaded successfully:")
print(f"   Training HUCs: {len(train_dfs)}")
print(f"   Validation HUCs: {len(val_dfs)}")

# Combine datasets
df_train = pd.concat(train_dfs, ignore_index=True)
df_val = pd.concat(val_dfs, ignore_index=True)

print(f"   Training samples: {len(df_train):,}")
print(f"   Validation samples: {len(df_val):,}")

# ============================================================================
# INITIALIZE MODEL
# ============================================================================

print(f"\n{'='*80}")
print("STEP 2: INITIALIZING MODEL")
print(f"{'='*80}")

model = LSTM_mod.LSTMModel(
    input_size=len(PARAMS["input_vars"]) + 1,  # +1 for lagged SWE
    hidden_size=PARAMS["hidden_size"],
    num_layers=PARAMS["num_layers"],
    output_size=PARAMS["num_class"],
    dropout=PARAMS["dropout"]
).to(PARAMS["device"])

optimizer = optim.Adam(model.parameters(), lr=PARAMS["learning_rate"])
loss_fn = torch.nn.MSELoss()

print(f"✅ Model initialized on {PARAMS['device']}")
print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

# ============================================================================
# TRAINING
# ============================================================================

print(f"\n{'='*80}")
print("STEP 3: TRAINING")
print(f"{'='*80}")

mlflow.set_tracking_uri(PARAMS["mlflow_tracking_uri"])
mlflow.set_experiment(PARAMS["expirement_name"])

with mlflow.start_run(run_name=PARAMS["run_name"]):
    # Log parameters
    mlflow.log_params({
        "n_train_hucs": len(train_dfs),
        "n_val_hucs": len(val_dfs),
        "features": ",".join(PARAMS["input_vars"]),
        "pilot_test": True,
        **{k: v for k, v in PARAMS.items() if isinstance(v, (int, float, str, bool))}
    })

    best_val_kge = -999

    for epoch in range(PARAMS["n_epochs"]):
        print(f"\n   ━━━ Epoch {epoch+1}/{PARAMS['n_epochs']} ━━━")
        epoch_start = datetime.now()

        # Train
        model.train()
        train_loss = LSTM_tr.train_epoch(model, optimizer, loss_fn, df_train, PARAMS)
        print(f"      Training loss: {train_loss:.4f}")

        # Validate
        model.eval()
        val_metrics = LSTM_tr.validate(model, df_val, PARAMS)

        epoch_time = (datetime.now() - epoch_start).total_seconds()

        print(f"      Validation KGE: {val_metrics['kge_median']:.4f}")
        print(f"      Validation MSE: {val_metrics['mse_median']:.4f}")
        print(f"      Time: {epoch_time:.1f}s")

        # Log metrics
        mlflow.log_metrics({
            "train_loss": train_loss,
            "val_kge_median": val_metrics["kge_median"],
            "val_kge_mean": val_metrics["kge_mean"],
            "val_mse_median": val_metrics["mse_median"],
            "epoch_time": epoch_time
        }, step=epoch)

        if val_metrics["kge_median"] > best_val_kge:
            best_val_kge = val_metrics["kge_median"]

    mlflow.log_metric("best_val_kge", best_val_kge)

    # Save model
    model_path = "models/exp1b_pilot.pt"
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), model_path)
    mlflow.log_artifact(model_path)

    run_id = mlflow.active_run().info.run_id

# ============================================================================
# RESULTS
# ============================================================================

print(f"\n{'='*80}")
print("✅ PILOT TEST COMPLETE!")
print(f"{'='*80}")
print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"\n📊 Results:")
print(f"   Best validation KGE: {best_val_kge:.4f}")
print(f"   MLflow run_id: {run_id}")
print(f"   Model saved: {model_path}")

# Save pilot results
pilot_results = {
    "best_val_kge": best_val_kge,
    "run_id": run_id,
    "n_train_hucs": len(train_dfs),
    "n_val_hucs": len(val_dfs),
    "timestamp": datetime.now().isoformat(),
    "success": True
}

with open("pilot_test_results.json", "w") as f:
    json.dump(pilot_results, f, indent=2)

print(f"\n✅ Pilot results saved to: pilot_test_results.json")

# Interpret results
print(f"\n{'='*80}")
print("INTERPRETATION")
print(f"{'='*80}")

if best_val_kge > 0.5:
    print("✅ EXCELLENT - Everything working correctly!")
    print("   Ready to run full experiment on g4dn.12xlarge")
    print(f"   Expected full KGE: 0.82-0.85 (this was only {len(TRAIN_HUCS)} HUCs)")
elif best_val_kge > 0.3:
    print("⚠️  MODERATE - System works but performance lower than expected")
    print("   May need to check data quality or hyperparameters")
    print("   Can proceed but review results carefully")
else:
    print("❌ LOW - Something may be wrong")
    print("   Review training logs before running full experiment")

print(f"\n📋 Next Steps:")
print(f"1. Review MLflow: mlflow ui --backend-store-uri sqlite:///mlflow.db")
print(f"2. If satisfied, run full experiment with g4dn.12xlarge")
print(f"3. Expected cost for full run: ~$20-25 (4-6 hours)")
print(f"\nTo launch full experiment:")
print(f"   ./launch_g4dn_training.sh")
