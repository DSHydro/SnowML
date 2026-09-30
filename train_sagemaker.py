
import os
import sys
import json
from datetime import datetime
import pandas as pd
import torch
from torch import optim

print("=" * 80)
print("SAGEMAKER TRAINING JOB")
print("=" * 80)

# Check GPU
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU count: {torch.cuda.device_count()}")

# Install SnowML
print("\nInstalling SnowML...")
os.system("git clone https://github.com/DSHydro/SnowML.git /opt/ml/code/SnowML")
os.system("pip install -q -e /opt/ml/code/SnowML")
os.system("pip install -q mlflow s3fs pyarrow")

from snowML.LSTM import LSTM_train as LSTM_tr
from snowML.LSTM import LSTM_model as LSTM_mod

# Configuration
PARAMS = {
    "hidden_size": 64,
    "num_layers": 1,
    "dropout": 0.3,
    "batch_size": 32,
    "lookback": 180,
    "n_epochs": 5,
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
    "input_vars": ["mean_tair", "mean_pr"],
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
}

# HUCs (subset for pilot)
TRAIN_HUCS = [
    "170100010101", "170100010102", "170100010103", "170100010201", "170100010202",
    "170100010203", "170100010301", "170100010302", "170100010401", "170100010402",
    "170100010403", "170100010404", "170100010501", "170100010502", "170100010503",
    "170100010504", "170100010601", "170100010602", "170100010603", "170100010604"
]

VAL_HUCS = [
    "170100010701", "170100010702", "170100010703", "170100010801", "170100010802",
    "170100010803", "170100010901", "170100010902", "170100010903", "170100011001"
]

print(f"\nLoading data from S3...")
print(f"Training HUCs: {len(TRAIN_HUCS)}")
print(f"Validation HUCs: {len(VAL_HUCS)}")

# Load training data
train_dfs = []
for huc in TRAIN_HUCS:
    try:
        df = pd.read_parquet(f"s3://snowml-model-ready/{huc}.parquet")
        train_dfs.append(df)
    except Exception as e:
        print(f"Warning: Could not load {huc}: {e}")

# Load validation data
val_dfs = []
for huc in VAL_HUCS:
    try:
        df = pd.read_parquet(f"s3://snowml-model-ready/{huc}.parquet")
        val_dfs.append(df)
    except Exception as e:
        print(f"Warning: Could not load {huc}: {e}")

if not train_dfs or not val_dfs:
    raise ValueError("No data loaded!")

df_train = pd.concat(train_dfs, ignore_index=True)
df_val = pd.concat(val_dfs, ignore_index=True)

print(f"Loaded {len(train_dfs)} train + {len(val_dfs)} val HUCs")
print(f"Training samples: {len(df_train):,}")
print(f"Validation samples: {len(df_val):,}")

# Initialize model
print("\nInitializing model...")
model = LSTM_mod.LSTMModel(
    input_size=len(PARAMS["input_vars"]) + 1,
    hidden_size=PARAMS["hidden_size"],
    num_layers=PARAMS["num_layers"],
    output_size=PARAMS["num_class"],
    dropout=PARAMS["dropout"]
).to(PARAMS["device"])

optimizer = optim.Adam(model.parameters(), lr=PARAMS["learning_rate"])
loss_fn = torch.nn.MSELoss()

print(f"Model on {PARAMS['device']}")
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

# Training
print("\n" + "=" * 80)
print("TRAINING")
print("=" * 80)

best_val_kge = -999

for epoch in range(PARAMS["n_epochs"]):
    print(f"\nEpoch {epoch+1}/{PARAMS['n_epochs']}")
    epoch_start = datetime.now()

    # Train
    model.train()
    train_loss = LSTM_tr.train_epoch(model, optimizer, loss_fn, df_train, PARAMS)

    # Validate
    model.eval()
    val_metrics = LSTM_tr.validate(model, df_val, PARAMS)

    epoch_time = (datetime.now() - epoch_start).total_seconds()

    print(f"  Train loss: {train_loss:.4f}")
    print(f"  Val KGE: {val_metrics['kge_median']:.4f}")
    print(f"  Time: {epoch_time:.1f}s")

    if val_metrics["kge_median"] > best_val_kge:
        best_val_kge = val_metrics["kge_median"]

# Save results
results = {
    "best_val_kge": float(best_val_kge),
    "n_train_hucs": len(train_dfs),
    "n_val_hucs": len(val_dfs),
    "timestamp": datetime.now().isoformat(),
    "success": True
}

output_dir = os.environ.get('SM_MODEL_DIR', '/opt/ml/model')
os.makedirs(output_dir, exist_ok=True)

with open(f"{output_dir}/results.json", "w") as f:
    json.dump(results, f, indent=2)

# Save model
torch.save(model.state_dict(), f"{output_dir}/model.pt")

print("\n" + "=" * 80)
print("TRAINING COMPLETE!")
print("=" * 80)
print(f"Best validation KGE: {best_val_kge:.4f}")
print(f"Results saved to: {output_dir}/results.json")

if best_val_kge > 0.5:
    print("\n✅ EXCELLENT - Ready for full experiments!")
elif best_val_kge > 0.3:
    print("\n⚠️  MODERATE - Review results")
else:
    print("\n❌ LOW - Something may be wrong")
