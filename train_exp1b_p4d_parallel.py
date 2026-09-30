#!/usr/bin/env python3
"""
Experiment 1B - PARALLEL GPU TRAINING for p4d.24xlarge
======================================================
Hardware: 8× NVIDIA A100 GPUs (40GB each)
Strategy: Train all 8 model variations in PARALLEL (one per GPU)
Duration: ~1 hour (all variations finish together)

Scientific Question:
"Can we train ONE LSTM on multiple deep snow watersheds and have it generalize?"

Model Variations (8 total running in parallel):
- 4 variable sets × 2 learning rates = 8 models
- Each model trains on a separate GPU simultaneously
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
import numpy as np
import multiprocessing as mp
from multiprocessing import Process, Queue

print("=" * 80)
print("EXPERIMENT 1B - PARALLEL GPU TRAINING")
print("Multi-HUC Deep Snow Only - 8 GPUs in Parallel")
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
from snowML.LSTM import set_hyperparams as sh
from snowML.LSTM import LSTM_pre_process as pp

# ============================================================================
# CONFIGURATION
# ============================================================================

# Base parameters (same for all variations)
BASE_PARAMS = {
    "hidden_size": 64,
    "num_layers": 1,
    "dropout": 0.3,
    "batch_size": 32,
    "lookback": 180,
    "n_epochs": 30,
    "num_workers": 4,  # Use multiple workers on powerful hardware
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
    "expirement_name": "Exp1B_Parallel_P4D",
    "mlflow_tracking_uri": "sqlite:///mlflow.db",
    "MLFLOW_ON": True
}

# Define 8 model variations
VARIATIONS = [
    {
        "name": "Base_LR0.001",
        "features": ["mean_tair", "mean_pr"],
        "learning_rate": 0.001,
        "gpu_id": 0
    },
    {
        "name": "Base_LR0.0003",
        "features": ["mean_tair", "mean_pr"],
        "learning_rate": 0.0003,
        "gpu_id": 1
    },
    {
        "name": "Base_Wind_LR0.001",
        "features": ["mean_tair", "mean_pr", "mean_vs"],
        "learning_rate": 0.001,
        "gpu_id": 2
    },
    {
        "name": "Base_Wind_LR0.0003",
        "features": ["mean_tair", "mean_pr", "mean_vs"],
        "learning_rate": 0.0003,
        "gpu_id": 3
    },
    {
        "name": "Base_Solar_LR0.001",
        "features": ["mean_tair", "mean_pr", "mean_srad"],
        "learning_rate": 0.001,
        "gpu_id": 4
    },
    {
        "name": "Base_Solar_LR0.0003",
        "features": ["mean_tair", "mean_pr", "mean_srad"],
        "learning_rate": 0.0003,
        "gpu_id": 5
    },
    {
        "name": "Base_Solar_Wind_LR0.001",
        "features": ["mean_tair", "mean_pr", "mean_srad", "mean_vs"],
        "learning_rate": 0.001,
        "gpu_id": 6
    },
    {
        "name": "Base_Solar_Wind_LR0.0003",
        "features": ["mean_tair", "mean_pr", "mean_srad", "mean_vs"],
        "learning_rate": 0.0003,
        "gpu_id": 7
    }
]

# Load HUC splits
DATA_DIR = Path("data")
with open(DATA_DIR / "exp1b_train_hucs.txt") as f:
    TRAIN_HUCS = [line.strip() for line in f]
with open(DATA_DIR / "exp1b_validation_hucs.txt") as f:
    VAL_HUCS = [line.strip() for line in f]

print(f"\n📊 Data Configuration:")
print(f"   Training HUCs: {len(TRAIN_HUCS)}")
print(f"   Validation HUCs: {len(VAL_HUCS)}")
print(f"   Model variations: {len(VARIATIONS)}")

# ============================================================================
# TRAINING FUNCTION (runs on single GPU)
# ============================================================================

def train_single_variation(variation, result_queue):
    """
    Train one model variation on one GPU.
    This function runs in a separate process.
    """
    try:
        # Set GPU for this process
        gpu_id = variation["gpu_id"]
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        device = torch.device("cuda:0")  # Will map to the visible GPU

        print(f"\n{'='*60}")
        print(f"GPU {gpu_id}: Starting {variation['name']}")
        print(f"{'='*60}")

        # Create parameters for this variation
        params = BASE_PARAMS.copy()
        params["learning_rate"] = variation["learning_rate"]
        params["input_vars"] = variation["features"]
        params["run_name"] = f"Exp1B_{variation['name']}"
        params["device"] = device

        # Load and prepare data
        print(f"GPU {gpu_id}: Loading training data...")
        train_dfs = []
        for huc in TRAIN_HUCS:
            try:
                df = pd.read_parquet(f"s3://snowml-model-ready/{huc}.parquet")
                train_dfs.append(df)
            except Exception as e:
                print(f"GPU {gpu_id}: Warning - couldn't load {huc}: {e}")
                continue

        print(f"GPU {gpu_id}: Loading validation data...")
        val_dfs = []
        for huc in VAL_HUCS:
            try:
                df = pd.read_parquet(f"s3://snowml-model-ready/{huc}.parquet")
                val_dfs.append(df)
            except Exception as e:
                print(f"GPU {gpu_id}: Warning - couldn't load {huc}: {e}")
                continue

        if not train_dfs or not val_dfs:
            raise ValueError(f"No data loaded for GPU {gpu_id}")

        print(f"GPU {gpu_id}: Loaded {len(train_dfs)} train + {len(val_dfs)} val HUCs")

        # Combine datasets
        df_train = pd.concat(train_dfs, ignore_index=True)
        df_val = pd.concat(val_dfs, ignore_index=True)

        # Initialize model
        print(f"GPU {gpu_id}: Initializing model...")
        model = LSTM_mod.LSTMModel(
            input_size=len(variation["features"]) + 1,  # +1 for lagged SWE
            hidden_size=params["hidden_size"],
            num_layers=params["num_layers"],
            output_size=params["num_class"],
            dropout=params["dropout"]
        ).to(device)

        optimizer = optim.Adam(model.parameters(), lr=params["learning_rate"])
        loss_fn = torch.nn.MSELoss()

        # Start MLflow run
        mlflow.set_tracking_uri(params["mlflow_tracking_uri"])
        mlflow.set_experiment(params["expirement_name"])

        with mlflow.start_run(run_name=params["run_name"]):
            # Log parameters
            mlflow.log_params({
                "gpu_id": gpu_id,
                "features": ",".join(variation["features"]),
                "learning_rate": variation["learning_rate"],
                "n_train_hucs": len(train_dfs),
                "n_val_hucs": len(val_dfs),
                **{k: v for k, v in params.items() if isinstance(v, (int, float, str, bool))}
            })

            # Training loop
            print(f"GPU {gpu_id}: Starting training ({params['n_epochs']} epochs)...")
            best_val_kge = -999

            for epoch in range(params["n_epochs"]):
                # Train
                model.train()
                epoch_start = datetime.now()
                train_loss = LSTM_tr.train_epoch(model, optimizer, loss_fn, df_train, params)

                # Validate
                model.eval()
                val_metrics = LSTM_tr.validate(model, df_val, params)

                epoch_time = (datetime.now() - epoch_start).total_seconds()

                # Log metrics
                mlflow.log_metrics({
                    "train_loss": train_loss,
                    "val_kge_median": val_metrics["kge_median"],
                    "val_kge_mean": val_metrics["kge_mean"],
                    "val_mse_median": val_metrics["mse_median"],
                    "epoch_time": epoch_time
                }, step=epoch)

                # Track best model
                if val_metrics["kge_median"] > best_val_kge:
                    best_val_kge = val_metrics["kge_median"]

                # Progress update
                if (epoch + 1) % 5 == 0:
                    print(f"GPU {gpu_id}: Epoch {epoch+1}/{params['n_epochs']} - "
                          f"Val KGE: {val_metrics['kge_median']:.3f} - "
                          f"Time: {epoch_time:.0f}s")

            # Log final best metrics
            mlflow.log_metric("best_val_kge", best_val_kge)

            # Save model
            model_path = f"models/exp1b_{variation['name']}.pt"
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), model_path)
            mlflow.log_artifact(model_path)

            run_id = mlflow.active_run().info.run_id
            print(f"\n✅ GPU {gpu_id}: {variation['name']} COMPLETE!")
            print(f"   Best validation KGE: {best_val_kge:.4f}")
            print(f"   MLflow run_id: {run_id}")

            # Return results
            result_queue.put({
                "variation": variation["name"],
                "gpu_id": gpu_id,
                "best_val_kge": best_val_kge,
                "run_id": run_id,
                "success": True
            })

    except Exception as e:
        print(f"\n❌ GPU {gpu_id}: ERROR in {variation['name']}")
        print(f"   Error: {str(e)}")
        import traceback
        traceback.print_exc()

        result_queue.put({
            "variation": variation["name"],
            "gpu_id": gpu_id,
            "error": str(e),
            "success": False
        })

# ============================================================================
# MAIN PARALLEL EXECUTION
# ============================================================================

if __name__ == "__main__":
    print(f"\n{'='*80}")
    print("LAUNCHING PARALLEL TRAINING")
    print(f"{'='*80}")
    print(f"Starting {len(VARIATIONS)} processes (one per GPU)...")

    # Create result queue
    result_queue = Queue()

    # Launch all training processes
    processes = []
    for variation in VARIATIONS:
        p = Process(target=train_single_variation, args=(variation, result_queue))
        p.start()
        processes.append(p)
        print(f"   ✅ Launched process for GPU {variation['gpu_id']}: {variation['name']}")

    # Wait for all processes to complete
    print(f"\n{'='*80}")
    print("WAITING FOR ALL GPUS TO COMPLETE")
    print(f"{'='*80}")
    print("(This will take ~45-60 minutes)")
    print("All 8 models training in parallel...")

    for p in processes:
        p.join()

    # Collect results
    print(f"\n{'='*80}")
    print("ALL TRAINING COMPLETE - COLLECTING RESULTS")
    print(f"{'='*80}")

    results = []
    while not result_queue.empty():
        results.append(result_queue.get())

    # Sort results by GPU ID
    results.sort(key=lambda x: x.get("gpu_id", 999))

    # Display results
    print("\n📊 FINAL RESULTS:")
    print("-" * 80)
    successful_results = [r for r in results if r.get("success")]
    failed_results = [r for r in results if not r.get("success")]

    if successful_results:
        print(f"\n✅ Successful ({len(successful_results)}/{len(VARIATIONS)}):")
        for r in successful_results:
            print(f"   GPU {r['gpu_id']}: {r['variation']:30s} | KGE: {r['best_val_kge']:.4f} | Run: {r['run_id'][:8]}")

        # Find best model
        best = max(successful_results, key=lambda x: x["best_val_kge"])
        print(f"\n🏆 BEST MODEL: {best['variation']}")
        print(f"   Validation KGE: {best['best_val_kge']:.4f}")
        print(f"   MLflow run_id: {best['run_id']}")

        # Save best model info
        with open("exp1b_best_model.json", "w") as f:
            json.dump(best, f, indent=2)
        print(f"   ✅ Saved to: exp1b_best_model.json")

    if failed_results:
        print(f"\n❌ Failed ({len(failed_results)}/{len(VARIATIONS)}):")
        for r in failed_results:
            print(f"   GPU {r['gpu_id']}: {r['variation']:30s} | Error: {r.get('error', 'Unknown')}")

    print(f"\n{'='*80}")
    print("🎉 EXPERIMENT 1B PARALLEL TRAINING COMPLETE!")
    print(f"{'='*80}")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nNext steps:")
    print(f"1. Review results in MLflow: mlflow ui --backend-store-uri sqlite:///mlflow.db")
    print(f"2. Best model saved to: exp1b_best_model.json")
    print(f"3. Ready to start Experiment 1A or Experiment 2")
