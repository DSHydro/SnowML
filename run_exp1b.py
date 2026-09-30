#!/usr/bin/env python3
"""
Experiment 1B - CORRECT Training Using Exp3 Methodology
========================================================
This uses the existing multi_huc_expirement.py from Exp3 to train 8 model variations.

Data: 270 HUCs (162 train / 54 val / 55 test) from Exp3 splits
Method: train_size_dimension="huc" (entire HUC time series, not time-based splits)
Duration: ~28-32 hours for all 8 variations
Cost: ~$15-17 on ml.g4dn.xlarge

Scientific Question: Can multi-HUC training generalize to unseen deep snow watersheds?
Expected Results: Validation & Test KGE ~0.82-0.85 (matching Exp3 results)
"""

import sys
from datetime import datetime

# Make sure imports work
sys.path.insert(0, '/home/ec2-user/SageMaker')

from snowML.Scripts.load_hucs import load_huc_splits as lh
from snowML.Scripts import multi_huc_expirement as mhe
from snowML.LSTM import set_hyperparams as sh

def log(message):
    """Print with timestamp"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] {message}", flush=True)

def main():
    log("=" * 80)
    log("EXPERIMENT 1B - CORRECT TRAINING (Using Exp3 Methodology)")
    log("=" * 80)
    log("This is the correct re-run of Exp3 for deep snow HUCs only")
    log("")

    # Load HUC splits from Exp3's saved data
    log("📂 Loading HUC splits from Exp3...")
    tr, val, te = lh.huc_split()

    log(f"✅ HUC splits loaded:")
    log(f"   Train: {len(tr)} HUCs")
    log(f"   Validation: {len(val)} HUCs")
    log(f"   Test: {len(te)} HUCs")
    log(f"   Total: {len(tr) + len(val) + len(te)} HUCs")
    log("")

    # Define 8 model variations (same as Exp3)
    variations = [
        {
            "name": "Base",
            "vars": ["mean_pr", "mean_tair", "Mean Elevation"],
            "lr": 0.001,
            "description": "Temperature, Precipitation, Elevation"
        },
        {
            "name": "Base",
            "vars": ["mean_pr", "mean_tair", "Mean Elevation"],
            "lr": 0.0003,
            "description": "Temperature, Precipitation, Elevation"
        },
        {
            "name": "Base_Solar",
            "vars": ["mean_pr", "mean_tair", "Mean Elevation", "mean_srad"],
            "lr": 0.001,
            "description": "Base + Solar Radiation"
        },
        {
            "name": "Base_Solar",
            "vars": ["mean_pr", "mean_tair", "Mean Elevation", "mean_srad"],
            "lr": 0.0003,
            "description": "Base + Solar Radiation"
        },
        {
            "name": "Base_Wind",
            "vars": ["mean_pr", "mean_tair", "Mean Elevation", "mean_vs"],
            "lr": 0.001,
            "description": "Base + Wind Speed"
        },
        {
            "name": "Base_Wind",
            "vars": ["mean_pr", "mean_tair", "Mean Elevation", "mean_vs"],
            "lr": 0.0003,
            "description": "Base + Wind Speed"
        },
        {
            "name": "Base_Humidity",
            "vars": ["mean_pr", "mean_tair", "Mean Elevation", "mean_hum"],
            "lr": 0.001,
            "description": "Base + Humidity"
        },
        {
            "name": "Base_Humidity",
            "vars": ["mean_pr", "mean_tair", "Mean Elevation", "mean_hum"],
            "lr": 0.0003,
            "description": "Base + Humidity"
        },
    ]

    log(f"📊 Will train {len(variations)} model variations")
    log("")

    # Track overall progress
    start_time_all = datetime.now()

    # Train each variation
    for i, var_config in enumerate(variations, 1):
        var_start_time = datetime.now()

        log("=" * 80)
        log(f"VARIATION {i}/{len(variations)}: {var_config['name']}_LR{var_config['lr']}")
        log("=" * 80)
        log(f"Description: {var_config['description']}")
        log(f"Learning Rate: {var_config['lr']}")
        log(f"Variables: {var_config['vars']}")
        log("")

        # Create parameters using Exp3's parameter structure
        params = sh.create_hyper_dict()

        # CRITICAL: Set correct parameters for multi-HUC training
        params["train_size_dimension"] = "huc"     # ← Use ENTIRE HUC (not time-based split)
        params["train_size_fraction"] = 1          # ← 100% of each HUC's time series
        params["var_list"] = var_config['vars']
        params["learning_rate"] = var_config['lr']
        params["n_epochs"] = 30
        params["batch_size"] = 32
        params["hidden_size"] = 64
        params["num_layers"] = 1
        params["dropout"] = 0.5
        params["loss_type"] = "mse"
        params["expirement_name"] = f"Exp1B_{var_config['name']}_LR{var_config['lr']}"
        params["mlflow_tracking_uri"] = "sqlite:///mlflow.db"
        params["MLFLOW_ON"] = True

        log(f"⚙️  Training parameters:")
        log(f"   train_size_dimension: {params['train_size_dimension']}")
        log(f"   train_size_fraction: {params['train_size_fraction']}")
        log(f"   Experiment name: {params['expirement_name']}")
        log(f"   MLflow URI: {params['mlflow_tracking_uri']}")
        log("")

        # Run experiment using Exp3's existing code
        # This function does EVERYTHING:
        # - Downloads data from S3 automatically
        # - Trains the model
        # - Validates each epoch
        # - Logs to MLflow (params, metrics per HUC, models)
        log(f"🚀 Starting training (this will take ~3.5-4 hours)...")
        log(f"   Multi-HUC training will process {len(tr)} training HUCs")
        log(f"   Validation on {len(val)} HUCs after each epoch")
        log("")

        try:
            mhe.run_expirement(tr, val, params)

            var_elapsed = (datetime.now() - var_start_time).total_seconds() / 3600
            log("")
            log(f"✅ Variation {i}/{len(variations)} COMPLETE!")
            log(f"   Duration: {var_elapsed:.2f} hours")
            log("")

        except Exception as e:
            log(f"❌ ERROR in variation {i}: {e}")
            log(f"   Continuing to next variation...")
            import traceback
            traceback.print_exc()
            log("")

    # Summary
    total_elapsed = (datetime.now() - start_time_all).total_seconds() / 3600

    log("=" * 80)
    log("🎉 ALL 8 VARIATIONS COMPLETE!")
    log("=" * 80)
    log(f"Started:  {start_time_all.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Total Duration: {total_elapsed:.2f} hours")
    log(f"Average per variation: {total_elapsed/len(variations):.2f} hours")
    log("")
    log("📁 Results saved to:")
    log("   - mlflow.db (all experiments, metrics, parameters)")
    log("   - Models logged in MLflow (accessible via mlflow UI)")
    log("")
    log("⚠️  IMPORTANT: Stop your AWS instance to avoid charges!")
    log("   aws sagemaker stop-notebook-instance --notebook-instance-name st-gnn-T4x1 --region us-west-2")
    log("=" * 80)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("\n⚠️  Training interrupted by user")
        log("Partial results may be in mlflow.db")
    except Exception as e:
        log(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
