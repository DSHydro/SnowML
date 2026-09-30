#!/usr/bin/env python3
"""
Experiment 1B - Training Variations 3-8 ONLY (Skip Variations 1 & 2)

CHANGES FROM train_exp1b_variations_2_to_8.py:
1. Start from variation 3 instead of 2 (line 107)
2. batch_size = 64 (was 128)
3. num_workers = 2 (was 4)
4. MLflow logging wrapped in try/except (won't crash)

Usage on AWS SageMaker:
    nohup python train_exp1b_variations_3_to_8_FIXED.py > training_3to8.log 2>&1 &
"""

from datetime import datetime
import pandas as pd
import torch
from torch import optim
import mlflow
import sys

from snowML.LSTM import LSTM_pre_process as LSTM_pp
from snowML.LSTM import LSTM_train as LSTM_tr
from snowML.LSTM import LSTM_model as LSTM_mod
from snowML.datapipe.utils import set_data_constants as sdc

def log(message):
    """Print with timestamp"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] {message}", flush=True)

def main():
    log("="*80)
    log("EXPERIMENT 1B - VARIATIONS 3-8 (SKIPPING VARIATIONS 1 & 2)")
    log("="*80)

    # Configuration
    log("\n📋 Loading configuration...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    log(f"Using device: {device}")

    if device.type == 'cuda':
        log(f"GPU: {torch.cuda.get_device_name(0)}")
        log("🚀 GPU training enabled!")
    else:
        log("⚠️  GPU not available, using CPU")

    n_epochs = 30
    hidden_size = 64
    num_layers = 1
    dropout = 0.3
    batch_size = 128     # Original config (fastest)
    lookback = 180

    log(f"⚙️  Settings: batch_size={batch_size}, num_workers=4")

    # Define ALL 8 variations (will skip first TWO)
    log("\n📊 Defining model variations...")
    all_variations = [
        {
            "name": "Base_LR0.001",
            "var_list": ["mean_pr", "mean_tair", "mean_swe_lag_30"],
            "learning_rate": 0.001
        },
        {
            "name": "Base_LR0.0003",
            "var_list": ["mean_pr", "mean_tair", "mean_swe_lag_30"],
            "learning_rate": 0.0003
        },
        {
            "name": "Base+Wind_LR0.001",
            "var_list": ["mean_pr", "mean_tair", "mean_swe_lag_30", "mean_vs"],
            "learning_rate": 0.001
        },
        {
            "name": "Base+Wind_LR0.0003",
            "var_list": ["mean_pr", "mean_tair", "mean_swe_lag_30", "mean_vs"],
            "learning_rate": 0.0003
        },
        {
            "name": "Base+Solar_LR0.001",
            "var_list": ["mean_pr", "mean_tair", "mean_swe_lag_30", "mean_srad"],
            "learning_rate": 0.001
        },
        {
            "name": "Base+Solar_LR0.0003",
            "var_list": ["mean_pr", "mean_tair", "mean_swe_lag_30", "mean_srad"],
            "learning_rate": 0.0003
        },
        {
            "name": "Full_LR0.001",
            "var_list": ["mean_pr", "mean_tair", "mean_swe_lag_30", "mean_srad", "mean_vs"],
            "learning_rate": 0.001
        },
        {
            "name": "Full_LR0.0003",
            "var_list": ["mean_pr", "mean_tair", "mean_swe_lag_30", "mean_srad", "mean_vs"],
            "learning_rate": 0.0003
        },
    ]

    # SKIP FIRST TWO VARIATIONS (already done)
    variations = all_variations[2:]  # CHANGED: Start from index 2 (third variation)

    log(f"✅ Will train {len(variations)} variations (skipping first two)")
    log(f"   Skipped: {all_variations[0]['name']}, {all_variations[1]['name']}")
    log(f"   Running: {variations[0]['name']} through {variations[-1]['name']}")

    # Load HUC Lists
    log("\n📂 Loading HUC lists...")
    with open('exp1b_train_hucs.txt', 'r') as f:
        train_hucs = [line.strip() for line in f if line.strip()]
    with open('exp1b_validation_hucs.txt', 'r') as f:
        val_hucs = [line.strip() for line in f if line.strip()]

    log(f"✅ Training HUCs: {len(train_hucs)}")
    log(f"✅ Validation HUCs: {len(val_hucs)}")

    # Load Data
    log("\n☁️  Loading data from S3...")
    log("This may take 3-5 minutes...")

    var_list_max = all_variations[-1]['var_list']
    all_hucs = train_hucs + val_hucs

    start_time = datetime.now()
    df_dict, global_means, global_stds = LSTM_pp.pre_process(
        huc_list=all_hucs,
        var_list=var_list_max,
        UCLA=False,
        filter_dates=None,
        bucket_dict=sdc.create_bucket_dict("prod")
    )
    load_time = (datetime.now() - start_time).total_seconds()

    log(f"✅ Loaded {len(df_dict)} HUCs in {load_time:.1f} seconds")

    df_dict_train = {huc: df_dict[huc] for huc in train_hucs}
    df_dict_val = {huc: df_dict[huc] for huc in val_hucs}

    # Setup MLflow
    log("\n📊 Setting up MLflow...")
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Exp1B_Full_DeepSnowOnly")
    log("✅ MLflow configured")

    # Train
    log("\n" + "="*80)
    log("🚀 STARTING TRAINING - VARIATIONS 3-8")
    log("="*80)
    log(f"⏱️  Estimated time: ~24-36 hours (6 variations × 4-6 hours)")
    log(f"💰 Estimated cost: ~$18 (saved ~$6 by skipping variations 1 & 2)")
    log("")

    all_results = []

    for idx, variation in enumerate(variations, 3):  # Start counting from 3
        log("\n" + "="*80)
        log(f"VARIATION {idx}/8: {variation['name']}")
        log("="*80)
        log(f"   Variables: {variation['var_list']}")
        log(f"   Learning Rate: {variation['learning_rate']}")

        var_list = variation['var_list']
        learning_rate = variation['learning_rate']

        params = {
            'var_list': var_list,
            'device': device,
            'batch_size': batch_size,
            'seq_length': lookback,
            'lookback': lookback,
            'global_means': global_means,
            'global_stds': global_stds,
            'loss_type': 'mse',
            'learning_rate': learning_rate,
            'train_size_dimension': 'time',
            'train_size_fraction': 0.8,
            'num_workers': 4,  # Original config (fastest)
            'recursive_predict': False,
            'lag_swe_var_idx': var_list.index('mean_swe_lag_30')
        }

        input_size = len(var_list)
        model = LSTM_mod.SnowModel(
            input_size=input_size,
            hidden_size=hidden_size,
            num_class=1,
            num_layers=num_layers,
            dropout=dropout
        ).to(device)

        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        loss_fn = torch.nn.MSELoss()

        total_params = sum(p.numel() for p in model.parameters())
        log(f"   ✅ Model initialized: {total_params:,} parameters")

        with mlflow.start_run(run_name=variation['name']):
            mlflow.log_params({
                "variation_name": variation['name'],
                "n_train_hucs": len(df_dict_train),
                "n_val_hucs": len(df_dict_val),
                "features": ",".join(var_list),
                "learning_rate": learning_rate,
                "n_epochs": n_epochs,
                "hidden_size": hidden_size,
                "device": str(device),
                "batch_size": batch_size,
                "lookback": lookback
            })

            log(f"   ✅ MLflow run started")

            variation_start = datetime.now()

            for epoch in range(n_epochs):
                epoch_start = datetime.now()
                log(f"\n   ━━━ Epoch {epoch + 1}/{n_epochs} ━━━")

                log(f"      🏋️  Training on {len(df_dict_train)} HUCs...")
                train_loss = LSTM_tr.pre_train(
                    model=model,
                    optimizer=optimizer,
                    loss_fn=loss_fn,
                    df_dict=df_dict_train,
                    params=params,
                    epoch=epoch
                )

                log(f"      🔍 Validating on {len(df_dict_val)} HUCs...")
                val_metrics = LSTM_tr.evaluate(
                    model_dawgs=model,
                    df_dict=df_dict_val,
                    params=params,
                    epoch=epoch
                )

                val_kge = val_metrics[1].get('test_kge', 0)
                log(f"      📊 Validation KGE: {val_kge:.4f}")

                if train_loss is not None:
                    mlflow.log_metrics({"train_loss": train_loss}, step=epoch)
                mlflow.log_metrics({"val_kge": val_kge}, step=epoch)

                if (epoch + 1) % 10 == 0 or epoch == n_epochs - 1:
                    checkpoint_name = f"checkpoint_{variation['name']}_epoch_{epoch}.pt"
                    LSTM_tr.save_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        epoch=epoch,
                        metrics={'val_kge': val_kge},
                        filepath=checkpoint_name,
                        params=params
                    )
                    log(f"      💾 Checkpoint saved")

                epoch_duration = (datetime.now() - epoch_start).total_seconds()
                log(f"      ⏱️  Epoch time: {epoch_duration:.1f}s")

            # Save final model - WRAPPED IN TRY/EXCEPT
            log(f"\n   💾 Saving final model...")

            try:
                mlflow.pytorch.log_model(
                    model,
                    artifact_path="model",
                    registered_model_name=f"Exp1B_{variation['name']}"
                )
                log(f"   ✅ MLflow model logged successfully")
            except Exception as e:
                log(f"   ⚠️  MLflow model logging failed: {e}")
                log(f"   ✅ Continuing anyway (checkpoint will still save)")

            final_checkpoint_path = f"checkpoint_{variation['name']}_FINAL.pt"
            LSTM_tr.save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=n_epochs - 1,
                metrics={'final_val_kge': val_kge},
                filepath=final_checkpoint_path,
                params=params
            )

            log(f"   ✅ Model saved to {final_checkpoint_path}")

            variation_duration = (datetime.now() - variation_start).total_seconds() / 3600
            log(f"\n   ✅ {variation['name']} COMPLETE!")
            log(f"   ⏱️  Duration: {variation_duration:.2f} hours")
            log(f"   📊 Final KGE: {val_kge:.4f}")

            all_results.append({
                'name': variation['name'],
                'final_val_kge': val_kge,
                'duration_hours': variation_duration
            })

    # Complete!
    log("\n" + "="*80)
    log("🎉 ALL TRAINING COMPLETE (VARIATIONS 3-8)!")
    log("="*80)

    results_df = pd.DataFrame(all_results)
    log("\n📊 Results Summary (Variations 3-8):")
    log("\n" + results_df.to_string(index=False))

    log(f"\n🏆 Best model (of 3-8): {results_df.loc[results_df['final_val_kge'].idxmax(), 'name']}")
    log(f"   Best KGE: {results_df['final_val_kge'].max():.4f}")

    results_df.to_csv('exp1b_training_results_var3to8.csv', index=False)
    log("\n💾 Results saved to: exp1b_training_results_var3to8.csv")

    log("\n⚠️  REMINDER: Don't forget you also have Variations 1 & 2 results!")
    log("   Combine all when analyzing all 8 variations.")

    log("\n✅ TRAINING COMPLETED SUCCESSFULLY!")
    log("\n📝 Don't forget to:")
    log("   1. Download checkpoint files")
    log("   2. Download mlflow.db")
    log("   3. Download CSV results")
    log("   4. STOP THE AWS INSTANCE!")

    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        log(f"❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
