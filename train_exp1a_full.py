#!/usr/bin/env python3
"""
Experiment 1A - FULL TRAINING (ALL 533 HUCs INCLUDING EPHEMERAL)
=================================================================
Purpose: Train 8 model variations on ALL basins (deep snow + ephemeral)
Duration: ~48-60 hours on GPU
Branch: simran-unified-experiments

Scientific Question:
"Does including ephemeral snow basins improve or hurt model generalization?"

This is Experiment 1A (All Basins):
- Training: 272 HUCs (all snow types including ephemeral)
- Validation: 90 HUCs (20% held-out)
- Test Set A: 92 HUCs (20% random held-out)
- Test Set B: 81 HUCs (Yakima/Naches - unseen region, same as Exp1B)

Model Variations (8 total):
- 4 variable sets × 2 learning rates = 8 models
- After all finish, select best by median validation KGE

Variable Sets:
1. Base: [mean_pr, mean_tair, mean_swe_lag_30]
2. Base + Wind: [mean_pr, mean_tair, mean_swe_lag_30, mean_vs]
3. Base + Solar: [mean_pr, mean_tair, mean_swe_lag_30, mean_srad]
4. Full: [mean_pr, mean_tair, mean_swe_lag_30, mean_srad, mean_vs]

Learning Rates:
- 0.001 (standard)
- 0.0003 (slower, more stable - BEST in Exp1B)

Settings (from successful Exp1B):
- batch_size: 128 ✅
- num_workers: 4 ✅
- lookback: 180 days ✅
- train_size_fraction: 0.8 ✅
- Global normalization: YES ✅
- MLflow error handling: ADDED ✅

Usage on AWS SageMaker:
    nohup python train_exp1a_full.py > training_exp1a.log 2>&1 &
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
    log("EXPERIMENT 1A - ALL 533 HUCs (INCLUDING EPHEMERAL)")
    log("="*80)

    # Configuration (USING SUCCESSFUL EXP1B SETTINGS!)
    log("\n📋 Loading configuration...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    log(f"Using device: {device}")

    if device.type == 'cuda':
        log(f"GPU: {torch.cuda.get_device_name(0)}")
        log("🚀 GPU training enabled!")
    else:
        log("⚠️  GPU not available, using CPU")

    # PROVEN SETTINGS FROM EXP1B (DO NOT CHANGE!)
    n_epochs = 30
    hidden_size = 64
    num_layers = 1
    dropout = 0.3
    batch_size = 128       # ✅ From successful Exp1B
    num_workers = 4        # ✅ From successful Exp1B
    lookback = 180         # ✅ From successful Exp1B

    log(f"⚙️  Settings (from successful Exp1B):")
    log(f"   - batch_size: {batch_size}")
    log(f"   - num_workers: {num_workers}")
    log(f"   - lookback: {lookback}")
    log(f"   - epochs: {n_epochs}")

    # Define ALL 8 variations
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

    variations = all_variations  # Train ALL 8

    log(f"✅ Will train {len(variations)} variations")
    for i, var in enumerate(variations, 1):
        log(f"   {i}. {var['name']} - Features: {len(var['var_list'])}")

    # Load HUC Lists (EXP1A SPLITS!)
    log("\n📂 Loading Exp1A HUC lists...")
    with open('exp1a_train_hucs.txt', 'r') as f:
        train_hucs = [line.strip() for line in f if line.strip()]
    with open('exp1a_validation_hucs.txt', 'r') as f:
        val_hucs = [line.strip() for line in f if line.strip()]

    log(f"✅ Training HUCs: {len(train_hucs)} (includes ephemeral!)")
    log(f"✅ Validation HUCs: {len(val_hucs)} (includes ephemeral!)")
    log(f"✅ Total: {len(train_hucs) + len(val_hucs)} HUCs")

    # Load Data
    log("\n☁️  Loading data from S3...")
    log("This may take 5-10 minutes (more HUCs than Exp1B)...")

    var_list_max = all_variations[-1]['var_list']
    all_hucs = train_hucs + val_hucs

    start_time = datetime.now()
    try:
        df_dict, global_means, global_stds = LSTM_pp.pre_process(
            huc_list=all_hucs,
            var_list=var_list_max,
            UCLA=False,
            filter_dates=None,
            bucket_dict=sdc.create_bucket_dict("prod")
        )
        load_time = (datetime.now() - start_time).total_seconds()

        log(f"✅ Loaded {len(df_dict)} HUCs in {load_time:.1f} seconds")
        log(f"✅ Global normalization computed (mean/std across all {len(all_hucs)} HUCs)")

        df_dict_train = {huc: df_dict[huc] for huc in train_hucs}
        df_dict_val = {huc: df_dict[huc] for huc in val_hucs}

        log(f"✅ Training dict: {len(df_dict_train)} HUCs")
        log(f"✅ Validation dict: {len(df_dict_val)} HUCs")

    except Exception as e:
        log(f"❌ ERROR during data loading: {e}")
        log(f"⚠️  Make sure you're running on SageMaker with S3 access!")
        return 1

    # Setup MLflow
    log("\n📊 Setting up MLflow...")
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Exp1A_All_Basins")  # NEW EXPERIMENT NAME!
    log("✅ MLflow configured for Exp1A")

    # Train
    log("\n" + "="*80)
    log("🚀 STARTING TRAINING - ALL 8 VARIATIONS")
    log("="*80)
    log(f"⏱️  Estimated time: ~48-60 hours (8 variations × 6-7.5 hours)")
    log(f"💰 Estimated cost: ~$25-30 on ml.g4dn.xlarge")
    log(f"📍 More HUCs than Exp1B (272 vs 231), so slightly longer per epoch")
    log("")

    all_results = []

    for idx, variation in enumerate(variations, 1):
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
            'train_size_fraction': 0.8,  # Same as Exp1B
            'num_workers': num_workers,
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

        # MLflow run with ERROR HANDLING
        try:
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
                    "lookback": lookback,
                    "experiment": "Exp1A_All_Basins",
                    "includes_ephemeral": "YES"
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

                    # MLflow logging with error handling
                    try:
                        if train_loss is not None:
                            mlflow.log_metrics({"train_loss": train_loss}, step=epoch)
                        mlflow.log_metrics({"val_kge": val_kge}, step=epoch)
                    except Exception as e:
                        log(f"      ⚠️  MLflow metric logging failed: {e}")

                    # Save checkpoints every 10 epochs or at the end
                    if (epoch + 1) % 10 == 0 or epoch == n_epochs - 1:
                        checkpoint_name = f"checkpoint_{variation['name']}_epoch_{epoch}.pt"
                        try:
                            LSTM_tr.save_checkpoint(
                                model=model,
                                optimizer=optimizer,
                                epoch=epoch,
                                metrics={'val_kge': val_kge},
                                filepath=checkpoint_name,
                                params=params
                            )
                            log(f"      💾 Checkpoint saved: {checkpoint_name}")
                        except Exception as e:
                            log(f"      ⚠️  Checkpoint save failed: {e}")

                    epoch_duration = (datetime.now() - epoch_start).total_seconds()
                    log(f"      ⏱️  Epoch time: {epoch_duration:.1f}s")

                # Save final model
                log(f"\n   💾 Saving final model...")

                # MLflow model logging with ERROR HANDLING (IMPORTANT!)
                try:
                    mlflow.pytorch.log_model(
                        model,
                        artifact_path="model",
                        registered_model_name=f"Exp1A_{variation['name']}"
                    )
                    log(f"   ✅ MLflow model logged successfully")
                except Exception as e:
                    log(f"   ⚠️  MLflow model logging failed: {e}")
                    log(f"   ⚠️  Continuing anyway - local checkpoint still saved!")

                final_checkpoint_path = f"checkpoint_{variation['name']}_FINAL.pt"
                try:
                    LSTM_tr.save_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        epoch=n_epochs - 1,
                        metrics={'final_val_kge': val_kge},
                        filepath=final_checkpoint_path,
                        params=params
                    )
                    log(f"   ✅ Final checkpoint saved: {final_checkpoint_path}")
                except Exception as e:
                    log(f"   ❌ Final checkpoint save failed: {e}")

                variation_duration = (datetime.now() - variation_start).total_seconds() / 3600
                log(f"\n   ✅ {variation['name']} COMPLETE!")
                log(f"   ⏱️  Duration: {variation_duration:.2f} hours")
                log(f"   📊 Final KGE: {val_kge:.4f}")

                all_results.append({
                    'name': variation['name'],
                    'final_val_kge': val_kge,
                    'duration_hours': variation_duration
                })

        except Exception as e:
            log(f"\n   ❌ ERROR in variation {variation['name']}: {e}")
            log(f"   ⚠️  Skipping to next variation...")
            continue

    # Complete!
    log("\n" + "="*80)
    log("🎉 ALL TRAINING COMPLETE!")
    log("="*80)

    if len(all_results) > 0:
        # Save results summary
        results_df = pd.DataFrame(all_results)
        results_file = 'exp1a_training_results.csv'
        results_df.to_csv(results_file, index=False)

        log(f"\n📊 Results Summary:")
        log("")
        print(results_df.to_string(index=False))

        # Find best model
        best_model = results_df.loc[results_df['final_val_kge'].idxmax()]
        log(f"\n🏆 BEST MODEL:")
        log(f"   Name: {best_model['name']}")
        log(f"   Validation KGE: {best_model['final_val_kge']:.4f}")
        log(f"   Training time: {best_model['duration_hours']:.2f} hours")

        log(f"\n💾 Results saved to: {results_file}")

    total_time = (datetime.now() - start_time).total_seconds() / 3600
    log(f"\n⏱️  Total training time: {total_time:.2f} hours")
    log(f"📍 Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    log("\n" + "="*80)
    log("NEXT STEPS:")
    log("="*80)
    log("   1. Upload checkpoints to S3:")
    log("      aws s3 sync . s3://snowml-results/simran/exp1a_checkpoints/ --exclude '*' --include 'checkpoint_*_FINAL.pt'")
    log("   2. Download mlflow.db")
    log("   3. Download CSV results")
    log("   4. ⚠️  STOP THE AWS INSTANCE! ⚠️")
    log("      aws sagemaker stop-notebook-instance --notebook-instance-name st-gnn-T4x1 --region us-west-2")
    log("="*80)

    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        log("\n\n⚠️  Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        log(f"\n\n❌ FATAL ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
