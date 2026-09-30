#!/usr/bin/env python3
"""
Evaluate Experiment 1B Models on Test Sets A and B - SageMaker Version

This script is optimized to run on AWS SageMaker with direct S3 access.
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import sys

# Import SnowML modules
sys.path.insert(0, '/home/ec2-user/SageMaker')
from snowML.LSTM import LSTM_model as LSTM_mod
from snowML.LSTM import LSTM_pre_process as pp
from snowML.LSTM import LSTM_metrics as met

print("="*80)
print("EXPERIMENT 1B - TEST SET EVALUATION (SageMaker)")
print("="*80)
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# =============================================================================
# CONFIGURATION
# =============================================================================

# Working directory on SageMaker
WORK_DIR = Path("/home/ec2-user/SageMaker/exp1b_evaluation")
MODELS_DIR = WORK_DIR / "models"
RESULTS_DIR = WORK_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# S3 bucket for model-ready data
S3_BUCKET = "snowml-model-ready"

# Models to evaluate (top 3)
MODELS = [
    {
        "name": "Base+Solar_LR0.0003",
        "file": "checkpoint_Base+Solar_LR0.0003_FINAL.pt",
        "rank": 1,
        "val_kge": 0.9461
    },
    {
        "name": "Base+Wind_LR0.0003",
        "file": "checkpoint_Base+Wind_LR0.0003_FINAL.pt",
        "rank": 2,
        "val_kge": 0.9387
    },
    {
        "name": "Full_LR0.0003",
        "file": "checkpoint_Full_LR0.0003_FINAL.pt",
        "rank": 3,
        "val_kge": 0.9028
    }
]

# Test sets
TEST_SETS = {
    "Test_A": {"file": "exp1b_test_a_hucs.txt", "desc": "78 random deep snow HUCs"},
    "Test_B": {"file": "exp1b_test_b_hucs.txt", "desc": "81 Yakima/Naches HUCs"}
}

# Device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"🖥️  Device: {DEVICE}")
if DEVICE == 'cuda':
    print(f"   GPU: {torch.cuda.get_device_name(0)}\n")

# =============================================================================
# FUNCTIONS
# =============================================================================

def load_huc_list(filename):
    """Load HUC IDs from text file"""
    with open(WORK_DIR / filename, 'r') as f:
        hucs = [line.strip() for line in f if line.strip()]
    return hucs

def load_huc_data_from_s3(huc_id):
    """Load preprocessed HUC data from S3"""
    s3_path = f"s3://{S3_BUCKET}/pnw_swe_data/{huc_id}.parquet"
    try:
        df = pd.read_parquet(s3_path, storage_options={"anon": False})
        return df
    except Exception as e:
        print(f"  ❌ Failed to load {huc_id}: {e}")
        return None

def evaluate_huc(model, df, params, huc_id):
    """Evaluate model on single HUC, return metrics"""

    # Split data
    train_frac = params.get('train_size_fraction', 0.67)
    df_train, df_test, _, _ = pp.train_test_split_time(df, train_frac)

    # Create test tensors
    X_test, y_test = pp.create_tensor(df_test, params['lookback'], params['var_list'])

    # Move to device
    model = model.to(DEVICE)
    model.eval()
    X_test = X_test.to(DEVICE)
    y_test = y_test.to(DEVICE)

    # Predict
    with torch.no_grad():
        y_pred = model(X_test)

    # To numpy
    y_true = y_test.cpu().numpy().flatten()
    y_pred = y_pred.cpu().numpy().flatten()

    # Calculate metrics
    return {
        'huc_id': huc_id,
        'n_samples': len(y_true),
        'kge': met.kge_metric(y_true, y_pred),
        'nse': met.nse_metric(y_true, y_pred),
        'rmse': met.rmse_metric(y_true, y_pred),
        'mse': np.mean((y_true - y_pred) ** 2),
        'mae': np.mean(np.abs(y_true - y_pred)),
        'mean_obs': np.mean(y_true),
        'mean_pred': np.mean(y_pred),
        'std_obs': np.std(y_true),
        'std_pred': np.std(y_pred)
    }

def evaluate_model_on_test_set(model_config, test_name, huc_list):
    """Evaluate one model on one test set"""

    print(f"\n{'='*80}")
    print(f"Model: {model_config['name']} | Test Set: {test_name}")
    print(f"{'='*80}")

    # Load checkpoint
    ckpt_path = MODELS_DIR / model_config['file']
    print(f"Loading: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location='cpu')
    params = checkpoint['params']

    print(f"  Features: {params['var_list']}")
    print(f"  Learning rate: {params['learning_rate']}")
    print(f"  Val KGE: {model_config['val_kge']:.4f}")

    # Create model
    model = LSTM_mod.SnowModel(
        input_size=len(params['var_list']),
        hidden_size=params['hidden_size'],
        num_class=1,
        num_layers=params['num_layers'],
        dropout=params['dropout']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Evaluate each HUC
    results = []
    n_success = 0
    n_failed = 0

    for i, huc_id in enumerate(huc_list, 1):
        if i % 20 == 0:
            print(f"  Progress: {i}/{len(huc_list)} HUCs | Success: {n_success} | Failed: {n_failed}")

        # Load data
        df = load_huc_data_from_s3(huc_id)
        if df is None:
            n_failed += 1
            continue

        # Check required columns
        required = params['var_list'] + ['SWE']
        if not all(col in df.columns for col in required):
            print(f"  ⚠️  {huc_id} missing columns")
            n_failed += 1
            continue

        # Evaluate
        try:
            metrics = evaluate_huc(model, df, params, huc_id)
            metrics['model'] = model_config['name']
            metrics['test_set'] = test_name
            results.append(metrics)
            n_success += 1
        except Exception as e:
            print(f"  ❌ {huc_id}: {e}")
            n_failed += 1

    print(f"\n✅ Completed: {n_success}/{len(huc_list)} HUCs")
    if n_failed > 0:
        print(f"⚠️  Failed: {n_failed} HUCs")

    # Summary stats
    if len(results) > 0:
        df_results = pd.DataFrame(results)
        print(f"\n📊 Summary:")
        print(f"   Median KGE: {df_results['kge'].median():.4f}")
        print(f"   Mean KGE: {df_results['kge'].mean():.4f}")
        print(f"   Std KGE: {df_results['kge'].std():.4f}")
        print(f"   Median NSE: {df_results['nse'].median():.4f}")
        print(f"   Median RMSE: {df_results['rmse'].median():.4f} mm")

        return df_results
    else:
        return pd.DataFrame()

# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run evaluation"""

    # Load test sets
    print("\n📋 Loading test sets...")
    test_hucs = {}
    for name, info in TEST_SETS.items():
        hucs = load_huc_list(info['file'])
        test_hucs[name] = hucs
        print(f"  {name}: {len(hucs)} HUCs ({info['desc']})")

    # Run all evaluations
    all_results = []

    for model in MODELS:
        for test_name, huc_list in test_hucs.items():
            df_result = evaluate_model_on_test_set(model, test_name, huc_list)
            if len(df_result) > 0:
                all_results.append(df_result)

    # Combine results
    if len(all_results) == 0:
        print("\n❌ No results generated!")
        return

    combined = pd.concat(all_results, ignore_index=True)

    # Save detailed results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    detail_file = RESULTS_DIR / f"exp1b_test_results_detailed_{timestamp}.csv"
    combined.to_csv(detail_file, index=False)
    print(f"\n💾 Detailed results: {detail_file}")

    # Create summary
    summary = combined.groupby(['model', 'test_set']).agg({
        'kge': ['median', 'mean', 'std', 'min', 'max'],
        'nse': ['median', 'mean'],
        'rmse': ['median', 'mean'],
        'huc_id': 'count'
    }).round(4)

    summary.columns = ['_'.join(col) for col in summary.columns]
    summary = summary.rename(columns={'huc_id_count': 'n_hucs'})

    summary_file = RESULTS_DIR / f"exp1b_test_results_summary_{timestamp}.csv"
    summary.to_csv(summary_file)
    print(f"💾 Summary: {summary_file}")

    # Print summary table
    print("\n" + "="*80)
    print("FINAL SUMMARY - ALL MODELS & TEST SETS")
    print("="*80)
    print(summary.to_string())

    # Comparison table
    print("\n" + "="*80)
    print("COMPARISON: Validation vs Test Sets")
    print("="*80)

    comparison = []
    for model in MODELS:
        row = {
            'Model': model['name'],
            'Rank': model['rank'],
            'Val_KGE': model['val_kge']
        }

        for test_name in ['Test_A', 'Test_B']:
            subset = combined[(combined['model'] == model['name']) &
                            (combined['test_set'] == test_name)]
            if len(subset) > 0:
                row[f'{test_name}_KGE_median'] = round(subset['kge'].median(), 4)
                row[f'{test_name}_KGE_mean'] = round(subset['kge'].mean(), 4)

        comparison.append(row)

    comp_df = pd.DataFrame(comparison)
    comp_file = RESULTS_DIR / f"exp1b_val_vs_test_comparison_{timestamp}.csv"
    comp_df.to_csv(comp_file, index=False)

    print(comp_df.to_string(index=False))
    print(f"\n💾 Comparison: {comp_file}")

    print("\n" + "="*80)
    print(f"✅ EVALUATION COMPLETE!")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

    print("\n📦 To download results to your laptop:")
    print("   aws s3 sync /home/ec2-user/SageMaker/exp1b_evaluation/results/ \\")
    print("       s3://uw-echoe/simran/exp1b_evaluation/results/")
    print("\n   Then on laptop:")
    print("   aws s3 sync s3://uw-echoe/simran/exp1b_evaluation/results/ \\")
    print("       ./results/exp1b_test_evaluation/")

if __name__ == "__main__":
    main()
