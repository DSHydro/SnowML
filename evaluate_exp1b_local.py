#!/usr/bin/env python3
"""
Evaluate Experiment 1B Models on Test Sets A and B - LOCAL VERSION

Runs on laptop with local checkpoints and downloads HUC data from S3 as needed.
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import sys

# Import SnowML modules
from snowML.LSTM import LSTM_model as LSTM_mod
from snowML.LSTM import LSTM_pre_process as pp
from snowML.LSTM import LSTM_metrics as met

print("="*80)
print("EXPERIMENT 1B - TEST SET EVALUATION (Local)")
print("="*80)
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# =============================================================================
# CONFIGURATION
# =============================================================================

# Paths
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models" / "exp1b"
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = BASE_DIR / "data" / "huc_cache"
RESULTS_DIR = BASE_DIR / "results" / "exp1b_test_evaluation"

# Create directories
CACHE_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# S3 bucket for model-ready data
S3_BUCKET = "snowml-model-ready"

# Models to evaluate (top 3 - already downloaded locally)
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
    "Test_A": {
        "file": DATA_DIR / "exp1b_test_a_hucs.txt",
        "desc": "78 random deep snow HUCs"
    },
    "Test_B": {
        "file": DATA_DIR / "exp1b_test_b_hucs.txt",
        "desc": "81 Yakima/Naches HUCs"
    }
}

# Device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"🖥️  Device: {DEVICE}")
if DEVICE == 'cpu':
    print("   (CPU is fine for inference - just a bit slower)")
print()

# =============================================================================
# FUNCTIONS
# =============================================================================

def load_huc_list(file_path):
    """Load HUC IDs from text file"""
    with open(file_path, 'r') as f:
        hucs = [line.strip() for line in f if line.strip()]
    return hucs


def load_huc_data_from_s3(huc_id):
    """
    Load preprocessed HUC data from S3, with local caching

    Downloads from S3 on first use, then uses local cache
    """
    # Check cache first
    cache_file = CACHE_DIR / f"{huc_id}.parquet"

    if cache_file.exists():
        # Load from cache
        try:
            df = pd.read_parquet(cache_file)
            return df
        except Exception as e:
            print(f"  ⚠️  Cache corrupt for {huc_id}, re-downloading...")

    # Download from S3 (CSV format with naming: model_ready_hucXXXXXXXXXXXX.csv)
    s3_path = f"s3://{S3_BUCKET}/model_ready_huc{huc_id}.csv"

    try:
        df = pd.read_csv(s3_path, storage_options={"anon": False})

        # Convert day column to datetime if it exists
        if 'day' in df.columns:
            df['day'] = pd.to_datetime(df['day'])

        # Save to cache as parquet (much faster for subsequent loads)
        df.to_parquet(cache_file)

        return df
    except Exception as e:
        print(f"  ❌ Failed to load {huc_id}: {str(e)[:50]}")
        return None


def evaluate_huc(model, df, params, huc_id):
    """Evaluate model on single HUC, return metrics"""

    # Note: create_tensor expects 'mean_swe' as target column (hardcoded in LSTM_pre_process.py)
    # CSV files have 'mean_swe', so no renaming needed

    # CRITICAL: Normalize data using GLOBAL means/stds from training
    # Training used pre_process() which normalizes ALL HUCs using global statistics
    # NOT per-HUC normalization!
    global_means = params['global_means']
    global_stds = params['global_stds']

    # Z-score normalize using GLOBAL statistics (same as training)
    df_normalized = df.copy()
    for col in params['var_list'] + ['mean_swe']:
        if col in df.columns and col in global_means.index and col in global_stds.index:
            df_normalized[col] = (df[col] - global_means[col]) / global_stds[col]

    # Split data (67/33 train/test split like in training)
    train_frac = params.get('train_size_fraction', 0.67)
    df_train, df_test, _, _ = pp.train_test_split_time(df_normalized, train_frac)

    # Create test tensors (now on NORMALIZED data using GLOBAL stats)
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
    y_true_norm = y_test.cpu().numpy().flatten()
    y_pred_norm = y_pred.cpu().numpy().flatten()

    # CRITICAL: Denormalize predictions back to original scale using GLOBAL stats
    global_means = params['global_means']
    global_stds = params['global_stds']

    y_true = y_true_norm * global_stds['mean_swe'] + global_means['mean_swe']
    y_pred = y_pred_norm * global_stds['mean_swe'] + global_means['mean_swe']

    # Calculate metrics using SnowML's metric functions (on DENORMALIZED values)
    kge, r, alpha, beta = met.kling_gupta_efficiency(y_true, y_pred)
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))

    # NSE (Nash-Sutcliffe Efficiency)
    nse = 1 - (np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2))

    return {
        'huc_id': huc_id,
        'n_samples': len(y_true),
        'kge': kge,
        'kge_r': r,
        'kge_alpha': alpha,
        'kge_beta': beta,
        'nse': nse,
        'rmse': rmse,
        'mse': mse,
        'mae': mae,
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

    if not ckpt_path.exists():
        print(f"  ❌ Checkpoint not found: {ckpt_path}")
        return pd.DataFrame()

    print(f"Loading: {ckpt_path.name}")

    checkpoint = torch.load(ckpt_path, map_location='cpu')
    params = checkpoint['params']

    print(f"  Features: {params['var_list']}")
    print(f"  Learning rate: {params['learning_rate']}")
    print(f"  Val KGE: {model_config['val_kge']:.4f}")

    # Create model (architecture params not saved in checkpoint, using known values from training)
    model = LSTM_mod.SnowModel(
        input_size=len(params['var_list']),
        hidden_size=64,  # From training config
        num_class=1,
        num_layers=1,  # From training config
        dropout=0.3  # From training config
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Evaluate each HUC
    results = []
    n_success = 0
    n_failed = 0

    print(f"\nEvaluating {len(huc_list)} HUCs...")

    for i, huc_id in enumerate(huc_list, 1):
        # Progress indicator
        if i == 1 or i % 10 == 0 or i == len(huc_list):
            print(f"  [{i}/{len(huc_list)}] {huc_id}...", end='')

        # Load data
        df = load_huc_data_from_s3(huc_id)
        if df is None:
            if i == 1 or i % 10 == 0 or i == len(huc_list):
                print(" FAILED (load)")
            n_failed += 1
            continue

        # Check required columns (target is mean_swe in CSV files)
        required = params['var_list'] + ['mean_swe']
        missing = [col for col in required if col not in df.columns]
        if missing:
            if i == 1 or i % 10 == 0 or i == len(huc_list):
                print(f" FAILED (missing: {missing})")
            n_failed += 1
            continue

        # Evaluate
        try:
            metrics = evaluate_huc(model, df, params, huc_id)
            metrics['model'] = model_config['name']
            metrics['test_set'] = test_name
            results.append(metrics)
            n_success += 1

            if i == 1 or i % 10 == 0 or i == len(huc_list):
                print(f" ✓ (KGE: {metrics['kge']:.3f})")

        except Exception as e:
            if i == 1 or i % 10 == 0 or i == len(huc_list):
                print(f" FAILED ({str(e)[:30]})")
            n_failed += 1

    print(f"\n✅ Completed: {n_success}/{len(huc_list)} HUCs")
    if n_failed > 0:
        print(f"⚠️  Failed: {n_failed} HUCs")

    # Summary stats
    if len(results) > 0:
        df_results = pd.DataFrame(results)
        print(f"\n📊 Summary:")
        print(f"   Median KGE: {df_results['kge'].median():.4f}")
        print(f"   Mean KGE:   {df_results['kge'].mean():.4f} ± {df_results['kge'].std():.4f}")
        print(f"   Median NSE: {df_results['nse'].median():.4f}")
        print(f"   Median RMSE: {df_results['rmse'].median():.1f} mm")

        return df_results
    else:
        return pd.DataFrame()


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run evaluation"""

    # Load test sets
    print("📋 Loading test sets...")
    test_hucs = {}
    for name, info in TEST_SETS.items():
        hucs = load_huc_list(info['file'])
        test_hucs[name] = hucs
        print(f"  {name}: {len(hucs)} HUCs ({info['desc']})")

    total_hucs = sum(len(hucs) for hucs in test_hucs.values())
    print(f"\n📥 Will download/cache {total_hucs} HUC files from S3")
    print(f"   Cache directory: {CACHE_DIR}")

    # Check which checkpoints exist
    print(f"\n📦 Checking local checkpoints...")
    available_models = []
    for model in MODELS:
        ckpt_path = MODELS_DIR / model['file']
        if ckpt_path.exists():
            print(f"  ✅ {model['name']}")
            available_models.append(model)
        else:
            print(f"  ❌ {model['name']} - NOT FOUND")

    if not available_models:
        print("\n❌ No checkpoints found! Please check models/exp1b/ directory")
        return

    print(f"\n🚀 Starting evaluation with {len(available_models)} models...")
    start_time = datetime.now()

    # Run all evaluations
    all_results = []

    for model in available_models:
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
    for model in available_models:
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
                row[f'{test_name}_n_hucs'] = len(subset)

        comparison.append(row)

    comp_df = pd.DataFrame(comparison)
    comp_file = RESULTS_DIR / f"exp1b_val_vs_test_comparison_{timestamp}.csv"
    comp_df.to_csv(comp_file, index=False)

    print(comp_df.to_string(index=False))
    print(f"\n💾 Comparison: {comp_file}")

    # Final stats
    elapsed = datetime.now() - start_time
    print("\n" + "="*80)
    print(f"✅ EVALUATION COMPLETE!")
    print(f"   Started:  {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Duration: {elapsed}")
    print(f"   HUCs evaluated: {len(combined)}")
    print(f"   Cache size: {len(list(CACHE_DIR.glob('*.parquet')))} files")
    print("="*80)

    return combined, summary, comp_df


if __name__ == "__main__":
    try:
        results, summary, comparison = main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user. Partial results may be saved.")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
