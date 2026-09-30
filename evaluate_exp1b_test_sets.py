#!/usr/bin/env python3
"""
Evaluate Experiment 1B Models on Test Sets A and B

This script loads the top 3 trained models from Exp1B and evaluates them on:
- Test Set A: 78 random held-out deep snow HUCs
- Test Set B: 81 Yakima/Naches HUCs (completely unseen region)

Generates:
- Per-HUC metrics (KGE, NSE, RMSE)
- Summary statistics
- Comparison tables
- CSV outputs for professor report
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime

# Import SnowML modules
from snowML.LSTM import LSTM_model as LSTM_mod
from snowML.LSTM import LSTM_train as LSTM_tr
from snowML.LSTM import LSTM_pre_process as pp
from snowML.LSTM import LSTM_metrics as met

# =============================================================================
# CONFIGURATION
# =============================================================================

# Paths
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models" / "exp1b"
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results" / "exp1b_test_evaluation"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# S3 bucket for data
S3_BUCKET = "snowml-model-ready"  # Bucket name without s3://

# Models to evaluate (top 3 from Exp1B)
MODELS = [
    {
        "name": "Base+Solar_LR0.0003",
        "checkpoint": "checkpoint_Base+Solar_LR0.0003_FINAL.pt",
        "rank": 1,
        "val_kge": 0.9461
    },
    {
        "name": "Base+Wind_LR0.0003",
        "checkpoint": "checkpoint_Base+Wind_LR0.0003_FINAL.pt",
        "rank": 2,
        "val_kge": 0.9387
    },
    {
        "name": "Full_LR0.0003",
        "checkpoint": "checkpoint_Full_LR0.0003_FINAL.pt",
        "rank": 3,
        "val_kge": 0.9028
    }
]

# Test sets
TEST_SETS = {
    "Test_A": {
        "file": DATA_DIR / "exp1b_test_a_hucs.txt",
        "description": "78 random held-out deep snow HUCs",
        "n_hucs": 78
    },
    "Test_B": {
        "file": DATA_DIR / "exp1b_test_b_hucs.txt",
        "description": "81 Yakima/Naches HUCs (unseen region)",
        "n_hucs": 81
    }
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_huc_list(file_path):
    """Load HUC IDs from text file"""
    with open(file_path, 'r') as f:
        hucs = [line.strip() for line in f if line.strip()]
    print(f"✅ Loaded {len(hucs)} HUCs from {file_path.name}")
    return hucs


def load_checkpoint(checkpoint_path):
    """Load model checkpoint and extract params"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    params = checkpoint['params']
    metrics = checkpoint.get('metrics', {})

    print(f"  Features: {params['var_list']}")
    print(f"  Learning rate: {params['learning_rate']}")
    print(f"  Validation KGE: {metrics.get('final_val_kge', 'N/A')}")

    return checkpoint, params


def create_model(params):
    """Create LSTM model from parameters"""
    model = LSTM_mod.SnowModel(
        input_size=len(params['var_list']),
        hidden_size=params['hidden_size'],
        num_class=1,
        num_layers=params['num_layers'],
        dropout=params['dropout']
    )
    return model


def load_huc_data(huc_id, params):
    """
    Load preprocessed data for a single HUC from S3

    Returns:
        DataFrame with required features
    """
    # Construct S3 path (following existing data structure)
    s3_path = f"s3://{S3_BUCKET}/pnw_swe_data/{huc_id}.parquet"

    try:
        # Load from S3 using pandas
        df = pd.read_parquet(s3_path, storage_options={"anon": False})

        # Verify required columns exist
        required_cols = params['var_list'] + ['SWE']
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            print(f"  ⚠️  Missing columns for {huc_id}: {missing_cols}")
            return None

        return df

    except Exception as e:
        print(f"  ❌ Error loading {huc_id}: {e}")
        return None


def evaluate_huc(model, df, params, huc_id):
    """
    Evaluate model on a single HUC

    Returns:
        dict with metrics (KGE, NSE, RMSE, etc.)
    """
    # Split data (using time-based split like in training)
    train_size_fraction = params.get('train_size_fraction', 0.67)
    df_train, df_test, _, _ = pp.train_test_split_time(df, train_size_fraction)

    # Create tensors for test set
    X_test, y_test = pp.create_tensor(df_test, params['lookback'], params['var_list'])

    # Move to device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()

    X_test = X_test.to(device)
    y_test = y_test.to(device)

    # Predict
    with torch.no_grad():
        y_pred = model(X_test)

    # Convert to numpy
    y_test_np = y_test.cpu().numpy().flatten()
    y_pred_np = y_pred.cpu().numpy().flatten()

    # Calculate metrics
    metrics = {
        'huc_id': huc_id,
        'n_samples': len(y_test_np),
        'kge': met.kge_metric(y_test_np, y_pred_np),
        'nse': met.nse_metric(y_test_np, y_pred_np),
        'rmse': met.rmse_metric(y_test_np, y_pred_np),
        'mse': np.mean((y_test_np - y_pred_np) ** 2),
        'mae': np.mean(np.abs(y_test_np - y_pred_np)),
        'mean_obs': np.mean(y_test_np),
        'mean_pred': np.mean(y_pred_np),
        'std_obs': np.std(y_test_np),
        'std_pred': np.std(y_pred_np)
    }

    return metrics


def evaluate_model_on_test_set(model_config, test_set_name, huc_list, params):
    """
    Evaluate one model on one test set

    Returns:
        DataFrame with per-HUC metrics
    """
    print(f"\n{'='*80}")
    print(f"Evaluating: {model_config['name']} on {test_set_name}")
    print(f"{'='*80}")

    # Load model
    checkpoint_path = MODELS_DIR / model_config['checkpoint']
    checkpoint, model_params = load_checkpoint(checkpoint_path)

    model = create_model(model_params)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Update params with model-specific settings
    eval_params = params.copy()
    eval_params.update(model_params)

    # Evaluate on each HUC
    results = []
    n_success = 0
    n_failed = 0

    for i, huc_id in enumerate(huc_list, 1):
        if i % 10 == 0:
            print(f"  Progress: {i}/{len(huc_list)} HUCs")

        # Load HUC data
        df = load_huc_data(huc_id, eval_params)
        if df is None:
            n_failed += 1
            continue

        # Evaluate
        try:
            metrics = evaluate_huc(model, df, eval_params, huc_id)
            metrics['model'] = model_config['name']
            metrics['test_set'] = test_set_name
            results.append(metrics)
            n_success += 1
        except Exception as e:
            print(f"  ❌ Evaluation failed for {huc_id}: {e}")
            n_failed += 1

    print(f"\n✅ Successfully evaluated: {n_success}/{len(huc_list)} HUCs")
    if n_failed > 0:
        print(f"⚠️  Failed: {n_failed} HUCs")

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Print summary statistics
    if len(results_df) > 0:
        print(f"\n📊 Summary Statistics for {test_set_name}:")
        print(f"   Median KGE: {results_df['kge'].median():.4f}")
        print(f"   Mean KGE: {results_df['kge'].mean():.4f}")
        print(f"   Std KGE: {results_df['kge'].std():.4f}")
        print(f"   Median NSE: {results_df['nse'].median():.4f}")
        print(f"   Median RMSE: {results_df['rmse'].median():.4f}")

    return results_df


# =============================================================================
# MAIN EVALUATION
# =============================================================================

def main():
    """Run full evaluation on all models and test sets"""

    print("=" * 80)
    print("EXPERIMENT 1B - TEST SET EVALUATION")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Results will be saved to: {RESULTS_DIR}")

    # Common params
    base_params = {
        'lookback': 180,
        'train_size_fraction': 0.67,
        'batch_size': 128,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }

    print(f"\n🖥️  Device: {base_params['device']}")

    # Load all test sets
    test_hucs = {}
    for test_name, test_info in TEST_SETS.items():
        test_hucs[test_name] = load_huc_list(test_info['file'])

    # Evaluate all combinations
    all_results = []

    for model_config in MODELS:
        for test_name, huc_list in test_hucs.items():
            results_df = evaluate_model_on_test_set(
                model_config,
                test_name,
                huc_list,
                base_params
            )
            all_results.append(results_df)

    # Combine all results
    combined_results = pd.concat(all_results, ignore_index=True)

    # Save detailed results
    output_file = RESULTS_DIR / f"exp1b_test_results_detailed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    combined_results.to_csv(output_file, index=False)
    print(f"\n💾 Detailed results saved to: {output_file}")

    # Create summary table
    summary = combined_results.groupby(['model', 'test_set']).agg({
        'kge': ['median', 'mean', 'std', 'min', 'max'],
        'nse': ['median', 'mean'],
        'rmse': ['median', 'mean'],
        'huc_id': 'count'
    }).round(4)

    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    summary = summary.rename(columns={'huc_id_count': 'n_hucs'})

    summary_file = RESULTS_DIR / f"exp1b_test_results_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    summary.to_csv(summary_file)
    print(f"💾 Summary saved to: {summary_file}")

    # Print final summary table
    print("\n" + "=" * 80)
    print("FINAL SUMMARY - ALL MODELS & TEST SETS")
    print("=" * 80)
    print(summary.to_string())

    # Create comparison: Validation vs Test A vs Test B
    print("\n" + "=" * 80)
    print("COMPARISON: Validation vs Test Sets")
    print("=" * 80)

    comparison = []
    for model in MODELS:
        row = {
            'Model': model['name'],
            'Rank': model['rank'],
            'Val_KGE': model['val_kge']
        }

        # Test Set A
        test_a_data = combined_results[
            (combined_results['model'] == model['name']) &
            (combined_results['test_set'] == 'Test_A')
        ]
        if len(test_a_data) > 0:
            row['TestA_KGE_median'] = test_a_data['kge'].median()
            row['TestA_KGE_mean'] = test_a_data['kge'].mean()

        # Test Set B
        test_b_data = combined_results[
            (combined_results['model'] == model['name']) &
            (combined_results['test_set'] == 'Test_B')
        ]
        if len(test_b_data) > 0:
            row['TestB_KGE_median'] = test_b_data['kge'].median()
            row['TestB_KGE_mean'] = test_b_data['kge'].mean()

        comparison.append(row)

    comparison_df = pd.DataFrame(comparison)
    comparison_file = RESULTS_DIR / f"exp1b_validation_vs_test_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    comparison_df.to_csv(comparison_file, index=False)

    print(comparison_df.to_string(index=False))
    print(f"\n💾 Comparison saved to: {comparison_file}")

    print("\n" + "=" * 80)
    print(f"✅ EVALUATION COMPLETE!")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    return combined_results, summary, comparison_df


if __name__ == "__main__":
    results, summary, comparison = main()
