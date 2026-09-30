#!/usr/bin/env python3
"""
Evaluate Exp1A Best Models on Test Sets A and B
================================================

Test Set A: 92 random held-out HUCs (spatial generalization)
Test Set B: 81 Yakima/Naches HUCs (transfer to unseen region)

Models to evaluate:
1. Exp1A_Wind_3e-4_epoch12 (Best: KGE 0.7598)
2. Exp1A_Humidity_3e-4_epoch2 (Second: KGE 0.7176)
3. Exp1A_Base_3e-4_epoch5 (Baseline: KGE 0.7081)

Optional: Also evaluate Exp1B best model for comparison
4. Exp1B_Wind_3e-4_epoch6 (Best from Exp1B: KGE 0.8104)
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
from datetime import datetime
from pathlib import Path

# Add snowML to path
sys.path.insert(0, '/home/sagemaker-user/src')
print(f"Added to path: /home/sagemaker-user/src")

# Try to import - if it fails, we'll get a helpful error
try:
    from snowML.LSTM import LSTM_train, set_hyperparams
    from snowML.LSTM.set_hyperparams import create_hyper_dict
    print("✅ Successfully imported snowML modules")
except ImportError as e:
    print(f"❌ Failed to import snowML: {e}")
    print(f"Python path: {sys.path}")
    print(f"Current directory: {os.getcwd()}")
    print("\nPlease check:")
    print("1. Where is SnowML located on your SageMaker instance?")
    print("2. Is the package installed? (pip install -e /path/to/SnowML)")
    sys.exit(1)

# Configuration
CHECKPOINT_DIR = "/home/sagemaker-user/checkpoints"
OUTPUT_DIR = "/home/sagemaker-user/evaluation_results"
DATA_DIR = "/home/sagemaker-user/src/data"

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Models to evaluate
MODELS_TO_EVALUATE = [
    {
        'name': 'Exp1A_Wind_3e-4_epoch12',
        'checkpoint': 'Exp1A_Wind_3e-4_epoch12.pth',
        'experiment': 'Exp1A',
        'features': ['mean_pr', 'mean_tair', 'Mean Elevation', 'mean_vs'],
        'lr': 0.0003,
        'description': 'Best Exp1A model (Val KGE 0.7598)'
    },
    {
        'name': 'Exp1A_Humidity_3e-4_epoch2',
        'checkpoint': 'Exp1A_Humidity_3e-4_epoch2.pth',
        'experiment': 'Exp1A',
        'features': ['mean_pr', 'mean_tair', 'Mean Elevation', 'mean_hum'],
        'lr': 0.0003,
        'description': 'Second best Exp1A (Val KGE 0.7176)'
    },
    {
        'name': 'Exp1A_Base_3e-4_epoch5',
        'checkpoint': 'Exp1A_Base_3e-4_epoch5.pth',
        'experiment': 'Exp1A',
        'features': ['mean_pr', 'mean_tair', 'Mean Elevation'],
        'lr': 0.0003,
        'description': 'Baseline Exp1A (Val KGE 0.7081)'
    },
]

# Optional: Add Exp1B for comparison
INCLUDE_EXP1B = True
if INCLUDE_EXP1B:
    MODELS_TO_EVALUATE.append({
        'name': 'Exp1B_Wind_3e-4_epoch6',
        'checkpoint': 'Exp1B_Wind_3e-4_epoch6.pth',
        'experiment': 'Exp1B',
        'features': ['mean_pr', 'mean_tair', 'Mean Elevation', 'mean_vs'],
        'lr': 0.0003,
        'description': 'Best Exp1B model (Val KGE 0.8104) - for comparison'
    })


def load_huc_list(file_path):
    """Load HUC IDs from text file"""
    with open(file_path, 'r') as f:
        hucs = [line.strip() for line in f if line.strip()]
    return hucs


def setup_params(features, lr):
    """Setup hyperparameters for model"""
    params = create_hyper_dict()

    # Model architecture (same as training)
    params["hidden_size"] = 64
    params["num_layers"] = 1
    params["dropout"] = 0.5
    params["batch_size"] = 32
    params["lookback"] = 180

    # Feature selection
    params["feature_names"] = features
    params["target_names"] = ["SWE"]

    # Other settings
    params["learning_rate"] = lr
    params["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    params["bucket_name"] = "snowml-model-ready"
    params["region_name"] = "us-west-2"

    # Evaluation settings
    params["train_size_dimension"] = "huc"
    params["train_size_fraction"] = 1.0  # Use all data for evaluation

    return params


def evaluate_model_on_hucs(model_info, huc_list, test_set_name):
    """Evaluate a model on a list of HUCs"""

    print(f"\n{'='*80}")
    print(f"Evaluating: {model_info['name']}")
    print(f"Test Set: {test_set_name} ({len(huc_list)} HUCs)")
    print(f"Description: {model_info['description']}")
    print(f"{'='*80}\n")

    # Load checkpoint
    checkpoint_path = os.path.join(CHECKPOINT_DIR, model_info['checkpoint'])
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return None

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)

    # Setup parameters
    params = setup_params(model_info['features'], model_info['lr'])

    # Initialize model
    from snowML.LSTM.LSTM_train import initialize_model
    model, _, _ = initialize_model(params)

    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"✅ Model loaded successfully")
    print(f"Device: {params['device']}")
    print(f"Features: {params['feature_names']}")

    # Evaluate on each HUC
    results = []

    for idx, huc_id in enumerate(huc_list, 1):
        if idx % 10 == 0:
            print(f"Progress: {idx}/{len(huc_list)} HUCs evaluated")

        try:
            # Load data for this HUC
            from snowML.LSTM.LSTM_train import get_huc_data
            df_huc = get_huc_data(huc_id, params)

            if df_huc is None or len(df_huc) == 0:
                print(f"⚠️  No data for HUC {huc_id}, skipping")
                continue

            # Split into train/test (we'll use test portion for evaluation)
            # Since this is held-out data, we evaluate on the full time series
            from snowML.LSTM.LSTM_train import evaluate
            metrics = evaluate(model, df_huc, params)

            results.append({
                'huc_id': huc_id,
                'test_kge': metrics.get('test_kge', np.nan),
                'test_mse': metrics.get('test_mse', np.nan),
                'test_mae': metrics.get('test_mae', np.nan),
                'test_r2': metrics.get('test_r2', np.nan),
            })

        except Exception as e:
            print(f"⚠️  Error evaluating HUC {huc_id}: {e}")
            results.append({
                'huc_id': huc_id,
                'test_kge': np.nan,
                'test_mse': np.nan,
                'test_mae': np.nan,
                'test_r2': np.nan,
            })

    # Convert to DataFrame
    df_results = pd.DataFrame(results)

    # Calculate summary statistics
    print(f"\n{'='*80}")
    print(f"RESULTS SUMMARY: {model_info['name']} on {test_set_name}")
    print(f"{'='*80}")

    valid_results = df_results[df_results['test_kge'].notna()]

    if len(valid_results) > 0:
        print(f"\nSuccessfully evaluated: {len(valid_results)}/{len(huc_list)} HUCs")
        print(f"\nTest KGE Statistics:")
        print(f"  Median: {valid_results['test_kge'].median():.4f}")
        print(f"  Mean:   {valid_results['test_kge'].mean():.4f}")
        print(f"  Std:    {valid_results['test_kge'].std():.4f}")
        print(f"  Min:    {valid_results['test_kge'].min():.4f}")
        print(f"  Max:    {valid_results['test_kge'].max():.4f}")
        print(f"  Q1:     {valid_results['test_kge'].quantile(0.25):.4f}")
        print(f"  Q3:     {valid_results['test_kge'].quantile(0.75):.4f}")

        print(f"\nTest MSE Statistics:")
        print(f"  Median: {valid_results['test_mse'].median():.6f}")
        print(f"  Mean:   {valid_results['test_mse'].mean():.6f}")

        print(f"\nTest R² Statistics:")
        print(f"  Median: {valid_results['test_r2'].median():.4f}")
        print(f"  Mean:   {valid_results['test_r2'].mean():.4f}")
    else:
        print("❌ No valid results")

    return df_results


def main():
    """Main evaluation script"""

    print("="*80)
    print("EXPERIMENT 1A TEST SET EVALUATION")
    print("="*80)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"\nModels to evaluate: {len(MODELS_TO_EVALUATE)}")
    for m in MODELS_TO_EVALUATE:
        print(f"  - {m['name']}: {m['description']}")

    # Load test HUC lists
    print(f"\n{'='*80}")
    print("Loading Test HUC Lists")
    print(f"{'='*80}")

    test_a_file = os.path.join(DATA_DIR, "exp1a_test_a_hucs.txt")
    test_b_file = os.path.join(DATA_DIR, "exp1a_test_b_hucs.txt")

    if not os.path.exists(test_a_file):
        print(f"❌ Test Set A file not found: {test_a_file}")
        return
    if not os.path.exists(test_b_file):
        print(f"❌ Test Set B file not found: {test_b_file}")
        return

    test_a_hucs = load_huc_list(test_a_file)
    test_b_hucs = load_huc_list(test_b_file)

    print(f"✅ Test Set A: {len(test_a_hucs)} HUCs (random held-out)")
    print(f"✅ Test Set B: {len(test_b_hucs)} HUCs (Yakima/Naches)")

    # Check GPU availability
    if torch.cuda.is_available():
        print(f"\n✅ GPU available: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print(f"\n⚠️  No GPU available, using CPU")

    # Evaluate each model on both test sets
    all_results = {}

    for model_info in MODELS_TO_EVALUATE:
        model_name = model_info['name']

        # Evaluate on Test Set A
        print(f"\n{'#'*80}")
        print(f"TEST SET A - {model_name}")
        print(f"{'#'*80}")
        df_test_a = evaluate_model_on_hucs(model_info, test_a_hucs, "Test Set A")

        if df_test_a is not None:
            output_file_a = os.path.join(OUTPUT_DIR, f"{model_name}_TestSetA_results.csv")
            df_test_a.to_csv(output_file_a, index=False)
            print(f"💾 Saved results to: {output_file_a}")
            all_results[f"{model_name}_TestA"] = df_test_a

        # Evaluate on Test Set B
        print(f"\n{'#'*80}")
        print(f"TEST SET B - {model_name}")
        print(f"{'#'*80}")
        df_test_b = evaluate_model_on_hucs(model_info, test_b_hucs, "Test Set B")

        if df_test_b is not None:
            output_file_b = os.path.join(OUTPUT_DIR, f"{model_name}_TestSetB_results.csv")
            df_test_b.to_csv(output_file_b, index=False)
            print(f"💾 Saved results to: {output_file_b}")
            all_results[f"{model_name}_TestB"] = df_test_b

    # Create comparison summary
    print(f"\n{'='*80}")
    print("FINAL COMPARISON SUMMARY")
    print(f"{'='*80}\n")

    summary_rows = []

    for model_info in MODELS_TO_EVALUATE:
        model_name = model_info['name']

        # Test Set A
        if f"{model_name}_TestA" in all_results:
            df = all_results[f"{model_name}_TestA"]
            valid = df[df['test_kge'].notna()]
            if len(valid) > 0:
                summary_rows.append({
                    'Model': model_name,
                    'Test_Set': 'A (Random)',
                    'N_HUCs': len(valid),
                    'Median_KGE': valid['test_kge'].median(),
                    'Mean_KGE': valid['test_kge'].mean(),
                    'Median_MSE': valid['test_mse'].median(),
                    'Median_R2': valid['test_r2'].median(),
                })

        # Test Set B
        if f"{model_name}_TestB" in all_results:
            df = all_results[f"{model_name}_TestB"]
            valid = df[df['test_kge'].notna()]
            if len(valid) > 0:
                summary_rows.append({
                    'Model': model_name,
                    'Test_Set': 'B (Yakima/Naches)',
                    'N_HUCs': len(valid),
                    'Median_KGE': valid['test_kge'].median(),
                    'Mean_KGE': valid['test_kge'].mean(),
                    'Median_MSE': valid['test_mse'].median(),
                    'Median_R2': valid['test_r2'].median(),
                })

    df_summary = pd.DataFrame(summary_rows)

    if len(df_summary) > 0:
        print(df_summary.to_string(index=False))

        # Save summary
        summary_file = os.path.join(OUTPUT_DIR, "evaluation_summary.csv")
        df_summary.to_csv(summary_file, index=False)
        print(f"\n💾 Summary saved to: {summary_file}")

    print(f"\n{'='*80}")
    print("EVALUATION COMPLETE!")
    print(f"{'='*80}")
    print(f"All results saved to: {OUTPUT_DIR}")
    print(f"\nNext steps:")
    print(f"1. Review results CSVs for each model")
    print(f"2. Compare Exp1A vs Exp1B on Test Set B (Yakima/Naches)")
    print(f"3. Proceed to Experiment 2 (fine-tuning) using best model")


if __name__ == '__main__':
    main()
