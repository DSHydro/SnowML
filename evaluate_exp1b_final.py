#!/usr/bin/env python3
"""
Evaluate Wind_3e-4 and Humidity_3e-4 on Test Sets A & B
Uses the EXACT same evaluation code as original Exp3 students
BUT loads from checkpoint files instead of MLflow

FIXED VERSION - All issues resolved:
- Correct paths for checkpoints and test sets
- Loads train/val HUCs from hucs_data.json
- Proper GPU/CPU device handling
"""

import sys
import torch
import json
import pandas as pd
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, '/home/sagemaker-user/src')

from snowML.LSTM import LSTM_evaluate as evaluate
from snowML.LSTM import LSTM_model as LSTM_mod
from snowML.LSTM import LSTM_metrics as met

print("="*80)
print("EXPERIMENT 1B - TEST SET EVALUATION")
print("Using Original Exp3 Evaluation Methodology")
print("="*80)
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# =============================================================================
# CONFIGURATION
# =============================================================================

# Checkpoint files (best epochs from training)
CHECKPOINTS = {
    "Wind_3e-4": {
        "file": "/home/sagemaker-user/checkpoints_best/Exp1B_Wind_3e-4_epoch6.pth",
        "val_kge": 0.8104
    },
    "Humidity_3e-4": {
        "file": "/home/sagemaker-user/checkpoints_best/Exp1B_Humidity_3e-4_epoch7.pth",
        "val_kge": 0.6212
    }
}

# Test set HUC lists
TEST_SETS = {
    "Test_A": "/home/sagemaker-user/test_sets/test_a_hucs.txt",
    "Test_B": "/home/sagemaker-user/test_sets/test_b_hucs.txt"
}

# Output directory
OUTPUT_DIR = Path("/home/sagemaker-user/exp1b_evaluation_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_train_val_hucs_from_json():
    """
    Load train and val HUCs from the JSON file that was used during training.
    This is needed because checkpoints may not have these saved.
    """
    json_path = "/home/sagemaker-user/src/snowML/datapipe/huc_lists/hucs_data.json"

    print(f"  Loading train/val HUCs from: {json_path}")

    with open(json_path, 'r') as f:
        data = json.load(f)

    train_hucs = data['train_hucs']
    val_hucs = data['val_hucs']

    print(f"  ✅ Loaded {len(train_hucs)} train HUCs")
    print(f"  ✅ Loaded {len(val_hucs)} val HUCs")

    return train_hucs, val_hucs

def load_huc_list(file_path):
    """Load HUC IDs from text file"""
    with open(file_path, 'r') as f:
        hucs = [line.strip() for line in f if line.strip()]
    return hucs

def load_checkpoint(checkpoint_path):
    """
    Load checkpoint and extract model + params
    Returns model and params in format compatible with original evaluate.py
    """
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Get params from checkpoint
    params = checkpoint['params']

    print(f"  Features: {params['var_list']}")
    print(f"  Train HUCs: {len(params.get('train_hucs', []))} HUCs")
    print(f"  Val HUCs: {len(params.get('val_hucs', []))} HUCs")
    print(f"  Learning rate: {params['learning_rate']}")
    print(f"  Train dimension: {params.get('train_size_dimension', 'time')}")

    # Create model
    model = LSTM_mod.SnowModel(
        input_size=len(params['var_list']),
        hidden_size=params.get('hidden_size', 64),
        num_class=1,
        num_layers=params.get('num_layers', 1),
        dropout=params.get('dropout', 0.5)
    )

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, params

def evaluate_model_on_test_set(model, params, test_hucs, test_name):
    """
    Evaluate model on test set using ORIGINAL Exp3 methodology

    This function replicates exactly what the original students did:
    1. Load test HUC data with renormalization (using train HUC stats)
    2. Evaluate each HUC using eval_from_saved_model()
    3. Collect metrics
    """
    print(f"\n{'='*80}")
    print(f"Evaluating on: {test_name} ({len(test_hucs)} HUCs)")
    print(f"{'='*80}\n")

    # FIX 1: Load train/val HUCs if not in checkpoint or if empty
    if 'train_hucs' not in params or 'val_hucs' not in params or \
       len(params.get('train_hucs', [])) == 0 or len(params.get('val_hucs', [])) == 0:
        print("⚠️  train_hucs/val_hucs not in checkpoint, loading from hucs_data.json...")
        train_hucs, val_hucs = load_train_val_hucs_from_json()
        params['train_hucs'] = train_hucs
        params['val_hucs'] = val_hucs
        print()

    # Prepare test data using ORIGINAL Exp3 method
    # This handles normalization correctly!
    if params["train_size_dimension"] == "huc":
        # Use renorm() - same as original Exp3
        print("Loading test HUC data...")
        df_dict_test = evaluate.assemble_df_dict(test_hucs, params["var_list"])

        print("Normalizing test data using training statistics...")
        df_dict_test = evaluate.renorm(
            params["train_hucs"],
            params["val_hucs"],
            test_hucs,
            params["var_list"]
        )
        print("✅ Using HUC-based normalization (same global stats as training)")
        print(f"   Normalizing using {len(params['train_hucs'])} train + {len(params['val_hucs'])} val HUCs")
    else:
        # Per-HUC normalization (shouldn't happen for your models)
        from snowML.LSTM import LSTM_pre_process as pp
        df_dict_test = pp.pre_process_separate(test_hucs, params["var_list"])
        print("⚠️  Using per-HUC normalization")

    print(f"✅ Loaded data for {len(df_dict_test)} HUCs\n")

    # FIX 2: Move model to GPU if available (same as training)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    params['device'] = device
    print(f"✅ Model moved to: {device}\n")

    # Evaluate each HUC using ORIGINAL eval_from_saved_model()
    all_metrics = []

    for i, huc in enumerate(test_hucs, 1):
        if i % 10 == 0:
            print(f"  Progress: {i}/{len(test_hucs)} HUCs evaluated")

        if huc not in df_dict_test:
            print(f"  ⚠️  Skipping {huc} - no data")
            continue

        try:
            # Use ORIGINAL evaluation function - ensures identical methodology!
            metric_dict_test, metric_dict_test_recur, data, y_tr_pred, \
                y_te_pred, y_tr_true, y_te_true, y_te_pred_recur, train_size = \
                evaluate.eval_from_saved_model(model, df_dict_test, huc, params)

            # Add HUC ID to metrics
            metric_dict_test['huc_id'] = huc
            all_metrics.append(metric_dict_test)

        except Exception as e:
            print(f"  ❌ Error evaluating {huc}: {e}")
            continue

    print(f"\n✅ Completed: {len(all_metrics)}/{len(test_hucs)} HUCs\n")

    # Convert to DataFrame
    results_df = pd.DataFrame(all_metrics)

    # Print summary statistics (same as original Exp3 notebooks)
    if len(results_df) > 0:
        print(f"📊 Summary Statistics:")
        print(f"   Median KGE: {results_df['test_kge'].median():.4f}")
        print(f"   Mean KGE: {results_df['test_kge'].mean():.4f}")
        print(f"   Std KGE: {results_df['test_kge'].std():.4f}")
        print(f"   Min KGE: {results_df['test_kge'].min():.4f}")
        print(f"   Max KGE: {results_df['test_kge'].max():.4f}")

        if 'test_nse' in results_df.columns:
            print(f"   Median NSE: {results_df['test_nse'].median():.4f}")
        if 'test_mse' in results_df.columns:
            print(f"   Median MSE: {results_df['test_mse'].median():.4f}")

    return results_df

# =============================================================================
# MAIN EVALUATION
# =============================================================================

def main():
    """Run full evaluation"""

    # Load test sets
    print("📋 Loading test sets...\n")
    test_sets = {}
    for name, file_path in TEST_SETS.items():
        hucs = load_huc_list(file_path)
        test_sets[name] = hucs
        print(f"  {name}: {len(hucs)} HUCs")

    print()

    # Evaluate each model on each test set
    all_results = {}

    for model_name, checkpoint_info in CHECKPOINTS.items():
        print("\n" + "="*80)
        print(f"MODEL: {model_name}")
        print("="*80)

        # Load checkpoint
        model, params = load_checkpoint(checkpoint_info['file'])

        # Evaluate on each test set
        for test_name, test_hucs in test_sets.items():
            results_df = evaluate_model_on_test_set(
                model, params, test_hucs, test_name
            )

            # Store results
            key = f"{model_name}_{test_name}"
            all_results[key] = results_df

            # Save individual CSV
            output_file = OUTPUT_DIR / f"{key}_metrics.csv"
            results_df.to_csv(output_file, index=False)
            print(f"💾 Saved: {output_file}")

    # Create comparison summary (like original Exp3 notebooks)
    print("\n" + "="*80)
    print("FINAL SUMMARY - VALIDATION vs TEST SETS")
    print("="*80)

    comparison = []

    for model_name, checkpoint_info in CHECKPOINTS.items():
        row = {
            'Model': model_name,
            'Validation_KGE': checkpoint_info['val_kge']
        }

        # Add test set results
        for test_name in test_sets.keys():
            key = f"{model_name}_{test_name}"
            if key in all_results and len(all_results[key]) > 0:
                df = all_results[key]
                row[f'{test_name}_KGE_median'] = round(df['test_kge'].median(), 4)
                row[f'{test_name}_KGE_mean'] = round(df['test_kge'].mean(), 4)
                row[f'{test_name}_n_hucs'] = len(df)

                # Calculate drop from validation
                val_kge = checkpoint_info['val_kge']
                test_kge = df['test_kge'].median()
                drop_pct = ((val_kge - test_kge) / val_kge) * 100
                row[f'{test_name}_drop_pct'] = round(drop_pct, 1)

        comparison.append(row)

    # Save and display comparison
    comparison_df = pd.DataFrame(comparison)
    comparison_file = OUTPUT_DIR / "validation_vs_test_comparison.csv"
    comparison_df.to_csv(comparison_file, index=False)

    print("\n" + comparison_df.to_string(index=False))
    print(f"\n💾 Comparison saved: {comparison_file}")

    # Print interpretation
    print("\n" + "="*80)
    print("INTERPRETATION")
    print("="*80)

    for _, row in comparison_df.iterrows():
        print(f"\n{row['Model']}:")
        print(f"  Validation KGE: {row['Validation_KGE']:.4f}")

        for test_name in test_sets.keys():
            median_col = f'{test_name}_KGE_median'
            drop_col = f'{test_name}_drop_pct'

            if median_col in row and pd.notna(row[median_col]):
                print(f"  {test_name} KGE: {row[median_col]:.4f} ({row[drop_col]:.1f}% drop)")

                # Interpretation
                if row[drop_col] < 10:
                    status = "✅ EXCELLENT"
                elif row[drop_col] < 20:
                    status = "✅ GOOD"
                elif row[drop_col] < 30:
                    status = "⚠️  FAIR"
                else:
                    status = "❌ POOR"

                print(f"     {status} generalization")

    # Compare to original Exp3
    print("\n" + "="*80)
    print("COMPARISON TO ORIGINAL EXP3 BASELINE")
    print("="*80)
    print("\nOriginal Exp3 Multi-HUC Results:")
    print("  Validation KGE: ~0.82")
    print("  Test Set A KGE: ~0.72 (12% drop)")
    print("\nYour Results:")
    for _, row in comparison_df.iterrows():
        print(f"  {row['Model']}:")
        print(f"    Validation: {row['Validation_KGE']:.4f} vs 0.82 original")
        if 'Test_A_KGE_median' in row:
            print(f"    Test A: {row['Test_A_KGE_median']:.4f} vs 0.72 original")

    print("\n" + "="*80)
    print("✅ EVALUATION COMPLETE!")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

    print(f"\n📁 All results saved to: {OUTPUT_DIR}")
    print("\nFiles generated:")
    for f in sorted(OUTPUT_DIR.glob("*.csv")):
        print(f"  - {f.name}")

if __name__ == "__main__":
    main()
