#!/usr/bin/env python3
"""
Evaluate Exp1A Best Models on Test Sets A & B
==============================================

Based on evaluate_exp1b_final.py (which works correctly)
Uses the EXACT same evaluation methodology as original Exp3 students

Models to evaluate:
1. Exp1A_Wind_3e-4_epoch12 (Best: Val KGE 0.7598)
2. Exp1A_Humidity_3e-4_epoch2 (Second: Val KGE 0.7176)
3. Exp1A_Base_3e-4_epoch5 (Baseline: Val KGE 0.7081)

Optional: Exp1B_Wind_3e-4_epoch6 for comparison (Val KGE 0.8104)
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
print("EXPERIMENT 1A - TEST SET EVALUATION")
print("Using Original Exp3 Evaluation Methodology")
print("="*80)
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# =============================================================================
# CONFIGURATION
# =============================================================================

# Checkpoint files (best epochs from Exp1A training)
CHECKPOINTS = {
    "Exp1A_Wind_3e-4_epoch12": {
        "file": "/home/sagemaker-user/checkpoints/Exp1A_Wind_3e-4_epoch12.pth",
        "val_kge": 0.7598,
        "description": "Best Exp1A model"
    },
    "Exp1A_Humidity_3e-4_epoch2": {
        "file": "/home/sagemaker-user/checkpoints/Exp1A_Humidity_3e-4_epoch2.pth",
        "val_kge": 0.7176,
        "description": "Second best Exp1A"
    },
    "Exp1A_Base_3e-4_epoch5": {
        "file": "/home/sagemaker-user/checkpoints/Exp1A_Base_3e-4_epoch5.pth",
        "val_kge": 0.7081,
        "description": "Baseline Exp1A"
    },
}

# OPTIONAL: Add Exp1B best model for comparison
INCLUDE_EXP1B = True

if INCLUDE_EXP1B:
    CHECKPOINTS["Exp1B_Wind_3e-4_epoch6"] = {
        "file": "/home/sagemaker-user/checkpoints/Exp1B_Wind_3e-4_epoch6.pth",
        "val_kge": 0.8104,
        "description": "Best Exp1B (for comparison)"
    }

# Test set HUC lists (Exp1A uses same Test Set B as Exp1B)
TEST_SETS = {
    "Test_A": "/home/sagemaker-user/src/data/exp1a_test_a_hucs.txt",
    "Test_B": "/home/sagemaker-user/src/data/exp1a_test_b_hucs.txt"
}

# Output directory
OUTPUT_DIR = Path("/home/sagemaker-user/exp1a_evaluation_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_huc_list(file_path):
    """Load HUC IDs from text file"""
    print(f"  Loading: {file_path}")
    with open(file_path, 'r') as f:
        hucs = [line.strip() for line in f if line.strip()]
    print(f"  ✅ Loaded {len(hucs)} HUCs")
    return hucs

def load_train_val_hucs_from_files(experiment_name):
    """
    Load train and val HUCs from split files
    Returns lists of HUC IDs for training and validation
    """
    if 'Exp1A' in experiment_name:
        train_file = "/home/sagemaker-user/src/data/exp1a_train_hucs.txt"
        val_file = "/home/sagemaker-user/src/data/exp1a_validation_hucs.txt"
    elif 'Exp1B' in experiment_name:
        train_file = "/home/sagemaker-user/src/data/exp1b_train_hucs.txt"
        val_file = "/home/sagemaker-user/src/data/exp1b_validation_hucs.txt"
    else:
        return [], []

    train_hucs = load_huc_list(train_file)
    val_hucs = load_huc_list(val_file)

    return train_hucs, val_hucs

def load_checkpoint(checkpoint_path, model_name):
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

    # FIX: Load train/val HUCs from split files if not in checkpoint
    if 'train_hucs' not in params or 'val_hucs' not in params or \
       len(params.get('train_hucs', [])) == 0 or len(params.get('val_hucs', [])) == 0:
        print(f"  ⚠️  train_hucs/val_hucs missing from checkpoint")
        print(f"  📂 Loading from split files...")

        train_hucs, val_hucs = load_train_val_hucs_from_files(model_name)

        if len(train_hucs) > 0 and len(val_hucs) > 0:
            params['train_hucs'] = train_hucs
            params['val_hucs'] = val_hucs
            print(f"  ✅ Loaded {len(train_hucs)} train + {len(val_hucs)} val HUCs from files")
        else:
            print(f"  ❌ Failed to load train/val HUCs from files!")

    # Create model (using LSTM_model.SnowModel)
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

    print(f"  ✅ Model loaded successfully")

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

    # Check if train_hucs and val_hucs are in params
    if 'train_hucs' not in params or 'val_hucs' not in params or \
       len(params.get('train_hucs', [])) == 0 or len(params.get('val_hucs', [])) == 0:
        print("❌ ERROR: train_hucs/val_hucs not in checkpoint!")
        print("   Cannot perform proper normalization without training HUC stats")
        print("   This checkpoint may be incomplete")
        return pd.DataFrame()

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

    # Move model to GPU if available (same as training)
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
            print(f"   Median MSE: {results_df['test_mse'].median():.6f}")

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

    print()

    # Check GPU
    if torch.cuda.is_available():
        print(f"✅ GPU available: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB\n")
    else:
        print("⚠️  No GPU available, using CPU\n")

    # Evaluate each model on each test set
    all_results = {}

    for model_name, checkpoint_info in CHECKPOINTS.items():
        print("\n" + "="*80)
        print(f"MODEL: {model_name}")
        print(f"Description: {checkpoint_info['description']}")
        print("="*80)

        # Check if checkpoint exists
        if not Path(checkpoint_info['file']).exists():
            print(f"❌ Checkpoint not found: {checkpoint_info['file']}")
            print(f"   Skipping this model\n")
            continue

        # Load checkpoint
        try:
            model, params = load_checkpoint(checkpoint_info['file'], model_name)
        except Exception as e:
            print(f"❌ Failed to load checkpoint: {e}")
            print(f"   Skipping this model\n")
            continue

        # Evaluate on each test set
        for test_name, test_hucs in test_sets.items():
            results_df = evaluate_model_on_test_set(
                model, params, test_hucs, test_name
            )

            # Store results
            key = f"{model_name}_{test_name}"
            all_results[key] = results_df

            # Save individual CSV
            if len(results_df) > 0:
                output_file = OUTPUT_DIR / f"{key}_metrics.csv"
                results_df.to_csv(output_file, index=False)
                print(f"💾 Saved: {output_file}")

    # Create comparison summary
    print("\n" + "="*80)
    print("FINAL SUMMARY - VALIDATION vs TEST SETS")
    print("="*80)

    comparison = []

    for model_name, checkpoint_info in CHECKPOINTS.items():
        row = {
            'Model': model_name,
            'Description': checkpoint_info['description'],
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
    if len(comparison) > 0:
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
            print(f"  {row['Description']}")
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

        # Compare Exp1A vs Exp1B (if both present)
        print("\n" + "="*80)
        print("COMPARISON: EXP1A vs EXP1B")
        print("="*80)

        exp1a_wind = comparison_df[comparison_df['Model'].str.contains('Exp1A_Wind')]
        exp1b_wind = comparison_df[comparison_df['Model'].str.contains('Exp1B_Wind')]

        if len(exp1a_wind) > 0 and len(exp1b_wind) > 0:
            print("\nWind Model Comparison:")
            print(f"  Exp1A (all 533 HUCs):")
            print(f"    Validation KGE: {exp1a_wind.iloc[0]['Validation_KGE']:.4f}")
            if 'Test_B_KGE_median' in exp1a_wind.columns:
                print(f"    Test Set B KGE: {exp1a_wind.iloc[0]['Test_B_KGE_median']:.4f}")

            print(f"\n  Exp1B (270 deep HUCs only):")
            print(f"    Validation KGE: {exp1b_wind.iloc[0]['Validation_KGE']:.4f}")
            if 'Test_B_KGE_median' in exp1b_wind.columns:
                print(f"    Test Set B KGE: {exp1b_wind.iloc[0]['Test_B_KGE_median']:.4f}")

            print("\n  KEY FINDING:")
            print("  - Does including ephemeral basins help or hurt?")
            print("  - Compare the Validation and Test Set B KGE values above")

    print("\n" + "="*80)
    print("✅ EVALUATION COMPLETE!")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

    print(f"\n📁 All results saved to: {OUTPUT_DIR}")
    print("\nFiles generated:")
    for f in sorted(OUTPUT_DIR.glob("*.csv")):
        print(f"  - {f.name}")

    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print("1. Download results to your local machine")
    print("2. Review validation_vs_test_comparison.csv")
    print("3. Compare Exp1A vs Exp1B performance")
    print("4. Decide which model to use for Experiment 2 (fine-tuning)")
    print("="*80)

if __name__ == "__main__":
    main()
