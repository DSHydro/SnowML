#!/usr/bin/env python3
"""
Evaluate 4 Exp1B Variations (Wind, Humidity, Base, Srad) - CORRECT EPOCHS
Based on working evaluate_exp1b_final.py (Aug 30)
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
print("EXPERIMENT 1B - 4 VARIATIONS EVALUATION (CORRECT EPOCHS)")
print("Using Original Exp3 Methodology")
print("="*80)
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# =============================================================================
# CONFIGURATION - CORRECT EPOCHS FROM training_log.txt
# =============================================================================

# Checkpoint files (CORRECT best epochs)
CHECKPOINTS = {
    "Wind_3e-4": {
        "file": "/home/sagemaker-user/checkpoints_best/Exp1B_Wind_3e-4_epoch6.pth",
        "val_kge": 0.8104
    },
    "Humidity_3e-4": {
        "file": "/home/sagemaker-user/checkpoints_best/Exp1B_Humidity_3e-4_epoch3.pth",
        "val_kge": 0.7971
    },
    "Base_3e-4": {
        "file": "/home/sagemaker-user/checkpoints_best/Exp1B_Base_3e-4_epoch14.pth",
        "val_kge": 0.8084
    },
    "Srad_3e-4": {
        "file": "/home/sagemaker-user/checkpoints_best/Exp1B_Srad_3e-4_epoch6.pth",
        "val_kge": 0.8018
    }
}

# Test set HUC lists
TEST_SETS = {
    "Test_A": "/home/sagemaker-user/test_sets/test_a_hucs.txt",
    "Test_B": "/home/sagemaker-user/test_sets/test_b_hucs.txt"
}

# Output directory
OUTPUT_DIR = Path("/home/sagemaker-user/exp1b_corrected_results")
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
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

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

    # Storage for all results
    all_results = []
    summary_data = []

    # Evaluate each model
    for model_name, config in CHECKPOINTS.items():
        print("\n" + "="*80)
        print(f"EVALUATING: {model_name}")
        print(f"Validation KGE: {config['val_kge']:.4f}")
        print("="*80 + "\n")

        checkpoint_path = config["file"]

        # Check if checkpoint exists
        if not Path(checkpoint_path).exists():
            print(f"❌ Checkpoint not found: {checkpoint_path}")
            print("Available checkpoints:")
            checkpoint_dir = Path(checkpoint_path).parent
            for f in checkpoint_dir.glob("*.pth"):
                print(f"  - {f.name}")
            continue

        try:
            # Load model and params
            model, params = load_checkpoint(checkpoint_path)
            print()

            # Evaluate on Test Set A
            print(f"{'='*80}")
            print(f"TEST SET A")
            print(f"{'='*80}")
            results_a = evaluate_model_on_test_set(model, params, test_sets["Test_A"], "Test_A")

            # Save Test A results
            if len(results_a) > 0:
                results_a['variation'] = model_name
                results_a['test_set'] = 'Test_A'
                output_file_a = OUTPUT_DIR / f"{model_name}_Test_A_metrics.csv"
                results_a.to_csv(output_file_a, index=False)
                print(f"\n✅ Saved: {output_file_a}")
                all_results.append(results_a)
                test_a_kge = results_a['test_kge'].median()
            else:
                test_a_kge = None

            # Evaluate on Test Set B
            print(f"\n{'='*80}")
            print(f"TEST SET B (Yakima/Naches)")
            print(f"{'='*80}")
            results_b = evaluate_model_on_test_set(model, params, test_sets["Test_B"], "Test_B")

            # Save Test B results
            if len(results_b) > 0:
                results_b['variation'] = model_name
                results_b['test_set'] = 'Test_B'
                output_file_b = OUTPUT_DIR / f"{model_name}_Test_B_metrics.csv"
                results_b.to_csv(output_file_b, index=False)
                print(f"\n✅ Saved: {output_file_b}")
                all_results.append(results_b)
                test_b_kge = results_b['test_kge'].median()
            else:
                test_b_kge = None

            # Add to summary
            if test_a_kge is not None and test_b_kge is not None:
                val_kge = config['val_kge']
                val_drop_a = ((val_kge - test_a_kge) / val_kge) * 100
                val_drop_b = ((val_kge - test_b_kge) / val_kge) * 100

                summary_data.append({
                    'model': model_name,
                    'validation_kge': val_kge,
                    'test_a_kge': test_a_kge,
                    'test_b_kge': test_b_kge,
                    'val_to_test_a_drop_%': val_drop_a,
                    'val_to_test_b_drop_%': val_drop_b
                })

            print(f"\n✅ {model_name} COMPLETE!")

        except Exception as e:
            print(f"\n❌ Error with {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Save combined results
    if all_results:
        print("\n" + "="*80)
        print("SAVING COMBINED RESULTS")
        print("="*80)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Save all results to one CSV
        combined_df = pd.concat(all_results, ignore_index=True)
        combined_file = OUTPUT_DIR / f"all_4_variations_complete_{timestamp}.csv"
        combined_df.to_csv(combined_file, index=False)
        print(f"\n✅ All results: {combined_file}")

        # Save summary
        summary_df = pd.DataFrame(summary_data)
        summary_file = OUTPUT_DIR / f"summary_4_variations_{timestamp}.csv"
        summary_df.to_csv(summary_file, index=False)
        print(f"✅ Summary: {summary_file}")

        # Print summary table
        print("\n" + "="*80)
        print("FINAL SUMMARY - ALL 4 VARIATIONS")
        print("="*80)
        print(summary_df.to_string(index=False))
        print("="*80)

    print(f"\n✅ EVALUATION COMPLETE!")
    print(f"\nResults saved to: {OUTPUT_DIR}/")
    print(f"\nExpected pattern (validation >= test):")
    print(f"  Validation: 0.80-0.81")
    print(f"  Test A: 0.75-0.80 (5-10% drop)")
    print(f"  Test B: 0.70-0.76 (10-15% drop)")

if __name__ == '__main__':
    main()
