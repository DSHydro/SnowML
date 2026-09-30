#!/usr/bin/env python3
"""
Evaluate Exp1A Srad_3e-4_epoch2 on Test Sets A & B
Based on working evaluate_4_variations_corrected.py
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

print("="*80)
print("EXPERIMENT 1A - SRAD EVALUATION")
print("Evaluating Exp1A_Srad_3e-4_epoch2 on Test A & B")
print("="*80)
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# =============================================================================
# CONFIGURATION
# =============================================================================

# Checkpoint file
CHECKPOINT = {
    "file": "/home/sagemaker-user/checkpoints/Exp1A_Srad_3e-4_epoch2.pth",
    "val_kge": 0.6855,
    "name": "Exp1A_Srad_3e-4_epoch2"
}

# Test set HUC lists (Exp1A test sets)
TEST_SETS = {
    "Test_A": "/home/sagemaker-user/test_sets/exp1a_test_a_hucs.txt",
    "Test_B": "/home/sagemaker-user/test_sets/exp1a_test_b_hucs.txt"
}

# Fallback if exp1a specific files don't exist
FALLBACK_TEST_SETS = {
    "Test_A": "/home/sagemaker-user/src/data/exp1a_test_hucs.txt",
    "Test_B": "/home/sagemaker-user/src/data/yakima_naches_hucs.txt"
}

# Output directory
OUTPUT_DIR = Path("/home/sagemaker-user/exp1a_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_train_val_hucs_from_files():
    """Load Exp1A train and val HUCs from split files"""

    # Try multiple locations
    locations = [
        ("/home/sagemaker-user/src/data/exp1a_train_hucs.txt",
         "/home/sagemaker-user/src/data/exp1a_validation_hucs.txt"),
        ("/home/sagemaker-user/test_sets/exp1a_train_hucs.txt",
         "/home/sagemaker-user/test_sets/exp1a_validation_hucs.txt"),
    ]

    for train_path, val_path in locations:
        if Path(train_path).exists() and Path(val_path).exists():
            print(f"  Loading from: {Path(train_path).parent}")

            with open(train_path, 'r') as f:
                train_hucs = [line.strip() for line in f if line.strip()]

            with open(val_path, 'r') as f:
                val_hucs = [line.strip() for line in f if line.strip()]

            print(f"  ✅ Loaded {len(train_hucs)} train HUCs")
            print(f"  ✅ Loaded {len(val_hucs)} val HUCs")
            return train_hucs, val_hucs

    raise FileNotFoundError("Could not find Exp1A train/val HUC split files")

def load_huc_list(file_path):
    """Load HUC IDs from text file"""
    if not Path(file_path).exists():
        print(f"  ⚠️ {file_path} not found, trying fallback...")
        return None

    with open(file_path, 'r') as f:
        hucs = [line.strip() for line in f if line.strip()]
    return hucs

def load_checkpoint(checkpoint_path):
    """Load checkpoint and extract model + params"""
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Get params from checkpoint
    params = checkpoint['params']

    print(f"  Features: {params['var_list']}")
    print(f"  Train HUCs: {len(params.get('train_hucs', []))} HUCs")
    print(f"  Val HUCs: {len(params.get('val_hucs', []))} HUCs")
    print(f"  Learning rate: {params['learning_rate']}")
    print(f"  Train dimension: {params.get('train_size_dimension', 'huc')}")

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

def evaluate_model_on_test_set(model, params, test_hucs, test_name, train_hucs, val_hucs):
    """Evaluate model on test set"""
    print(f"\n{'='*80}")
    print(f"Evaluating on: {test_name} ({len(test_hucs)} HUCs)")
    print(f"{'='*80}\n")

    # Update params with train/val HUCs
    params['train_hucs'] = train_hucs
    params['val_hucs'] = val_hucs

    # Prepare test data
    if params["train_size_dimension"] == "huc":
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
        print(f"   Normalizing using {len(train_hucs)} train + {len(val_hucs)} val HUCs")
    else:
        from snowML.LSTM import LSTM_pre_process as pp
        df_dict_test = pp.pre_process_separate(test_hucs, params["var_list"])
        print("⚠️  Using per-HUC normalization")

    print(f"✅ Loaded data for {len(df_dict_test)} HUCs\n")

    # Move model to GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    params['device'] = device
    print(f"✅ Model moved to: {device}\n")

    # Evaluate each HUC
    all_metrics = []

    for i, huc in enumerate(test_hucs, 1):
        if i % 10 == 0:
            print(f"  Progress: {i}/{len(test_hucs)} HUCs evaluated")

        if huc not in df_dict_test:
            print(f"  ⚠️  Skipping {huc} - no data")
            continue

        try:
            metric_dict_test, metric_dict_test_recur, data, y_tr_pred, \
                y_te_pred, y_tr_true, y_te_true, y_te_pred_recur, train_size = \
                evaluate.eval_from_saved_model(model, df_dict_test, huc, params)

            metric_dict_test['huc_id'] = huc
            all_metrics.append(metric_dict_test)

        except Exception as e:
            print(f"  ❌ Error evaluating {huc}: {e}")
            continue

    print(f"\n✅ Completed: {len(all_metrics)}/{len(test_hucs)} HUCs\n")

    # Convert to DataFrame
    results_df = pd.DataFrame(all_metrics)

    # Print summary
    if len(results_df) > 0:
        print(f"📊 Summary Statistics:")
        print(f"   Median KGE: {results_df['test_kge'].median():.4f}")
        print(f"   Mean KGE: {results_df['test_kge'].mean():.4f}")
        print(f"   Std KGE: {results_df['test_kge'].std():.4f}")
        print(f"   Min KGE: {results_df['test_kge'].min():.4f}")
        print(f"   Max KGE: {results_df['test_kge'].max():.4f}")

    return results_df

# =============================================================================
# MAIN EVALUATION
# =============================================================================

def main():
    """Run evaluation"""

    # Load train/val HUCs
    print("📋 Loading Exp1A train/val splits...\n")
    train_hucs, val_hucs = load_train_val_hucs_from_files()
    print()

    # Load test sets
    print("📋 Loading test sets...\n")
    test_sets = {}

    for name, file_path in TEST_SETS.items():
        hucs = load_huc_list(file_path)
        if hucs is None and name in FALLBACK_TEST_SETS:
            print(f"  Trying fallback: {FALLBACK_TEST_SETS[name]}")
            hucs = load_huc_list(FALLBACK_TEST_SETS[name])

        if hucs:
            test_sets[name] = hucs
            print(f"  {name}: {len(hucs)} HUCs")
        else:
            print(f"  ❌ Could not load {name}")

    print()

    # Check checkpoint exists
    checkpoint_path = CHECKPOINT["file"]
    if not Path(checkpoint_path).exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        print("\nAvailable Exp1A checkpoints:")
        checkpoint_dir = Path(checkpoint_path).parent
        for f in sorted(checkpoint_dir.glob("Exp1A*.pth")):
            print(f"  - {f.name}")
        return

    # Load model
    model, params = load_checkpoint(checkpoint_path)
    print()

    # Evaluate on Test A
    if "Test_A" in test_sets:
        print(f"{'='*80}")
        print(f"TEST SET A")
        print(f"{'='*80}")
        results_a = evaluate_model_on_test_set(
            model, params, test_sets["Test_A"], "Test_A", train_hucs, val_hucs
        )

        if len(results_a) > 0:
            results_a['variation'] = CHECKPOINT["name"]
            results_a['test_set'] = 'Test_A'
            output_file = OUTPUT_DIR / f"{CHECKPOINT['name']}_Test_A_metrics.csv"
            results_a.to_csv(output_file, index=False)
            print(f"\n✅ Saved: {output_file}")

    # Evaluate on Test B
    if "Test_B" in test_sets:
        print(f"\n{'='*80}")
        print(f"TEST SET B (Yakima/Naches)")
        print(f"{'='*80}")
        results_b = evaluate_model_on_test_set(
            model, params, test_sets["Test_B"], "Test_B", train_hucs, val_hucs
        )

        if len(results_b) > 0:
            results_b['variation'] = CHECKPOINT["name"]
            results_b['test_set'] = 'Test_B'
            output_file = OUTPUT_DIR / f"{CHECKPOINT['name']}_Test_B_metrics.csv"
            results_b.to_csv(output_file, index=False)
            print(f"\n✅ Saved: {output_file}")

    print(f"\n✅ EVALUATION COMPLETE!")
    print(f"\nResults saved to: {OUTPUT_DIR}/")
    print(f"\nExp1A Srad Summary:")
    print(f"  Validation KGE: {CHECKPOINT['val_kge']:.4f}")
    if "Test_A" in test_sets and len(results_a) > 0:
        print(f"  Test A KGE: {results_a['test_kge'].median():.4f}")
    if "Test_B" in test_sets and len(results_b) > 0:
        print(f"  Test B KGE: {results_b['test_kge'].median():.4f}")

if __name__ == '__main__':
    main()
