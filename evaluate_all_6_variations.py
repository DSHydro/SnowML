#!/usr/bin/env python3
"""
Evaluate ALL 6 Variations on Test Sets A & B
Uses the EXACT same evaluation code as original Exp3 students
Based on evaluate_exp1b_final.py but extended for all 6 variations

VARIATIONS TO EVALUATE:
======================

1. Base_1e-3
   Features: mean_tair (temp) + mean_pr (precip) + Mean Elevation
   Learning Rate: 0.001 (HIGH)
   Expected: Poor performance - LR too high, unstable training

2. Base_3e-4
   Features: mean_tair (temp) + mean_pr (precip) + Mean Elevation
   Learning Rate: 0.0003 (OPTIMAL)
   Expected: Good baseline performance

3. Srad_1e-3
   Features: Base + mean_srad (solar radiation)
   Learning Rate: 0.001 (HIGH)
   Expected: Unstable - LR too high despite extra feature

4. Srad_3e-4
   Features: Base + mean_srad (solar radiation)
   Learning Rate: 0.0003 (OPTIMAL)
   Expected: Good - solar radiation should help
   Note: Only 28 epochs (training stopped at epoch 27)

5. Wind_3e-4 ✅
   Features: Base + mean_vs (wind speed)
   Learning Rate: 0.0003 (OPTIMAL)
   Already Validated: Test A KGE=0.8375, Test B KGE=0.7647
   Expected: Excellent (re-running for consistency)

6. Humidity_3e-4 ✅
   Features: Base + mean_rh (relative humidity)
   Learning Rate: 0.0003 (OPTIMAL)
   Already Validated: Test A KGE=0.8119, Test B KGE=0.7899
   Expected: Excellent (re-running for consistency)

TEST SETS:
==========
- Test A: 55 HUCs (randomly held-out from deep snow HUCs)
- Test B: 81 HUCs (Yakima/Naches region - completely unseen)
- Normalization: Using 162 train + 54 val HUCs global statistics
- Method: HUC-based spatial cross-validation (corrected Aug 2026)
"""

import sys
import torch
import json
import pandas as pd
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from snowML.LSTM import LSTM_evaluate as evaluate
from snowML.LSTM import LSTM_model as LSTM_mod
from snowML.LSTM import LSTM_metrics as met

print("="*80)
print("EXPERIMENT 1B - COMPLETE EVALUATION")
print("Evaluating ALL 6 Variations on Test Sets A & B")
print("Using Original Exp3 Evaluation Methodology")
print("="*80)
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# =============================================================================
# CONFIGURATION
# =============================================================================

# Base directory on AWS SageMaker
BASE_DIR = "/home/sagemaker-user/exp1b_all_6_eval"

# All 6 variations with their best epochs
# Will find best epochs automatically from checkpoints
CHECKPOINTS = {
    "Base_1e-3": {
        "pattern": f"{BASE_DIR}/Exp1B_Base_1e-3_epoch{{}}.pth",
        "best_epoch": None,  # Will find from validation
        "max_epoch": 29,
        "expected_performance": "Poor (LR too high)"
    },
    "Base_3e-4": {
        "pattern": f"{BASE_DIR}/Exp1B_Base_3e-4_epoch{{}}.pth",
        "best_epoch": None,  # Will find from validation
        "max_epoch": 29,
        "expected_performance": "Good baseline"
    },
    "Srad_1e-3": {
        "pattern": f"{BASE_DIR}/Exp1B_Srad_1e-3_epoch{{}}.pth",
        "best_epoch": None,  # Will find from validation
        "max_epoch": 29,
        "expected_performance": "Poor (LR too high)"
    },
    "Srad_3e-4": {
        "pattern": f"{BASE_DIR}/Exp1B_Srad_3e-4_epoch{{}}.pth",
        "best_epoch": None,  # Will find from validation
        "max_epoch": 27,  # Stopped early
        "expected_performance": "Good with solar radiation"
    },
    "Wind_3e-4": {
        "pattern": f"{BASE_DIR}/Exp1B_Wind_3e-4_epoch{{}}.pth",
        "best_epoch": 6,  # Already known
        "max_epoch": 29,
        "expected_performance": "Excellent (already validated)"
    },
    "Humidity_3e-4": {
        "pattern": f"{BASE_DIR}/Exp1B_Humidity_3e-4_epoch{{}}.pth",
        "best_epoch": 7,  # Already known
        "max_epoch": 29,
        "expected_performance": "Excellent (already validated)"
    }
}

# Test set HUC lists (CORRECTED - from correct_test_sets folder!)
TEST_SETS = {
    "Test_A": f"{BASE_DIR}/correct_test_sets/test_a_hucs.txt",  # 55 HUCs
    "Test_B": f"{BASE_DIR}/correct_test_sets/test_b_hucs.txt"   # 81 HUCs
}

# Output directory
OUTPUT_DIR = Path(f"{BASE_DIR}/results/all_6_variations_evaluation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_train_val_hucs_from_splits():
    """
    Load train and val HUCs from the CORRECTED splits in correct_test_sets folder.
    These are the 162/54 splits used in the corrected training (Aug 2026).
    """
    # Use the CORRECT splits from correct_test_sets folder on AWS
    base_dir = "/home/sagemaker-user/exp1b_all_6_eval"
    train_path = f"{base_dir}/correct_test_sets/train_hucs.txt"
    val_path = f"{base_dir}/correct_test_sets/val_hucs.txt"

    print(f"  Loading CORRECTED splits from {base_dir}/correct_test_sets/")

    # Load train HUCs
    with open(train_path, 'r') as f:
        train_hucs = [line.strip() for line in f if line.strip()]

    # Load val HUCs
    with open(val_path, 'r') as f:
        val_hucs = [line.strip() for line in f if line.strip()]

    print(f"  ✅ Loaded {len(train_hucs)} train HUCs")
    print(f"  ✅ Loaded {len(val_hucs)} val HUCs")

    return train_hucs, val_hucs

def load_huc_list(file_path):
    """Load HUC IDs from text file"""
    with open(file_path, 'r') as f:
        hucs = [line.strip() for line in f if line.strip()]
    return hucs

def find_best_epoch_from_checkpoints(variation_name, pattern, max_epoch):
    """
    Find best epoch by loading validation KGE from each checkpoint
    """
    print(f"\n  Finding best epoch for {variation_name}...")

    best_epoch = 0
    best_kge = -999.0

    for epoch in range(max_epoch + 1):
        checkpoint_path = pattern.format(epoch)

        if not Path(checkpoint_path).exists():
            continue

        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')

            # Try different possible locations for validation KGE
            val_kge = None
            if 'metrics' in checkpoint and 'val_kge' in checkpoint['metrics']:
                val_kge = checkpoint['metrics']['val_kge']
            elif 'metrics' in checkpoint and 'final_val_kge' in checkpoint['metrics']:
                val_kge = checkpoint['metrics']['final_val_kge']
            elif 'val_kge' in checkpoint:
                val_kge = checkpoint['val_kge']

            if val_kge is not None and val_kge > best_kge:
                best_kge = val_kge
                best_epoch = epoch

        except Exception as e:
            print(f"    Warning: Could not load epoch {epoch}: {e}")
            continue

    print(f"  ✅ Best epoch: {best_epoch} (Val KGE: {best_kge:.4f})")
    return best_epoch, best_kge

def load_checkpoint(checkpoint_path):
    """
    Load checkpoint and extract model + params
    Returns model and params in format compatible with original evaluate.py
    """
    print(f"\n  Loading checkpoint: {Path(checkpoint_path).name}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Get params from checkpoint
    params = checkpoint['params']

    print(f"    Features: {params['var_list']}")
    print(f"    Learning rate: {params['learning_rate']}")
    print(f"    Train dimension: {params.get('train_size_dimension', 'time')}")

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

def evaluate_model_on_test_set(model, params, test_hucs, test_name, variation_name):
    """
    Evaluate model on test set using ORIGINAL Exp3 methodology
    """
    print(f"\n{'='*80}")
    print(f"Evaluating {variation_name} on {test_name} ({len(test_hucs)} HUCs)")
    print(f"{'='*80}\n")

    # Load train/val HUCs if not in checkpoint
    if 'train_hucs' not in params or 'val_hucs' not in params or \
       len(params.get('train_hucs', [])) == 0 or len(params.get('val_hucs', [])) == 0:
        train_hucs, val_hucs = load_train_val_hucs_from_splits()
        params['train_hucs'] = train_hucs
        params['val_hucs'] = val_hucs

    # Prepare test data using ORIGINAL Exp3 method
    if params["train_size_dimension"] == "huc":
        print("  Loading and normalizing test data (HUC-based)...")
        df_dict_test = evaluate.assemble_df_dict(test_hucs, params["var_list"])
        df_dict_test = evaluate.renorm(
            params["train_hucs"],
            params["val_hucs"],
            test_hucs,
            params["var_list"]
        )
        print(f"  ✅ Using global normalization from {len(params['train_hucs'])} train + {len(params['val_hucs'])} val HUCs")
    else:
        # Per-HUC normalization
        from snowML.LSTM import LSTM_pre_process as pp
        df_dict_test = pp.pre_process_separate(test_hucs, params["var_list"])
        print("  ⚠️  Using per-HUC normalization")

    print(f"  ✅ Loaded data for {len(df_dict_test)} HUCs\n")

    # Move model to GPU if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    params['device'] = device
    print(f"  Device: {device}\n")

    # Evaluate each HUC
    all_metrics = []

    for i, huc in enumerate(test_hucs, 1):
        if i % 10 == 0:
            print(f"    Progress: {i}/{len(test_hucs)} HUCs")

        if huc not in df_dict_test:
            print(f"    ⚠️  Skipping {huc} - no data")
            continue

        try:
            # Use ORIGINAL evaluation function
            metric_dict_test, metric_dict_test_recur, data, y_tr_pred, \
                y_te_pred, y_tr_true, y_te_true, y_te_pred_recur, train_size = \
                evaluate.eval_from_saved_model(model, df_dict_test, huc, params)

            # Add identifiers
            metric_dict_test['huc_id'] = huc
            metric_dict_test['variation'] = variation_name
            metric_dict_test['test_set'] = test_name
            all_metrics.append(metric_dict_test)

        except Exception as e:
            print(f"    ❌ Error evaluating {huc}: {e}")
            continue

    print(f"\n  ✅ Completed: {len(all_metrics)}/{len(test_hucs)} HUCs\n")

    # Convert to DataFrame
    results_df = pd.DataFrame(all_metrics)

    # Print summary
    if len(results_df) > 0:
        print(f"  📊 Summary Statistics:")
        print(f"     Median KGE: {results_df['test_kge'].median():.4f}")
        print(f"     Mean KGE:   {results_df['test_kge'].mean():.4f}")
        print(f"     Std KGE:    {results_df['test_kge'].std():.4f}")
        print(f"     Range:      {results_df['test_kge'].min():.4f} to {results_df['test_kge'].max():.4f}")

    return results_df

# =============================================================================
# MAIN EVALUATION
# =============================================================================

def main():
    """Run full evaluation on all 6 variations"""

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Load test sets
    print("📋 Loading test sets...\n")
    test_sets = {}
    for name, file_path in TEST_SETS.items():
        hucs = load_huc_list(file_path)
        test_sets[name] = hucs
        print(f"  {name}: {len(hucs)} HUCs")

    print()

    # Find best epochs for variations that don't have them
    print("\n" + "="*80)
    print("STEP 1: Finding Best Epochs for Each Variation")
    print("="*80)

    for var_name, var_info in CHECKPOINTS.items():
        if var_info['best_epoch'] is None:
            best_epoch, best_kge = find_best_epoch_from_checkpoints(
                var_name,
                var_info['pattern'],
                var_info['max_epoch']
            )
            var_info['best_epoch'] = best_epoch
            var_info['best_val_kge'] = best_kge

    # Print summary of what will be evaluated
    print("\n" + "="*80)
    print("STEP 2: Evaluation Plan")
    print("="*80)

    for var_name, var_info in CHECKPOINTS.items():
        best_ep = var_info['best_epoch']
        max_ep = var_info['max_epoch']
        expected = var_info['expected_performance']
        best_kge = var_info.get('best_val_kge', 'N/A')
        print(f"  {var_name:20s} | Epoch: {best_ep:2d}/{max_ep} | Val KGE: {best_kge if isinstance(best_kge, str) else f'{best_kge:.4f}':<8s} | {expected}")

    # Evaluate each variation on each test set
    print("\n" + "="*80)
    print("STEP 3: Running Evaluations")
    print("="*80)

    all_results = []

    for var_name, var_info in CHECKPOINTS.items():
        print(f"\n\n{'#'*80}")
        print(f"# VARIATION: {var_name}")
        print(f"{'#'*80}")

        # Load checkpoint
        checkpoint_path = var_info['pattern'].format(var_info['best_epoch'])

        try:
            model, params = load_checkpoint(checkpoint_path)

            # Evaluate on each test set
            for test_name, test_hucs in test_sets.items():
                results_df = evaluate_model_on_test_set(
                    model, params, test_hucs, test_name, var_name
                )

                if len(results_df) > 0:
                    all_results.append(results_df)

                    # Save individual results
                    output_file = OUTPUT_DIR / f"{var_name}_{test_name}_metrics.csv"
                    results_df.to_csv(output_file, index=False)
                    print(f"  💾 Saved: {output_file}")

        except Exception as e:
            print(f"  ❌ Failed to evaluate {var_name}: {e}")
            continue

    # Combine all results
    if all_results:
        print("\n" + "="*80)
        print("STEP 4: Generating Summary")
        print("="*80)

        all_results_df = pd.concat(all_results, ignore_index=True)

        # Save complete results
        complete_file = OUTPUT_DIR / f"all_6_variations_complete_{timestamp}.csv"
        all_results_df.to_csv(complete_file, index=False)
        print(f"\n💾 Complete results saved: {complete_file}")

        # Generate summary by variation and test set
        summary = all_results_df.groupby(['variation', 'test_set'])['test_kge'].agg([
            ('median', 'median'),
            ('mean', 'mean'),
            ('std', 'std'),
            ('min', 'min'),
            ('max', 'max'),
            ('count', 'count')
        ]).reset_index()

        summary_file = OUTPUT_DIR / f"summary_all_variations_{timestamp}.csv"
        summary.to_csv(summary_file, index=False)
        print(f"💾 Summary saved: {summary_file}")

        # Print final summary table
        print("\n" + "="*80)
        print("FINAL SUMMARY: All 6 Variations")
        print("="*80)
        print("\n" + summary.to_string(index=False))

    print("\n" + "="*80)
    print("✅ EVALUATION COMPLETE!")
    print("="*80)
    print(f"Results saved in: {OUTPUT_DIR}")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
