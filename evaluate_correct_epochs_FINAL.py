#!/usr/bin/env python3
"""
Re-evaluate Exp1B with CORRECT best epochs from training_log.txt
Wind: epoch 6, Humidity: epoch 3, Base: epoch 14, Srad: epoch 6
Based on working evaluate_all_6_variations_FIXED.py
"""

import sys
import torch
import pandas as pd
from pathlib import Path
from datetime import datetime

sys.path.insert(0, '/home/sagemaker-user/src')

from snowML.LSTM import LSTM_evaluate as evaluate
from snowML.LSTM import LSTM_model as LSTM_mod

print("="*80)
print("EXP1B RE-EVALUATION - CORRECT EPOCHS")
print("="*80)
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# =============================================================================
# CONFIGURATION - CORRECT EPOCHS FROM training_log.txt
# =============================================================================

CHECKPOINTS = {
    "Wind_3e-4": {
        "file": "/home/sagemaker-user/checkpoints_best/Exp1B_Wind_3e-4_epoch6.pth",
        "val_kge": 0.8104,
        "features": ['mean_pr', 'mean_tair', 'Mean Elevation', 'mean_ws']
    },
    "Humidity_3e-4": {
        "file": "/home/sagemaker-user/checkpoints_best/Exp1B_Humidity_3e-4_epoch3.pth",
        "val_kge": 0.7971,
        "features": ['mean_pr', 'mean_tair', 'Mean Elevation', 'mean_rh']
    },
    "Base_3e-4": {
        "file": "/home/sagemaker-user/checkpoints_best/Exp1B_Base_3e-4_epoch14.pth",
        "val_kge": 0.8084,
        "features": ['mean_pr', 'mean_tair', 'Mean Elevation']
    },
    "Srad_3e-4": {
        "file": "/home/sagemaker-user/checkpoints_best/Exp1B_Srad_3e-4_epoch6.pth",
        "val_kge": 0.8018,
        "features": ['mean_pr', 'mean_tair', 'Mean Elevation', 'mean_srad']
    }
}

TEST_SETS = {
    "Test_A": "/home/sagemaker-user/test_sets/test_a_hucs.txt",
    "Test_B": "/home/sagemaker-user/test_sets/test_b_hucs.txt"
}

OUTPUT_DIR = Path("/home/sagemaker-user/exp1b_corrected_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# HELPER FUNCTIONS (from working script)
# =============================================================================

def load_train_val_hucs_from_splits():
    """Load train and val HUCs from splits"""
    train_path = "/home/sagemaker-user/test_sets/train_hucs.txt"
    val_path = "/home/sagemaker-user/test_sets/val_hucs.txt"

    # Try primary location first
    if not Path(train_path).exists():
        # Fallback to correct_test_sets
        train_path = "/home/sagemaker-user/exp1b_all_6_eval/correct_test_sets/train_hucs.txt"
        val_path = "/home/sagemaker-user/exp1b_all_6_eval/correct_test_sets/val_hucs.txt"

    with open(train_path, 'r') as f:
        train_hucs = [line.strip() for line in f if line.strip()]

    with open(val_path, 'r') as f:
        val_hucs = [line.strip() for line in f if line.strip()]

    return train_hucs, val_hucs

def load_huc_list(file_path):
    """Load HUC IDs from text file"""
    with open(file_path, 'r') as f:
        hucs = [line.strip() for line in f if line.strip()]
    return hucs

def load_checkpoint(checkpoint_path, variation_name, features):
    """Load checkpoint and extract model + params"""
    print(f"\n  Loading checkpoint: {Path(checkpoint_path).name}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Extract params
    params = checkpoint.get('params', {})

    # Set features explicitly
    params['var_list'] = features
    params['learning_rate'] = 0.0003
    params['hidden_size'] = params.get('hidden_size', 64)
    params['num_layers'] = params.get('num_layers', 1)
    params['dropout'] = params.get('dropout', 0.5)
    params['train_size_dimension'] = params.get('train_size_dimension', 'huc')
    params['lookback'] = params.get('lookback', 180)
    params['batch_size'] = params.get('batch_size', 32)
    params['recursive_predict'] = params.get('recursive_predict', False)
    params['device'] = 'cpu'

    print(f"    Features: {params['var_list']}")
    print(f"    Input size: {len(params['var_list'])}")

    # Create model with CORRECT signature
    model = LSTM_mod.SnowModel(
        input_size=len(params['var_list']),
        hidden_size=params['hidden_size'],
        num_class=1,  # Predicting SWE (single output)
        num_layers=params['num_layers'],
        dropout=params['dropout']
    )

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, params

def evaluate_model_on_test_set(model, params, test_hucs, test_name, variation_name, train_hucs, val_hucs):
    """Evaluate model on test set using original Exp3 methodology"""
    print(f"\n{'='*80}")
    print(f"Evaluating {variation_name} on {test_name} ({len(test_hucs)} HUCs)")
    print(f"{'='*80}\n")

    # Set train/val HUCs in params
    params['train_hucs'] = train_hucs
    params['val_hucs'] = val_hucs

    # Prepare test data (HUC-based normalization)
    # IMPORTANT: Some HUCs may be missing required features (e.g., mean_ws, mean_rh)
    # assemble_df_dict will print warnings and try to load all HUCs
    # Those missing features will be filtered out in df_dict_test
    print("  Loading and normalizing test data...")
    print(f"  Required features: {params['var_list']}")

    try:
        df_dict_test = evaluate.assemble_df_dict(test_hucs, params["var_list"])
        df_dict_test = evaluate.renorm(
            params["train_hucs"],
            params["val_hucs"],
            test_hucs,
            params["var_list"]
        )
    except KeyError as e:
        # Some HUCs are missing required features - filter them out
        print(f"  ⚠️ Some HUCs missing features: {e}")
        print(f"  Will evaluate only HUCs with all required features...")

        # Try loading each HUC individually and keep only those with all features
        valid_hucs = []
        for huc in test_hucs:
            try:
                # Try to load this single HUC
                test_dict = evaluate.assemble_df_dict([huc], params["var_list"])
                if huc in test_dict:
                    valid_hucs.append(huc)
            except:
                continue

        print(f"  ✅ Found {len(valid_hucs)}/{len(test_hucs)} HUCs with all features")

        if len(valid_hucs) == 0:
            print(f"  ❌ No valid HUCs for this variation!")
            return pd.DataFrame()

        # Load only valid HUCs
        df_dict_test = evaluate.assemble_df_dict(valid_hucs, params["var_list"])
        df_dict_test = evaluate.renorm(
            params["train_hucs"],
            params["val_hucs"],
            valid_hucs,
            params["var_list"]
        )
        test_hucs = valid_hucs  # Update to only valid HUCs

    print(f"  ✅ Using global normalization from {len(train_hucs)} train + {len(val_hucs)} val HUCs")
    print(f"  ✅ Loaded data for {len(df_dict_test)} HUCs\n")

    # Move model to GPU
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
            metric_dict_test, metric_dict_test_recur, data, y_tr_pred, \
                y_te_pred, y_tr_true, y_te_true, y_te_pred_recur, train_size = \
                evaluate.eval_from_saved_model(model, df_dict_test, huc, params)

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

print("[1/4] Loading HUC splits...")
train_hucs, val_hucs = load_train_val_hucs_from_splits()
print(f"✅ Train: {len(train_hucs)} HUCs")
print(f"✅ Val: {len(val_hucs)} HUCs")

print("\n[2/4] Loading test sets...")
test_a_hucs = load_huc_list(TEST_SETS["Test_A"])
test_b_hucs = load_huc_list(TEST_SETS["Test_B"])
print(f"✅ Test A: {len(test_a_hucs)} HUCs")
print(f"✅ Test B: {len(test_b_hucs)} HUCs (Yakima/Naches)")

print("\n[3/4] Evaluating models...")
print("="*80)

all_results = []
summary = []

for model_name, config in CHECKPOINTS.items():
    print(f"\n{'='*80}")
    print(f"MODEL: {model_name}")
    print(f"Validation KGE: {config['val_kge']:.4f}")
    print(f"{'='*80}")

    checkpoint_path = config["file"]
    if not Path(checkpoint_path).exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        continue

    try:
        # Load model
        model, params = load_checkpoint(checkpoint_path, model_name, config['features'])
        print(f"✅ Model loaded successfully")

        # Evaluate on Test A
        test_a_results = evaluate_model_on_test_set(
            model, params, test_a_hucs, "Test_A", model_name, train_hucs, val_hucs
        )

        if len(test_a_results) > 0:
            output_file = OUTPUT_DIR / f"{model_name}_Test_A_metrics.csv"
            test_a_results.to_csv(output_file, index=False)
            print(f"  ✅ Saved: {output_file}")
            test_a_median = test_a_results['test_kge'].median()
        else:
            test_a_median = None

        # Evaluate on Test B
        test_b_results = evaluate_model_on_test_set(
            model, params, test_b_hucs, "Test_B", model_name, train_hucs, val_hucs
        )

        if len(test_b_results) > 0:
            output_file = OUTPUT_DIR / f"{model_name}_Test_B_metrics.csv"
            test_b_results.to_csv(output_file, index=False)
            print(f"  ✅ Saved: {output_file}")
            test_b_median = test_b_results['test_kge'].median()
        else:
            test_b_median = None

        # Add to summary
        if test_a_median is not None and test_b_median is not None:
            val_drop_a = ((config['val_kge'] - test_a_median) / config['val_kge']) * 100
            val_drop_b = ((config['val_kge'] - test_b_median) / config['val_kge']) * 100

            summary.append({
                'model': model_name,
                'validation_kge': config['val_kge'],
                'test_a_kge': test_a_median,
                'test_b_kge': test_b_median,
                'val_to_test_a_drop_%': val_drop_a,
                'val_to_test_b_drop_%': val_drop_b
            })

        all_results.append(test_a_results)
        all_results.append(test_b_results)

        print(f"\n✅ {model_name} COMPLETE!")

    except Exception as e:
        print(f"\n❌ Error with {model_name}: {e}")
        import traceback
        traceback.print_exc()
        continue

print("\n[4/4] Saving summary...")
summary_df = pd.DataFrame(summary)
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
summary_file = OUTPUT_DIR / f"summary_corrected_{timestamp}.csv"
summary_df.to_csv(summary_file, index=False)
print(f"✅ Summary saved: {summary_file}")

print("\n" + "="*80)
print("FINAL RESULTS:")
print("="*80)
print(summary_df.to_string(index=False))
print("="*80)
print(f"\n✅ COMPLETE! Results saved to: {OUTPUT_DIR}/")
print("\nExpected pattern (validation >= test):")
print("  Validation: 0.80-0.81")
print("  Test A: 0.75-0.80 (5-10% drop)")
print("  Test B: 0.70-0.76 (10-15% drop)")
