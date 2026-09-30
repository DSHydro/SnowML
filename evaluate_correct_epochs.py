#!/usr/bin/env python3
"""
Re-evaluate Exp1B with CORRECT best epochs from training_log.txt
Wind: epoch 6, Humidity: epoch 3, Base: epoch 14, Srad: epoch 6
"""

import sys
import torch
import json
import pandas as pd
from pathlib import Path
from datetime import datetime

sys.path.insert(0, '/home/sagemaker-user/src')

from snowML.LSTM import LSTM_evaluate as evaluate
from snowML.LSTM import LSTM_model as LSTM_mod
from snowML.LSTM import LSTM_metrics as met

print("="*80)
print("EXP1B RE-EVALUATION - CORRECT EPOCHS")
print("="*80)

# Models with CORRECT best epochs
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

TEST_SETS = {
    "Test_A": "/home/sagemaker-user/test_sets/test_a_hucs.txt",
    "Test_B": "/home/sagemaker-user/test_sets/test_b_hucs.txt"
}

OUTPUT_DIR = Path("/home/sagemaker-user/exp1b_corrected_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_train_val_hucs():
    """Load train/val HUCs from JSON"""
    json_path = "/home/sagemaker-user/src/snowML/datapipe/huc_lists/hucs_data.json"
    print(f"  Loading HUCs from: {json_path}")
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data['train_hucs'], data['val_hucs']

def load_test_hucs(filepath):
    """Load test HUCs from text file"""
    with open(filepath, 'r') as f:
        return [line.strip() for line in f if line.strip()]

print("\n[1/4] Loading HUCs...")
train_hucs, val_hucs = load_train_val_hucs()
print(f"✅ Train: {len(train_hucs)} HUCs")
print(f"✅ Val: {len(val_hucs)} HUCs")

test_a_hucs = load_test_hucs(TEST_SETS["Test_A"])
test_b_hucs = load_test_hucs(TEST_SETS["Test_B"])
print(f"✅ Test A: {len(test_a_hucs)} HUCs")
print(f"✅ Test B: {len(test_b_hucs)} HUCs (Yakima/Naches)")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"✅ Device: {device}")

summary = []

print("\n[2/4] Evaluating models...")
for model_name, config in CHECKPOINTS.items():
    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"Validation KGE: {config['val_kge']:.4f}")
    print(f"{'='*60}")

    checkpoint_path = config["file"]
    if not Path(checkpoint_path).exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        continue

    print(f"✅ Loading checkpoint...")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        params = checkpoint.get('params', {})

        # Get model parameters
        variables = params.get('variables', ['mean_pr', 'mean_tair', 'Mean Elevation'])
        input_size = len(variables)
        hidden_size = params.get('hidden_size', 64)
        num_class = 1  # Predicting SWE (single output)
        num_layers = params.get('num_layers', 1)
        dropout = params.get('dropout', 0.5)

        # Create model with correct signature
        model = LSTM_mod.SnowModel(
            input_size=input_size,
            hidden_size=hidden_size,
            num_class=num_class,
            num_layers=num_layers,
            dropout=dropout
        ).to(device)

        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        print(f"✅ Model loaded - Features: {variables}, Input size: {input_size}")

    except Exception as e:
        print(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        continue

    # Evaluate Test A
    print(f"\n[Test A] Evaluating {len(test_a_hucs)} HUCs...")
    test_a_results = []
    for i, huc in enumerate(test_a_hucs):
        try:
            metrics = evaluate.evaluate_on_huc(
                model=model,
                huc_id=huc,
                train_hucs=train_hucs,
                val_hucs=val_hucs,
                variables=variables,
                device=device
            )
            test_a_results.append({
                'huc_id': huc,
                'test_kge': metrics['kge'],
                'test_r2': metrics['r2'],
                'test_mse': metrics['mse'],
                'test_mae': metrics['mae']
            })

            if (i + 1) % 10 == 0:
                print(f"  Progress: {i+1}/{len(test_a_hucs)} HUCs")

        except Exception as e:
            print(f"  ⚠️ Error on {huc}: {e}")
            continue

    if len(test_a_results) > 0:
        test_a_df = pd.DataFrame(test_a_results)
        test_a_median = test_a_df['test_kge'].median()
        test_a_mean = test_a_df['test_kge'].mean()
        print(f"✅ Test A - Median KGE: {test_a_median:.4f}, Mean: {test_a_mean:.4f}")

        test_a_df['variation'] = model_name
        test_a_df['test_set'] = 'Test_A'
        output_file = OUTPUT_DIR / f"{model_name}_Test_A_metrics.csv"
        test_a_df.to_csv(output_file, index=False)
        print(f"✅ Saved: {output_file}")
    else:
        test_a_median = None
        test_a_mean = None
        print(f"❌ No Test A results")

    # Evaluate Test B
    print(f"\n[Test B] Evaluating {len(test_b_hucs)} HUCs...")
    test_b_results = []
    for i, huc in enumerate(test_b_hucs):
        try:
            metrics = evaluate.evaluate_on_huc(
                model=model,
                huc_id=huc,
                train_hucs=train_hucs,
                val_hucs=val_hucs,
                variables=variables,
                device=device
            )
            test_b_results.append({
                'huc_id': huc,
                'test_kge': metrics['kge'],
                'test_r2': metrics['r2'],
                'test_mse': metrics['mse'],
                'test_mae': metrics['mae']
            })

            if (i + 1) % 10 == 0:
                print(f"  Progress: {i+1}/{len(test_b_hucs)} HUCs")

        except Exception as e:
            print(f"  ⚠️ Error on {huc}: {e}")
            continue

    if len(test_b_results) > 0:
        test_b_df = pd.DataFrame(test_b_results)
        test_b_median = test_b_df['test_kge'].median()
        test_b_mean = test_b_df['test_kge'].mean()
        print(f"✅ Test B - Median KGE: {test_b_median:.4f}, Mean: {test_b_mean:.4f}")

        test_b_df['variation'] = model_name
        test_b_df['test_set'] = 'Test_B'
        output_file = OUTPUT_DIR / f"{model_name}_Test_B_metrics.csv"
        test_b_df.to_csv(output_file, index=False)
        print(f"✅ Saved: {output_file}")
    else:
        test_b_median = None
        test_b_mean = None
        print(f"❌ No Test B results")

    # Summary
    if test_a_median is not None and test_b_median is not None:
        val_drop_a = ((config['val_kge'] - test_a_median) / config['val_kge']) * 100
        val_drop_b = ((config['val_kge'] - test_b_median) / config['val_kge']) * 100

        summary.append({
            'model': model_name,
            'validation_kge': config['val_kge'],
            'test_a_median': test_a_median,
            'test_a_mean': test_a_mean,
            'test_b_median': test_b_median,
            'test_b_mean': test_b_mean,
            'val_to_test_a_drop_%': val_drop_a,
            'val_to_test_b_drop_%': val_drop_b
        })

    print(f"\n✅ {model_name} complete!")

print("\n[3/4] Saving summary...")
summary_df = pd.DataFrame(summary)
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
summary_file = OUTPUT_DIR / f"summary_corrected_{timestamp}.csv"
summary_df.to_csv(summary_file, index=False)
print(f"✅ Summary saved: {summary_file}")

print("\n[4/4] RESULTS:")
print("="*80)
print(summary_df.to_string(index=False))
print("="*80)
print(f"\n✅ Complete! Results saved to: {OUTPUT_DIR}/")
print("\nExpected pattern (validation >= test):")
print("  Validation: 0.80-0.81")
print("  Test A: 0.75-0.80 (5-10% drop)")
print("  Test B: 0.70-0.76 (10-15% drop)")
