#!/usr/bin/env python3
"""
Re-evaluate Exp1B models using CORRECT BEST EPOCHS from training_log.txt

Based on training_log.txt analysis:
- Wind_3e-4: Best epoch 6, validation KGE = 0.8104
- Humidity_3e-4: Best epoch 3, validation KGE = 0.7971 (NOT epoch 7!)
- Base_3e-4: Best epoch 14, validation KGE = 0.8084
- Srad_3e-4: Best epoch 6, validation KGE = 0.8018

This will give us clean, trustworthy results with validation ≈ test.
"""

import sys
import os
import torch
import pandas as pd
from datetime import datetime

# Add src to path
sys.path.insert(0, '/home/sagemaker-user/src')

from snowML.LSTM import LSTM_model
from snowML.LSTM.set_hyperparams import create_hyper_dict
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("EXP1B RE-EVALUATION - CORRECT EPOCHS FROM training_log.txt")
print("="*80)

# Configuration
CHECKPOINT_DIR = "/home/sagemaker-user/checkpoints"
DATA_DIR = "/home/sagemaker-user/src/data"
OUTPUT_DIR = "/home/sagemaker-user/exp1b_results_corrected"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load train/val HUCs for normalization
def load_train_val_hucs():
    """Load train and validation HUCs from split files"""
    train_file = os.path.join(DATA_DIR, 'exp1b_train_hucs.txt')
    val_file = os.path.join(DATA_DIR, 'exp1b_validation_hucs.txt')

    with open(train_file, 'r') as f:
        train_hucs = [line.strip() for line in f if line.strip()]

    with open(val_file, 'r') as f:
        val_hucs = [line.strip() for line in f if line.strip()]

    print(f"✅ Loaded {len(train_hucs)} train HUCs and {len(val_hucs)} val HUCs")
    return train_hucs, val_hucs

# Load test sets
def load_test_sets():
    """Load Test A and Test B HUCs"""
    test_a_file = os.path.join(DATA_DIR, 'exp1b_test_hucs.txt')
    test_b_file = os.path.join(DATA_DIR, 'yakima_naches_hucs.txt')

    with open(test_a_file, 'r') as f:
        test_a_hucs = [line.strip() for line in f if line.strip()]

    with open(test_b_file, 'r') as f:
        test_b_hucs = [line.strip() for line in f if line.strip()]

    print(f"✅ Test Set A: {len(test_a_hucs)} HUCs")
    print(f"✅ Test Set B: {len(test_b_hucs)} HUCs (Yakima/Naches)")
    return test_a_hucs, test_b_hucs

# Models to evaluate with CORRECT epochs
MODELS = [
    {
        'name': 'Exp1B_Wind_3e-4_epoch6',
        'checkpoint': 'Exp1B_Wind_3e-4_epoch6.pth',
        'features': ['mean_pr', 'mean_tair', 'Mean Elevation', 'mean_ws'],
        'validation_kge': 0.8104,
        'lr': 0.0003,
    },
    {
        'name': 'Exp1B_Humidity_3e-4_epoch3',
        'checkpoint': 'Exp1B_Humidity_3e-4_epoch3.pth',
        'features': ['mean_pr', 'mean_tair', 'Mean Elevation', 'mean_rh'],
        'validation_kge': 0.7971,
        'lr': 0.0003,
    },
    {
        'name': 'Exp1B_Base_3e-4_epoch14',
        'checkpoint': 'Exp1B_Base_3e-4_epoch14.pth',
        'features': ['mean_pr', 'mean_tair', 'Mean Elevation'],
        'validation_kge': 0.8084,
        'lr': 0.0003,
    },
    {
        'name': 'Exp1B_Srad_3e-4_epoch6',
        'checkpoint': 'Exp1B_Srad_3e-4_epoch6.pth',
        'features': ['mean_pr', 'mean_tair', 'Mean Elevation', 'mean_srad'],
        'validation_kge': 0.8018,
        'lr': 0.0003,
    },
]

def evaluate_model_on_hucs(model, test_hucs, train_hucs, val_hucs, features, device):
    """Evaluate model on a set of HUCs"""
    from snowML.LSTM.LSTM_train import evaluate_on_huc

    results = []

    for i, huc in enumerate(test_hucs):
        try:
            metrics = evaluate_on_huc(
                model=model,
                huc_id=huc,
                train_hucs=train_hucs,
                val_hucs=val_hucs,
                variables=features,
                device=device
            )

            results.append({
                'huc_id': huc,
                'test_mse': metrics['mse'],
                'test_kge': metrics['kge'],
                'test_r2': metrics['r2'],
                'test_mae': metrics['mae']
            })

            if (i + 1) % 10 == 0:
                print(f"  Progress: {i+1}/{len(test_hucs)} HUCs evaluated")

        except Exception as e:
            print(f"  ⚠️ Error evaluating HUC {huc}: {e}")
            continue

    return pd.DataFrame(results)

def main():
    print("\n[1/5] Loading train/val HUCs for normalization...")
    train_hucs, val_hucs = load_train_val_hucs()

    print("\n[2/5] Loading test sets...")
    test_a_hucs, test_b_hucs = load_test_sets()

    print("\n[3/5] Setting up device...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✅ Using device: {device}")

    # Summary results
    summary = []

    print("\n[4/5] Evaluating models...")
    print("="*80)

    for model_info in MODELS:
        print(f"\n{'='*80}")
        print(f"MODEL: {model_info['name']}")
        print(f"Validation KGE: {model_info['validation_kge']:.4f}")
        print(f"Features: {', '.join(model_info['features'])}")
        print(f"{'='*80}")

        checkpoint_path = os.path.join(CHECKPOINT_DIR, model_info['checkpoint'])

        # Check if checkpoint exists
        if not os.path.exists(checkpoint_path):
            print(f"⚠️ Checkpoint not found: {checkpoint_path}")
            print(f"Available checkpoints:")
            for f in os.listdir(CHECKPOINT_DIR):
                if 'Exp1B' in f and '.pth' in f:
                    print(f"  - {f}")
            continue

        print(f"✅ Loading checkpoint: {model_info['checkpoint']}")

        try:
            # Create model
            hyperparams = create_hyper_dict()
            hyperparams['learning_rate'] = model_info['lr']

            model = LSTM_model.SnowModel(
                n_input=len(model_info['features']),
                hidden_size=hyperparams['hidden_size'],
                n_layers=hyperparams['n_layers'],
                dropout=hyperparams['dropout'],
                learning_rate=hyperparams['learning_rate']
            ).to(device)

            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()

            print(f"✅ Model loaded successfully")

            # Evaluate on Test A
            print(f"\n[Test Set A] Evaluating on {len(test_a_hucs)} HUCs...")
            test_a_results = evaluate_model_on_hucs(
                model, test_a_hucs, train_hucs, val_hucs,
                model_info['features'], device
            )

            if len(test_a_results) > 0:
                test_a_kge_median = test_a_results['test_kge'].median()
                test_a_kge_mean = test_a_results['test_kge'].mean()
                print(f"✅ Test A - Median KGE: {test_a_kge_median:.4f}, Mean: {test_a_kge_mean:.4f}")

                # Save Test A results
                output_file = os.path.join(OUTPUT_DIR, f"{model_info['name']}_Test_A_metrics.csv")
                test_a_results['variation'] = model_info['name']
                test_a_results['test_set'] = 'Test_A'
                test_a_results.to_csv(output_file, index=False)
                print(f"✅ Saved: {output_file}")
            else:
                test_a_kge_median = None
                test_a_kge_mean = None
                print(f"⚠️ No Test A results")

            # Evaluate on Test B
            print(f"\n[Test Set B] Evaluating on {len(test_b_hucs)} HUCs (Yakima/Naches)...")
            test_b_results = evaluate_model_on_hucs(
                model, test_b_hucs, train_hucs, val_hucs,
                model_info['features'], device
            )

            if len(test_b_results) > 0:
                test_b_kge_median = test_b_results['test_kge'].median()
                test_b_kge_mean = test_b_results['test_kge'].mean()
                print(f"✅ Test B - Median KGE: {test_b_kge_median:.4f}, Mean: {test_b_kge_mean:.4f}")

                # Save Test B results
                output_file = os.path.join(OUTPUT_DIR, f"{model_info['name']}_Test_B_metrics.csv")
                test_b_results['variation'] = model_info['name']
                test_b_results['test_set'] = 'Test_B'
                test_b_results.to_csv(output_file, index=False)
                print(f"✅ Saved: {output_file}")
            else:
                test_b_kge_median = None
                test_b_kge_mean = None
                print(f"⚠️ No Test B results")

            # Add to summary
            summary.append({
                'model': model_info['name'],
                'validation_kge': model_info['validation_kge'],
                'test_a_kge_median': test_a_kge_median,
                'test_a_kge_mean': test_a_kge_mean,
                'test_b_kge_median': test_b_kge_median,
                'test_b_kge_mean': test_b_kge_mean,
                'val_to_test_a_drop_pct': ((model_info['validation_kge'] - test_a_kge_median) / model_info['validation_kge'] * 100) if test_a_kge_median else None,
                'val_to_test_b_drop_pct': ((model_info['validation_kge'] - test_b_kge_median) / model_info['validation_kge'] * 100) if test_b_kge_median else None,
            })

            print(f"\n✅ {model_info['name']} evaluation complete!")

        except Exception as e:
            print(f"❌ Error evaluating {model_info['name']}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Save summary
    print("\n[5/5] Saving summary...")
    summary_df = pd.DataFrame(summary)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    summary_file = os.path.join(OUTPUT_DIR, f'summary_corrected_epochs_{timestamp}.csv')
    summary_df.to_csv(summary_file, index=False)
    print(f"✅ Summary saved: {summary_file}")

    # Display summary
    print("\n" + "="*80)
    print("SUMMARY: CORRECTED EPOCH EVALUATION")
    print("="*80)
    print(summary_df.to_string(index=False))

    print("\n" + "="*80)
    print("✅ RE-EVALUATION COMPLETE!")
    print("="*80)
    print(f"\nResults saved to: {OUTPUT_DIR}/")
    print("\nExpected pattern (validation ≈ test):")
    print("  Validation: 0.80-0.81")
    print("  Test A: 0.75-0.80 (5-10% drop)")
    print("  Test B: 0.70-0.76 (10-15% drop)")

if __name__ == '__main__':
    main()
