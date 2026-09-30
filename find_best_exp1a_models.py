#!/usr/bin/env python3
"""
Find best Exp1A models from MLflow
Run this on SageMaker after activating pytorch_p310
"""

import mlflow
import pandas as pd

# Set MLflow tracking URI
mlflow.set_tracking_uri("arn:aws:sagemaker:us-west-2:677276086662:mlflow-tracking-server/dawgsML")

# Exp1A variations to check
variations = [
    "Exp1A_Base_3e-4",
    "Exp1A_Srad_3e-4",
    "Exp1A_Wind_3e-4",
    "Exp1A_Humidity_3e-4"
]

print("=" * 80)
print("EXP1A BEST MODELS SUMMARY")
print("=" * 80)
print()

results = []

for exp_name in variations:
    try:
        # Get experiment
        experiment = mlflow.get_experiment_by_name(exp_name)
        if experiment is None:
            print(f"❌ {exp_name}: NOT FOUND in MLflow")
            continue

        # Search runs
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["metrics.median_val_kge DESC"],
            max_results=1
        )

        if len(runs) == 0:
            print(f"❌ {exp_name}: No runs found")
            continue

        # Get best run
        best_run = runs.iloc[0]

        # Extract metrics
        median_kge = best_run.get("metrics.median_val_kge", None)
        mean_kge = best_run.get("metrics.mean_val_kge", None)
        epoch = best_run.get("params.epoch", None)

        # Try to get epoch from tags if not in params
        if epoch is None:
            epoch = best_run.get("tags.epoch", None)

        print(f"✅ {exp_name}")
        print(f"   Best Epoch: {epoch}")
        print(f"   Median Val KGE: {median_kge:.4f}" if median_kge else "   Median Val KGE: N/A")
        print(f"   Mean Val KGE: {mean_kge:.4f}" if mean_kge else "   Mean Val KGE: N/A")
        print()

        results.append({
            'Variation': exp_name.replace('Exp1A_', ''),
            'Best_Epoch': epoch,
            'Median_Val_KGE': median_kge,
            'Mean_Val_KGE': mean_kge,
            'Run_ID': best_run['run_id']
        })

    except Exception as e:
        print(f"❌ {exp_name}: Error - {str(e)}")
        print()

print("=" * 80)
print("COMPARISON TABLE")
print("=" * 80)

if results:
    df = pd.DataFrame(results)
    df = df.sort_values('Median_Val_KGE', ascending=False)
    print(df.to_string(index=False))
    print()

    # Find overall best
    best = df.iloc[0]
    print("=" * 80)
    print("🏆 BEST MODEL:")
    print(f"   Variation: {best['Variation']}")
    print(f"   Epoch: {best['Best_Epoch']}")
    print(f"   Median Val KGE: {best['Median_Val_KGE']:.4f}")
    print("=" * 80)
    print()
    print(f"📥 Download this checkpoint:")
    print(f"   scp checkpoints/Exp1A_{best['Variation']}_epoch{best['Best_Epoch']}.pth <local-path>")
else:
    print("No results found!")

print()
