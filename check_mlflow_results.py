#!/usr/bin/env python3
"""
Check MLflow results for completed Exp1B variations
"""

import mlflow
import pandas as pd

# Set tracking URI
mlflow.set_tracking_uri("arn:aws:sagemaker:us-west-2:677276086662:mlflow-tracking-server/dawgsML")

print("=" * 80)
print("CHECKING EXP1B RESULTS FROM MLFLOW")
print("=" * 80)
print()

# Search for Exp1B experiments
print("Searching for Exp1B experiments...")
try:
    experiments = mlflow.search_experiments()
    exp1b_experiments = [exp for exp in experiments if "Exp1B" in exp.name]

    print(f"\nFound {len(exp1b_experiments)} Exp1B experiments:")
    for exp in exp1b_experiments:
        print(f"  - {exp.name} (ID: {exp.experiment_id})")
    print()
except Exception as e:
    print(f"Error searching experiments: {e}")
    print()

# Check each completed variation
completed_variations = ["Exp1B_Base_1e-3", "Exp1B_Base_3e-4", "Exp1B_Srad_1e-3"]

print("=" * 80)
print("RESULTS FOR COMPLETED VARIATIONS")
print("=" * 80)
print()

for var_name in completed_variations:
    print(f"--- {var_name} ---")
    try:
        runs = mlflow.search_runs(
            experiment_names=[var_name],
            order_by=["start_time DESC"],
            max_results=1
        )

        if len(runs) > 0:
            run = runs.iloc[0]
            print(f"Status: {run['status']}")
            print(f"Start time: {run['start_time']}")
            print(f"End time: {run.get('end_time', 'N/A')}")

            # Get all metrics columns
            metric_cols = [col for col in run.index if col.startswith('metrics.')]

            # Look for KGE metrics specifically
            kge_metrics = [col for col in metric_cols if 'kge' in col.lower()]

            if kge_metrics:
                print(f"\nKGE Metrics:")
                for m in sorted(kge_metrics)[:10]:  # Show first 10
                    value = run[m]
                    if pd.notna(value):
                        print(f"  {m.replace('metrics.', '')}: {value:.4f}")
            else:
                print("\nNo KGE metrics found!")

            # Show other available metrics
            other_metrics = [col for col in metric_cols if 'kge' not in col.lower()][:5]
            if other_metrics:
                print(f"\nOther metrics:")
                for m in other_metrics:
                    value = run[m]
                    if pd.notna(value):
                        print(f"  {m.replace('metrics.', '')}: {value:.4f}")

        else:
            print(f"No runs found for {var_name}")

    except Exception as e:
        print(f"Error: {e}")

    print()

print("=" * 80)
print("EXPECTED RESULTS (from original Exp3):")
print("  - Validation KGE: 0.82-0.85")
print("  - If your results are similar, variations 4-8 are worth running")
print("  - If results are poor, we need to fix the issue first")
print("=" * 80)
