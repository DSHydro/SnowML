#!/usr/bin/env python3
import mlflow
import pandas as pd

mlflow.set_tracking_uri('arn:aws:sagemaker:us-west-2:677276086662:mlflow-tracking-server/dawgsML')

for exp_name in ['Exp1B_Humidity_3e-4']:
    print("=" * 70)
    print(f"{exp_name} - ALL METRICS")
    print("=" * 70)

    try:
        runs = mlflow.search_runs(experiment_names=[exp_name], max_results=1)
        if len(runs) > 0:
            run = runs.iloc[0]

            # Get ALL metric columns
            metric_cols = [c for c in run.index if c.startswith('metrics.')]

            print(f"\nTotal metrics logged: {len(metric_cols)}")
            print("\nAll metric names (first 100):")
            for i, col in enumerate(sorted(metric_cols)[:100]):
                val = run[col]
                if not pd.isna(val):
                    print(f"{i+1}. {col.replace('metrics.', '')}: {val:.4f}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "=" * 70)
print("Looking for metrics with 'val', 'median', 'mean', or 'epoch' in name")
print("=" * 70)
