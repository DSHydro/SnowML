#!/usr/bin/env python3
import mlflow
import pandas as pd

mlflow.set_tracking_uri('arn:aws:sagemaker:us-west-2:677276086662:mlflow-tracking-server/dawgsML')

# Check Wind_3e-4
print("=" * 60)
print("Wind_3e-4 Results:")
print("=" * 60)
try:
    runs = mlflow.search_runs(experiment_names=['Exp1B_Wind_3e-4'], max_results=1)
    if len(runs) > 0:
        kge_cols = [c for c in runs.columns if 'kge' in c.lower()]
        for col in sorted(kge_cols)[:10]:
            val = runs[col].iloc[0]
            if not pd.isna(val):
                print(f'{col.replace("metrics.", "")}: {val:.4f}')
    else:
        print("No runs found!")
except Exception as e:
    print(f"Error: {e}")

# Check Humidity_3e-4
print("\n" + "=" * 60)
print("Humidity_3e-4 Results:")
print("=" * 60)
try:
    runs = mlflow.search_runs(experiment_names=['Exp1B_Humidity_3e-4'], max_results=1)
    if len(runs) > 0:
        kge_cols = [c for c in runs.columns if 'kge' in c.lower()]
        for col in sorted(kge_cols)[:10]:
            val = runs[col].iloc[0]
            if not pd.isna(val):
                print(f'{col.replace("metrics.", "")}: {val:.4f}')
    else:
        print("No runs found!")
except Exception as e:
    print(f"Error: {e}")

print("\n" + "=" * 60)
print("Expected from original Exp3: Humidity KGE ~0.82")
print("=" * 60)
