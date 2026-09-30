#!/usr/bin/env python3
import mlflow
import pandas as pd

mlflow.set_tracking_uri('arn:aws:sagemaker:us-west-2:677276086662:mlflow-tracking-server/dawgsML')

for exp_name in ['Exp1B_Wind_3e-4', 'Exp1B_Humidity_3e-4']:
    print("=" * 70)
    print(f"{exp_name}")
    print("=" * 70)

    try:
        runs = mlflow.search_runs(experiment_names=[exp_name], max_results=1)
        if len(runs) > 0:
            run = runs.iloc[0]

            # Get all metrics
            all_cols = [c for c in run.index if c.startswith('metrics.')]

            # Find epoch-by-epoch metrics
            epoch_kges = {}
            for col in all_cols:
                # Look for patterns like "val_kge_epoch_0", "epoch_0_val_kge", etc.
                if 'epoch' in col.lower() and 'kge' in col.lower() and 'val' in col.lower():
                    val = run[col]
                    if not pd.isna(val):
                        epoch_kges[col] = val

            if epoch_kges:
                print("\nPer-epoch validation KGE:")
                sorted_epochs = sorted(epoch_kges.items(), key=lambda x: x[0])
                for col, val in sorted_epochs:
                    print(f"  {col.replace('metrics.', '')}: {val:.4f}")

                # Find best epoch
                best_col, best_kge = max(epoch_kges.items(), key=lambda x: x[1])
                print(f"\n🏆 BEST EPOCH: {best_col.replace('metrics.', '')}")
                print(f"   KGE: {best_kge:.4f}")
            else:
                print("\nNo per-epoch metrics found. Showing all available metrics:")
                kge_metrics = [c for c in all_cols if 'kge' in c.lower()]
                for col in sorted(kge_metrics)[:30]:
                    val = run[col]
                    if not pd.isna(val):
                        print(f"  {col.replace('metrics.', '')}: {val:.4f}")
        else:
            print("No runs found!")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

    print()

print("=" * 70)
print("Expected: Humidity best epoch KGE ~0.82 (from original Exp3)")
print("=" * 70)
