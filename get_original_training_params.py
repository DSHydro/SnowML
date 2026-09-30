#!/usr/bin/env python3
"""
Get the EXACT training parameters from the previous students' MLflow runs.
This will show us exactly what they did so we can reproduce it.
"""

import mlflow
import pandas as pd

# AWS MLflow server
tracking_uri = "arn:aws:sagemaker:us-west-2:677276086662:mlflow-tracking-server/dawgsML"

# The 8 runs from previous students (from download_metrics.py)
run_dict = {
    "base_1e-3": "a6c611d4c4cf410e9666796e3a8892b7",
    "hum_1e-3": "d71b47a8db534a059578162b9a8808b7",
    "srad_1e-3": "deed782fda71472fb47cf8670b668473",
    "vs_1e-3": "4653005687094d9ba54c295b943a4667",
    "base_3e-4": "e989030c272d4de59c84aff739d8063c",
    "hum_3e-4": "51884b406ec545ec96763d9eefd38c36",
    "srad_3e-4": "2b49d6cce3844ede8a66821ae9aec27b",
    "vs_3e-4": "bc031cafad7445adb73173adc43b63c6"
}

print("=" * 80)
print("QUERYING AWS MLFLOW SERVER FOR ORIGINAL TRAINING PARAMETERS")
print("=" * 80)
print(f"Server: {tracking_uri}")
print()

mlflow.set_tracking_uri(tracking_uri)
client = mlflow.MlflowClient()

# Get parameters from the first run (they should all be similar except vars and lr)
first_run_id = run_dict["base_1e-3"]
print(f"Checking run: base_1e-3 ({first_run_id})")
print()

try:
    run = client.get_run(first_run_id)
    params = run.data.params

    print("TRAINING PARAMETERS:")
    print("-" * 80)
    for key in sorted(params.keys()):
        print(f"  {key}: {params[key]}")

    print()
    print("=" * 80)
    print("PARAMETERS FOR ALL 8 VARIATIONS:")
    print("=" * 80)

    all_params = []
    for name, run_id in run_dict.items():
        run = client.get_run(run_id)
        params = run.data.params

        # Extract key parameters
        row = {
            "name": name,
            "run_id": run_id,
            "var_list": params.get("var_list", "N/A"),
            "learning_rate": params.get("learning_rate", "N/A"),
            "n_epochs": params.get("n_epochs", "N/A"),
            "batch_size": params.get("batch_size", "N/A"),
            "hidden_size": params.get("hidden_size", "N/A"),
            "train_size_dimension": params.get("train_size_dimension", "N/A"),
            "train_size_fraction": params.get("train_size_fraction", "N/A"),
            "mlflow_tracking_uri": params.get("mlflow_tracking_uri", "N/A"),
        }
        all_params.append(row)

    df = pd.DataFrame(all_params)
    print(df.to_string(index=False))

    print()
    print("=" * 80)
    print("TRAINING HUCS:")
    print("=" * 80)
    train_hucs = params.get("train_hucs", "Not found")
    val_hucs = params.get("val_hucs", "Not found")
    print(f"Train HUCs: {train_hucs}")
    print(f"Val HUCs: {val_hucs}")

except Exception as e:
    print(f"ERROR: {e}")
    print()
    print("This might mean:")
    print("1. You need to authenticate to AWS first")
    print("2. The MLflow server is down")
    print("3. You don't have permission to access these runs")
    print()
    print("Try running: aws sagemaker describe-mlflow-tracking-server \\")
    print("               --tracking-server-name dawgsML \\")
    print("               --region us-west-2")
