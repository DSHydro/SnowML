""" Module to download metrics from MLflow """
# pylint: disable=C0103

import os
import mlflow
import pandas as pd

def create_run_dict_Ex3():
    """ Create a dictionary of run_ids by recognizable short names """

    run_dict = {}
    run_dict["base_1e-3"] = "a6c611d4c4cf410e9666796e3a8892b7" # debonair_dove (all 30)
    run_dict["hum_1e-3"] = "d71b47a8db534a059578162b9a8808b7" #peaceful stork (all 30)
    run_dict["srad_1e-3"] = "deed782fda71472fb47cf8670b668473" # enchanting-roo (aborted at 27)
    run_dict["vs_1e-3"] = "4653005687094d9ba54c295b943a4667" # puzzled cow (all 30)

    run_dict["base_3e-4"] = "e989030c272d4de59c84aff739d8063c" # spiffy whale (all 30)
    run_dict["hum_3e-4"] = "51884b406ec545ec96763d9eefd38c36" # capricious snipe (all 30)
    run_dict["srad_3e-4"] = "2b49d6cce3844ede8a66821ae9aec27b" # judicious_mare (all 30)
    run_dict["vs_3e-4"] = "bc031cafad7445adb73173adc43b63c6" # placid-croc (all 30)

    return run_dict


# function to retrieve metrics from ML server
def load_ml_metrics(
    run_id,
    save_local=True,
    folder = "data/",
    tracking_uri = "arn:aws:sagemaker:us-west-2:677276086662:mlflow-tracking-server/dawgsML"):

    """ Function that retreives all the logged ML metrics for a give run_id"""

    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.MlflowClient()
    run_data = client.get_run(run_id).data
    metric_keys = run_data.metrics.keys()
    # Retrieve full metric history for each key
    all_metrics = []
    for metric in metric_keys:
        history = client.get_metric_history(run_id, metric)
        for record in history:
            all_metrics.append({
                "Metric": metric,
                "Step": record.step,
                "Value": record.value
            })

    # Convert to DataFrame
    metrics_df = pd.DataFrame(all_metrics)

    # Save to CSV if needed
    if save_local:
        f_out = f"{folder}/run_id_data/metrics_from_{run_id}.csv"
        metrics_df.to_csv(f_out, index=False)

    return metrics_df

def download_all(run_dict, folder ="data/", overwrite = False):
    for run_id in run_dict.values():
        f_out =  f"{folder}/run_id_data/metrics_from_{run_id}.csv"
        print(f"processing file {f_out}")
        if (not os.path.exists(f_out) or overwrite):
            load_ml_metrics(run_id)
        else:
            print(f"File already exists: {f_out}")
