""" Script to eval from a single multi-huc trained model """

import mlflow 
import pandas as pd
from snowML.LSTM import LSTM_evaluate as evaluate
import json
import os

import importlib
importlib.reload(evaluate)


# Define constants needed to retrieve model 
mlflow_tracking_uri = "arn:aws:sagemaker:us-west-2:677276086662:mlflow-tracking-server/dawgsML" # update with your tracking_uri
model_uri_prefix = "s3://sues-test/298/51884b406ec545ec96763d9eefd38c36/artifacts/epoch27_model"
run_id = "51884b406ec545ec96763d9eefd38c36" # capricious snipe


def eval_huc (huc_id):
    model_uri = model_uri_prefix
    model_dawgs, df_dict_test, params  = evaluate.set_up(
        [huc_id], 
        run_id, 
        model_uri, 
        mlflow_tracking_uri, 
        recur_predict = False)
   

    metric_dict_test, _, data, _, y_te_pred, _, y_te_true, _, train_size = evaluate.eval_from_saved_model(
        model_dawgs, df_dict_test, huc_id, params)
    
    results_df = join_results(data, y_te_pred, train_size, params["lookback"])

    save_results(metric_dict_test, results_df, huc_id)

    return metric_dict_test, results_df

def join_results(data, y_te_pred, train_size, lookback): 
    results = data.iloc[train_size + lookback:].copy()
    results["mean_swe_predict"] = y_te_pred.flatten()
    return results[["mean_swe", "mean_swe_predict"]]


def save_results(md, results_df, huc):
    folder = "mlflow_data/predictions"
    os.makedirs(folder, exist_ok=True)  # Make sure the folder exists

    f_dict = f"{folder}/metric_dict_{huc}_multi.json"
    f_df = f"{folder}/predictions_{huc}_multi.csv"

    results_df.to_csv(f_df)  

    # Save md as a JSON file
    md_clean = {k: float(v) for k, v in md.items()}
    with open(f_dict, 'w') as f:
        json.dump(md_clean, f, indent=4)

def eval_all(huc_list): 
    for huc in huc_list: 
        eval_huc(huc)

