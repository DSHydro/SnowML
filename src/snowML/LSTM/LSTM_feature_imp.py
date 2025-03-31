""" Module to Evaluate Feature Importance From a Trained Model """
# pylint: disable=C0103

import time
import mlflow
import mlflow.pytorch
import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np
from captum.attr import IntegratedGradients
from snowML.LSTM import LSTM_evaluate as LSTM_eval
from snowML.LSTM import LSTM_pre_process as pp
from snowML.datapipe import data_utils as du


def create_X_tensor(df_dict, huc, params):
    data = df_dict[huc]
    # no train/test split
    X_test, _ = pp.create_tensor(data, params['lookback'], params['var_list'])
    return X_test

def plot_feature_imp(feature_importance, huc, var_list):
    mean_importance = np.mean(np.abs(feature_importance), axis=(0, 1))  # Mean over samples & time

    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(len(mean_importance)), mean_importance)  # Now 1D
    plt.xlabel("Feature")
    plt.ylabel("Importance Score")

    # Set the x-tick labels using the var_list
    if len(var_list) == len(mean_importance):
        plt.xticks(range(len(mean_importance)), var_list, rotation=90)
    else:
        print("Warning: Length of var_list does not match num features. Using default labels.")
        plt.xticks(range(len(mean_importance)), range(len(mean_importance)), rotation=90)

    ttl = f"Feature_Importance_using_Integrated_Gradients_for_{huc}"
    plt.title(ttl)

    f_out = f"feature_importance_graphs/{ttl}.png"
    plt.savefig(f_out, dpi=300)
    print(f"Figure saved for {huc}")
    plt.close()


def feature_imp(df_dict_test, huc, batch_size, ig, params):
    X_tensor = create_X_tensor(df_dict_test, huc, params)
    baseline = torch.zeros_like(X_tensor)
    num_samples = X_tensor.shape[0]
    attributions_list = []
    for i in range(0, num_samples, batch_size):
        X_batch = X_tensor[i : i + batch_size]
        baseline_batch = baseline[i : i + batch_size]

        print(f"Processing batch {i // batch_size + 1} / {num_samples // batch_size + 1}")

        attributions_batch = ig.attribute(X_batch, baselines=baseline_batch, target=0)
        attributions_list.append(attributions_batch.detach().cpu()) # Move to CPU to free GPU memory

        attributions = torch.cat(attributions_list, dim=0)  # Concatenate batches

    feature_importance = attributions.numpy()  # Convert to NumPy
    return feature_importance

def feature_imp_all(model_uri, mlflow_tracking_uri, run_id, test_hucs, mlflow_log_now=False):
    model = LSTM_eval.load_model(model_uri)
    model.eval()
    ig = IntegratedGradients(model)
    params = LSTM_eval.get_params(mlflow_tracking_uri, run_id)


    if params["train_size_dimension"] == "huc":
        df_dict_test = LSTM_eval.assemble_df_dict(test_hucs, params["var_list"])
        # normalize test data using same means/standard dev used in training
        df_dict_test = LSTM_eval.renorm(params["train_hucs"],  params["val_hucs"],
            test_hucs, params["var_list"])
    else:
        # normalize test huc against itself only (as in training)
        df_dict_test =  pp.pre_process_separate(test_hucs, params["var_list"])


    if mlflow_log_now:
        mlflow.set_experiment("Feature_Importance")
        with mlflow.start_run():
            mlflow.log_params(params)
            mlflow.log_param("test_hucs", test_hucs)
            mlflow.log_param("model_uri", model_uri)

    batch_size = 128 # Adjust based on memory availability
    feature_importance_data = {}
    for huc in test_hucs:
        time_start = time.time()
        feature_importance = feature_imp(df_dict_test, huc, batch_size, ig, params)
        var_list = params["var_list"]
        plot_feature_imp(feature_importance, huc, var_list)  # Visualization
        mean_importance = np.mean(np.abs(feature_importance), axis=(0, 1)) # Mean over samples&time
        du.elapsed(time_start)

        # Store mean importance in the dictionary with huc as the key
        feature_importance_data[huc] = mean_importance
        importance_df = pd.DataFrame.from_dict(feature_importance_data, orient="index", columns=var_list)

        if mlflow_log_now:
            for i in range(importance_df.shape[0]):
                for var in var_list:
                    metric_name = f"{huc}_importance_{var}"
                    mlflow.log_metric(metric_name,importance_df.iloc[i][var])
    return importance_df
