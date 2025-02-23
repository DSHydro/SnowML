# pylint: disable=C0103

import random
import os
import torch
from torch import nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
from sklearn.metrics import mean_squared_error
from snowML.LSTM import LSTM_pre_process as pp



# Helper Function: Load data into DataLoader
def create_dataloader(df, params):
    """ Creates a DataLoader for a given HUC dataset """
    if params["train_size_dimension"] == "time":
        train_data, _, _, _ = pp.train_test_split(df, params["train_size_fraction"])
        X_train, y_train = pp.create_tensor(train_data, params["lookback"], params["var_list"])
    else:
        X_train, y_train = pp.create_tensor(df, params["lookback"], params["var_list"])

    # Create a DataLoader
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_train, y_train),
        shuffle=True,
        batch_size=params["batch_size"],
        num_workers=params["num_workers"]  # Enable multi-threading for data loading
    )
    return loader

# Pre-training Phase: Train on multiple HUCs
def pre_train(model, optimizer, loss_fn, df_dict, params):
    """ Pre-train the model on multiple HUCs """
    # Initialize available keys for sampling without replacement

    available_keys = list(df_dict.keys())
    random.shuffle(available_keys)

    model.train()  # Set model to training mode


    for i, selected_key in enumerate(available_keys, start=1):
        if i % 5 == 0:
            print(f"Pre-training on HUC{i}: {selected_key}")

        loader = create_dataloader(
            df_dict[selected_key],
            params
            )

        # Training Loop
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            loss.backward()
            optimizer.step()


# Fine-tuning Phase: Train on target HUC
def fine_tune(model, optimizer, loss_fn, df_dict, target_key, params, epoch):

    df_target = df_dict[target_key]

    # Create DataLoader for fine-tuning (target HUC)
    loader = create_dataloader(
        df_target,
        params
        )

    model.train()  # Set model to training mode
    #print(f"Epoch {epoch}: Fine-tuning on target HUC {target_key}")

    # Training Loop on target HUC
    for X_batch, y_batch in loader:
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = loss_fn(y_pred, y_batch)
        loss.backward()
        optimizer.step()


def kling_gupta_efficiency(y_true, y_pred):
    # Check for NaNs
    if np.isnan(y_true).any() or np.isnan(y_pred).any():
        print("Error: NaN values detected in y_true or y_pred")
        return np.nan, np.nan, np.nan, np.nan

    # Check for zero variance
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        print("Error: Zero variance detected in y_true or y_pred")
        return np.nan, np.nan, np.nan, np.nan

    # Compute correlation
    r = np.corrcoef(y_true.ravel(), y_pred.ravel())[0, 1]

    # Compute KGE components
    alpha = np.std(y_pred) / np.std(y_true)  
    beta = np.mean(y_pred) / np.mean(y_true)  
    kg = 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)

    print(f"r: {r}, alpha: {alpha}, beta: {beta}")
    return kg, r, alpha, beta

# def kling_gupta_efficiency(y_true, y_pred):
#     r = np.corrcoef(y_true.ravel(), y_pred.ravel())[0, 1] # Correlation coefficient
#     alpha = np.std(y_pred) / np.std(y_true)  # Variability ratio
#     beta = np.mean(y_pred) / np.mean(y_true)  # Bias ratio
#     kg = 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
#     print(f"r: {r}, alpha: {alpha}, beta: {beta}")
#     return kg, r, alpha, beta

def store_metrics(metric_names, metrics_list_dict, available_keys, epoch):
    df = pd.DataFrame({
        metric_names[0]: metrics_list_dict[metric_names[0]],
        metric_names[1]: metrics_list_dict[metric_names[1]],
        metric_names[2]: metrics_list_dict[metric_names[2]],
        metric_names[3]: metrics_list_dict[metric_names[3]]
        }, index=available_keys)

    # Compute the mean and median for each metric
    if df.shape[0] > 1:
        mean_values = df.mean()
        median_values = df.median()

        print("Mean Metrics:")
        print(mean_values)
        print("")

        print("Median Metrics:")
        print(median_values)
        print("")

        # Log each mean and median metric in mlflow
        for metric_name, mean_value in mean_values.items():
            mlflow.log_metric(f"mean_{metric_name}", mean_value, step=epoch)

        for metric_name, median_value in median_values.items():
            mlflow.log_metric(f"median_{metric_name}", median_value, step=epoch)

        df.loc['Total'] = mean_values



def predict (model_dawgs, df_dict, selected_key, params):
    data = df_dict[selected_key]

    if params["train_size_dimension"] == "time":
        train_main, test_main, train_size_main, _  = pp.train_test_split(data, params['train_size_fraction'])
        X_train, y_train = pp.create_tensor(train_main,
                                        params['lookback'],
                                        params['var_list'])

        X_test, y_test = pp.create_tensor(test_main,
                                    params['lookback'],
                                    params['var_list'])
        with torch.no_grad():
            y_train_pred = model_dawgs(X_train).cpu().numpy()
            y_test_pred = model_dawgs(X_test).cpu().numpy()
            y_train = y_train.numpy()
            y_test = y_test.numpy()

    else: # split along entire huc
        X_test, y_test = pp.create_tensor(data, params['lookback'], params['var_list'])
        with torch.no_grad():
            y_test_pred = model_dawgs(X_test).cpu().numpy()
            y_test = y_test.numpy()
        y_train_pred = None
        y_train = None
        train_size_main = 0
    return data, y_train_pred, y_test_pred, y_train, y_test, train_size_main


def evaluate(model_dawgs, df_dict, params, epoch, selected_keys = None):
    if selected_keys is None:
        available_keys = list(df_dict.keys())
        random.shuffle(available_keys)
        

    else:
        available_keys = selected_keys
    metric_names = ["train_mse", "test_mse", "train_kge", "test_kge"]
    metrics_list_dict = {metric: [] for metric in metric_names}

    # Loop through each HUC
    for selected_key in available_keys:
        data, y_train_pred, y_test_pred, y_train_true, y_test_true, train_size_main = predict(model_dawgs, df_dict, selected_key, params)
        # Compute MSE
        if params["train_size_dimension"] == "time":
            train_mse = mean_squared_error(y_train_true, y_train_pred)
        else:
            train_mse = -500  # TO DO: Make more elegant
        test_mse = mean_squared_error(y_test_true, y_test_pred)

        # Compute KGE
        if params["train_size_dimension"] == "time":
            train_kge, _, _, _ = kling_gupta_efficiency(y_train_true, y_train_pred)
        else:
            train_kge = -500 # TO DO: Make more elegant
        test_kge, _, _, _ = kling_gupta_efficiency(y_test_true, y_test_pred)

        metrics = [train_mse, test_mse, train_kge, test_kge]


        # Log, print, & save metrics
        for i in range(len(metrics)):
            mlflow.log_metric(f"{metric_names[i]}_{str(selected_key)}", metrics[i], step=epoch)
            print(f"{metric_names[i]}_{str(selected_key)}: {metrics[i]}")
            metrics_list_dict[metric_names[i]].append(metrics[i])
        print("")


        # store plots for final epooch
        if epoch == params["n_epochs"] - 1:
            try:
                plot(data, y_train_pred, y_test_pred, train_size_main, selected_key, params)
            except Exception as e:
                print(f"Error occurred while plotting: {e}")

    store_metrics(metric_names, metrics_list_dict, available_keys, epoch)


def plot(data, y_train_pred, y_test_pred, train_size, huc_id, params):

    train_plot = np.full_like(data['mean_swe'].values, np.nan, dtype=float)
    test_plot = np.full_like(data['mean_swe'].values, np.nan, dtype=float)

    # Convert tensor to numpy safely
    if params["train_size_dimension"] == "time":
        train_plot[params["lookback"]:train_size] = y_train_pred.flatten()
    test_plot[train_size + params["lookback"] : len(data)] = y_test_pred.flatten()

    # plot
    plt.figure(figsize=(12,  6))
    # Use consistent y axis to enable comparison between hucs
    plt.ylim(0, 2)
    plt.plot(data.index, data['mean_swe'], c='b', label='Actual')
    if params["train_size_dimension"] == "time":
        plt.plot(data.index, train_plot, c='r', label='Train Predictions')
    plt.plot(
        data.index[train_size+params["lookback"]:],
        test_plot[train_size+params["lookback"]:],
        c='g',
        label='Test Predictions')
    plt.legend()
    plt.xlabel('Date')
    plt.ylabel('swe')
    mdl_name = params["expirement_name"]
    ttle = f"SWE_Predictions_for_huc{huc_id} using {mdl_name}"
    plt.title(ttle)
    #plt.savefig(f"{ttle}.png")
    mlflow.log_figure(plt.gcf(), ttle+".png")
    plt.close()
    