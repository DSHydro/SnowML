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


class SnowModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_class, num_layers, dropout):
        super(SnowModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.lstm1 = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            dropout=self.dropout,
            batch_first=True)
        self.linear = nn.Linear(hidden_size, num_class)
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x):
        device = x.device
        hidden_states = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        cell_states = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm1(x, (hidden_states, cell_states))
        out = self.linear(out[:, -1, :])
        out = self.leaky_relu(out)
        return out


# Helper Function: Load data into DataLoader
def create_dataloader(df, params):
    """ Creates a DataLoader for a given HUC dataset """
    train_data, _, _, _ = pp.train_test_split(df, params["train_size_fraction"])
    X_train, y_train = pp.create_tensor(train_data, params["lookback"], params["var_list"])

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
def fine_tune(model, optimizer, loss_fn, df_dict, target_key, params):

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
    r = np.corrcoef(y_true.ravel(), y_pred.ravel())[0, 1] # Correlation coefficient
    alpha = np.std(y_pred) / np.std(y_true)  # Variability ratio
    beta = np.mean(y_pred) / np.mean(y_true)  # Bias ratio
    kg = 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
    #print(f"r: {r}, alpha: {alpha}, beta: {beta}")
    return kg, r, alpha, beta

def store_metrics(metric_names, metrics_list_dict, available_keys, epoch):
    df = pd.DataFrame({
        metric_names[0]: metrics_list_dict[metric_names[0]],
        metric_names[1]: metrics_list_dict[metric_names[1]],
        metric_names[2]: metrics_list_dict[metric_names[2]],
        metric_names[3]: metrics_list_dict[metric_names[3]]
        }, index=available_keys)

    # Compute the mean for each metric
    if df.shape[0] > 1:
        mean_values = df.mean()
        print("Mean Metrics:")
        print(mean_values)
        # Log each mean metric in mlflow
        for metric_name, mean_value in mean_values.items():
            mlflow.log_metric(f"mean_{metric_name}", mean_value, step = epoch)
        df.loc['Total'] = mean_values

    # Log the dataframe in mlflow
    csv_path = f"results_epoch{epoch}.csv"
    df.to_csv(csv_path, index=True)
    mlflow.log_artifact(csv_path)
    os.remove(csv_path)


def predict(model_dawgs, df_dict, selected_key, params):
    data = df_dict[selected_key]
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
        train_mse = mean_squared_error(y_train_true, y_train_pred)
        test_mse = mean_squared_error(y_test_true, y_test_pred)

        # Compute KGE
        train_kge, _, _, _ = kling_gupta_efficiency(y_train_true, y_train_pred)
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
            plot(data, y_train_pred, y_test_pred, train_size_main, selected_key, params)

    store_metrics(metric_names, metrics_list_dict, available_keys, epoch)


def plot(data, y_train_pred, y_test_pred, train_size, huc_id, params):

    train_plot = np.full_like(data['mean_swe'].values, np.nan, dtype=float)
    test_plot = np.full_like(data['mean_swe'].values, np.nan, dtype=float)

    # Convert tensor to numpy safely
    train_plot[params["lookback"]:train_size] = y_train_pred.flatten()
    test_plot[train_size + params["lookback"] : len(data)] = y_test_pred.flatten()

    # plot
    plt.figure(figsize=(12,  6))
    plt.plot(data.index, data['mean_swe'], c='b', label='Actual')
    plt.plot(data.index, train_plot, c='r', label='Train Predictions')
    plt.plot(
        data.index[train_size+params["lookback"]:],
        test_plot[train_size+params["lookback"]:],
        c='g',
        label='Test Predictions')
    plt.legend()
    plt.xlabel('Date')
    plt.ylabel('swe')
    ttle = f"SWE_Predictions_for_huc{huc_id} using Baseline Model" # TO DO: Make dynamic
    plt.title(ttle)
    #plt.savefig(f"{ttle}.png")
    mlflow.log_figure(plt.gcf(), ttle+".png")
    plt.close()
    