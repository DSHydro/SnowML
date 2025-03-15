""" Module to make poster-ready plots"""
# pylint: disable=C0103

import ast
import mlflow
import numpy as np
from snowML.LSTM import LSTM_evaluate as eval
from snowML.LSTM import LSTM_pre_process as pp
import matplotlib.pyplot as plt



def plot_basic(data, y_train_pred, y_test_pred, train_size, huc_id, 
         params, mlflow_on=False, metrics_dict=None):
    """
    Plots the actual and predicted SWE (Snow Water Equivalent) values for 
    training and testing datasets.

    Args:
        data (pd.DataFrame): DataFrame containing the actual SWE values 
            with a datetime index.
        y_train_pred (np.ndarray): Array of predicted SWE for the training set.
        y_test_pred (np.ndarray): Array of predicted SWE for the testing set.
        train_size (int): The size of the training dataset.
        huc_id (str): Hydrologic Unit Code identifier for the dataset.
        params (dict): Dictionary containing parameters for the plot, including:
            - "train_size_dimension" (str): Training size dim, e.g., "time".
            - "lookback" (int): Number of time steps to consider for predictions.
            - "expirement_name" (str): Name of experiment for labeling the plot.
        metrics_dict (dict, optional): Dictionary of metrics to display on the plot. Defaults to None.

    Returns:
        None
    """

    train_plot = np.full_like(data['mean_swe'].values, np.nan, dtype=float)
    test_plot = np.full_like(data['mean_swe'].values, np.nan, dtype=float)

    # Convert tensor to numpy safely
    if params["train_size_dimension"] == "time":
        train_plot[params["lookback"]:train_size] = y_train_pred.flatten()
    test_plot[train_size + params["lookback"] : len(data)] = y_test_pred.flatten()

    # Plot
    plt.figure(figsize=(12, 6))
    #plt.ylim(0, 2)
    plt.plot(data.index, data['mean_swe'], c='b', label='SWE Estimates From Physics Based Model')
    if params["train_size_dimension"] == "time":
        plt.plot(data.index, train_plot, c='#E6E6FA', label='LSTM Predictions Training Phase')
    plt.subplots_adjust(right=0.8)
    plt.plot(
        data.index[train_size + params["lookback"]:],
        test_plot[train_size + params["lookback"]:],
        c='g',
        label='LSTM Predictions Forecasting Phase'
    )
    plt.legend()
    plt.xlabel('Date')
    plt.ylabel('SWE (meters)')
    mdl_name = params["expirement_name"]
    ttle = f"SWE_Predictions_for_huc_{huc_id}"
    plt.title(ttle)

    if metrics_dict is not None:
        metric_text = "\n".join([f"{key}: {value:.3f}" for key, value in metrics_dict.items()])
        plt.gcf().text(
            1.02, 0.5, metric_text, fontsize=10, color='black',
            ha='left', va='center', weight='bold', transform=plt.gca().transAxes
        )

    if mlflow_on:
        mlflow.log_figure(plt.gcf(), ttle + ".png")
    else:
        plt.savefig(f"docs/swe_prediction_plots/{ttle}.png", bbox_inches="tight")
    plt.close()


def plot_last_n_years(n, data, y_train_pred, y_test_pred, train_size, huc_id, 
         params, mlflow_on=False):

    train_plot = np.full_like(data['mean_swe'].values, np.nan, dtype=float)
    test_plot = np.full_like(data['mean_swe'].values, np.nan, dtype=float)

    # Convert tensor to numpy safely
    if params["train_size_dimension"] == "time":
        train_plot[params["lookback"]:train_size] = y_train_pred.flatten()
    test_plot[train_size + params["lookback"] : len(data)] = y_test_pred.flatten()

    # Plot
    plt.figure(figsize=(12, 6))
    #plt.ylim(0, 2)
    plt.plot(data.index[-n*365:], data['mean_swe'][-n*365:], c='b', label='SWE Estimates From Physics Based Model')
    if params["train_size_dimension"] == "time":
        plt.plot(data.index, train_plot, c='#E6E6FA', label='LSTM Predictions Training Phase')
    plt.subplots_adjust(right=0.8)
    plt.plot(
        data.index[-n*365:],
        test_plot[-n*365:],
        c='g',
        label='LSTM Predictions Forecasting Phase'
    )
    plt.legend()
    plt.xlabel('Date')
    plt.ylabel('SWE (meters)')
    mdl_name = params["expirement_name"]
    ttle = f"SWE_Predictions_for_huc_{huc_id}_last_{n}_years"
    plt.title(ttle)

   

    if mlflow_on:
        mlflow.log_figure(plt.gcf(), ttle + ".png")
    else:
        plt.savefig(f"docs/swe_prediction_plots/{ttle}.png", bbox_inches="tight")
    plt.close()
