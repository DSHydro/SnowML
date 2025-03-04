""" Module to plot actual and predicted model values"""
# pylint: disable=C0103

import matplotlib.pyplot as plt
import mlflow
import numpy as np

def plot(data, y_train_pred, y_test_pred, train_size, huc_id, params, mlflow_on = True):
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

    Returns:
        None
    """

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
    if mlflow_on:
        mlflow.log_figure(plt.gcf(), ttle+".png")
    else:
        plt.savefig(f"{ttle}.png")
    plt.close()
