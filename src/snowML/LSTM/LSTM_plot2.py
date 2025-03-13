""" Module to plot actual and predicted model values; warm colors and flexible scale"""
# pylint: disable=C0103

import matplotlib.pyplot as plt
import mlflow
import numpy as np

def plot(data, y_train_pred, y_test_pred, train_size, huc_id, params, mlflow_on=True, metrics_dict=None):
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
    plt.plot(
        data.index[train_size + params["lookback"]:],
        test_plot[train_size + params["lookback"]:],
        c='g',
        label='LSTM Predictions Forecasting Phase'
    )
    plt.legend()
    plt.xlabel('Date')
    plt.ylabel('SWE')
    mdl_name = params["expirement_name"]
    ttle = f"SWE_Predictions_for_huc{huc_id} using {mdl_name}"
    plt.title(ttle)

    # Display metrics in the upper-right corner if metrics_dict is not None
    if metrics_dict is not None:
        ax = plt.gca()
        metric_text = "\n".join([f"{key}: {value:.3f}" for key, value in metrics_dict.items()])
        
        ax.text(
            0.02, 0.98, metric_text, transform=ax.transAxes, ha='left', va='top',
            fontsize=10, color='black', bbox=dict(facecolor='white', alpha=0.7, edgecolor='black')
        )


  
    plt.savefig(f"docs/model_results/{ttle}.png")
    plt.close()
    
