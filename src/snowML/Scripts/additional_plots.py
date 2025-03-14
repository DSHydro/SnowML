""" Module to make poster-ready plots"""
# pylint: disable=C0103

from snowML.LSTM import LSTM_evaluate as eval
from snowML.Scripts import additional_plots as plots
import matplotlib.pyplot as plt
import mlflow
import numpy as np


def load_model(model_uri):
    """
    Load a PyTorch model from the given URI using MLflow.

    Args:
        model_uri (str): The URI of the model to load.

    Returns:
        torch.nn.Module: The loaded PyTorch model.
    """
    print(model_uri)
    model = mlflow.pytorch.load_model(model_uri)
    print(model)
    return model

def plot_basic(data, y_train_pred, y_test_pred, train_size, huc_id, 
         lookback= 180, train_size_dim = 'huc', mlflow_on=False, metrics_dict=None):
    """
    Plots the actual and predicted SWE (Snow Water Equivalent) values for 
    training and testing datasets.
    """
    train_plot = np.full_like(data['mean_swe'].values, np.nan, dtype=float)
    test_plot = np.full_like(data['mean_swe'].values, np.nan, dtype=float)

    # Convert tensor to numpy safely
    if train_size_dim == "time":
        train_plot[lookback:train_size] = y_train_pred.flatten()
    test_plot[train_size + lookback : len(data)] = y_test_pred.flatten()

    # Plot
    plt.figure(figsize=(12, 6))
    #plt.ylim(0, 2)
    plt.plot(data.index, data['mean_swe'], c='black', label='SWE Estimates From Physics Based Model')
    if params["train_size_dimension"] == "time":
        plt.plot(data.index, train_plot, c='#E6E6FA', label='LSTM Estimates Training Phase')
    plt.plot(
        data.index[train_size + lookback],
        test_plot[train_size + lookback:],
        c='g',
        label='LSTM Predictions - Forecasting'
    )
    plt.legend()
    plt.xlabel('Date')
    plt.ylabel('SWE')
    ttle = f"SWE_Predictions_for_huc{huc_id}"
    plt.title(ttle)

    # Display metrics in the upper-right corner if metrics_dict is not None
    if metrics_dict is not None:
        ax = plt.gca()
        metric_text = "\n".join([f"{key}: {value:.3f}" for key, value in metrics_dict.items()])
    
        ax.text(
            0.02, 0.98, metric_text, transform=ax.transAxes, ha='left', va='top',
            fontsize=10, color='black', bbox=dict(facecolor='white', alpha=0.7, edgecolor='black')
        )

    if mlflow_on:
        mlflow.log_figure(plt.gcf(), ttle + ".png")
    else:
        plt.savefig(f"docs/swe_prediction_plots/{ttle}.png")
    plt.close()


huc_to_plot = '170300010701'
test_hucs = [huc_to_plot]

model_uri = "s3://sues-test/298/51884b406ec545ec96763d9eefd38c36/artifacts/epoch27_model" 
run_id = "d71b47a8db534a059578162b9a8808b7" 
mlflow_tracking_uri = "arn:aws:sagemaker:us-west-2:677276086662:mlflow-tracking-server/dawgsML" 

model_dawgs = load_model(model_uri)

metric_dict, data, y_tr_pred, y_te_pred, y_tr_true, y_te_true, train_size_main= eval.eval_from_saved_model(model_dawgs,
         df_dict_test, huc_to_plot)
plot.plot_basic()
