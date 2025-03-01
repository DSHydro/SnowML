# pylint: disable=C0103, R0913, R0914, R0917

import random
import torch
import numpy as np
import matplotlib.pyplot as plt
import mlflow
from sklearn.metrics import mean_squared_error
from snowML.LSTM import LSTM_pre_process as pp


# Helper Function: Load data into DataLoader
def create_dataloader(df, params):
    """ Creates a DataLoader for a given HUC dataset """
    if params["train_size_dimension"] == "time":
        train_data, _, _, _ = pp.train_test_split_time(
                                    df, params["train_size_fraction"])
        X_train, y_train = pp.create_tensor(train_data,
                                    params["lookback"], params["var_list"])
    else:
        X_train, y_train = pp.create_tensor(df,
                                    params["lookback"], params["var_list"])

    # Create a DataLoader
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_train, y_train),
        shuffle=True,
        batch_size=params["batch_size"],
        num_workers=params["num_workers"]  # Enable multi-threading for data loading
    )
    return loader

# Pre-training Phase
def pre_train(model, optimizer, loss_fn, df_dict, params, epoch):
    """ First training pass. Model saved for potential fine tuning."""

    # Initialize available keys for sampling without replacement

    available_keys = list(df_dict.keys())
    random.shuffle(available_keys)

    if params["loss_type"] == "hybrid":
        loss_fn.set_epoch(epoch)
    model.train()  # Set model to training mode

    loss_val_list = []

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

            loss_val_list.append(loss.item())

    avg_loss = np.mean(loss_val_list)
    print(f"Average loss for epoch {epoch} is {avg_loss}")
    mlflow.log_metric("avg_training_mse", avg_loss, step=epoch)


def fine_tune(model, optimizer, loss_fn, df_train, params, epoch):
    """
    Fine-tunes the given model on the target dataset specified by target_key.

    Args:
        model (torch.nn.Module): The model to be fine-tuned.
        optimizer (torch.optim.Optimizer): The optimizer ufor updating the model.
        loss_fn (torch.nn.Module): The loss function used to calc loss.
        df_train (dict): The training data. 
        params (dict): A dictionary of parameters for creating the DataLoader.
        epoch (int): The current epoch number.

    Returns:
        None
    """

    # Create DataLoader for fine-tuning (target HUC)
    loader = create_dataloader(
        df_train,
        params
        )

    if params["loss_type"] == "hybrid":
        loss_fn.set_epoch(epoch)
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
    """
    Calculate the Kling-Gupta Efficiency (KGE) and its components.

    The KGE is a metric used to evaluate the performance of hydrological models.
    It is composed of three components: correlation (r), variability ratio 
    (alpha), and bias ratio (beta).

    Parameters:
    y_true (np.ndarray): Array of true values.
    y_pred (np.ndarray): Array of predicted values.

    Returns:
    tuple: A tuple containing the following elements:
        - kg (float): Kling-Gupta Efficiency.
        - r (float): Correlation coefficient between y_true and y_pred.
        - alpha (float): Variability ratio (standard deviation of y_pred / 
            standard deviation of y_true).
        - beta (float): Bias ratio (mean of y_pred / mean of y_true).

    Notes:
    - If NaN values are detected in y_true or y_pred, the function will print 
        an error message and return (np.nan, np.nan, np.nan, np.nan).
    - If zero variance is detected in y_true or y_pred, the function will print 
        an error message and return (np.nan, np.nan, np.nan, np.nan).

    Example:
    >>> y_true = np.array([1, 2, 3, 4, 5])
    >>> y_pred = np.array([1.1, 2.1, 2.9, 4.1, 5.1])
    >>> kling_gupta_efficiency(y_true, y_pred)
    (0.993, 0.998, 1.0, 1.02)
    """
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

def store_summ_metrics(metric_names, metrics_list_dict, epoch):
    """
   Computes and logs the mean and median of given metrics for each epoch.

    Parameters:
        metric_names (list of str): List of metric names to be stored and logged.
        metrics_list_dict (dict): Dict of metric values, keyed by metric names.
        epoch (int): The current epoch number, used for logging metrics in mlflow.

    Returns:
        None
    """
    for i in range(len(metric_names)):
        metric_nm = metric_names[i]
        metric_list = metrics_list_dict[metric_nm]
        mean_value = np.mean(metric_list)
        print(f"mean_{metric_nm} is {mean_value}")
        mlflow.log_metric(f"mean_{metric_nm}", mean_value, step=epoch)
        median_value = np.median(metric_list)
        print(f"median_{metric_nm} is {median_value}")
        mlflow.log_metric(f"median_{metric_nm}", median_value, step=epoch)


def predict (model_dawgs, df_dict, selected_key, params):
    data = df_dict[selected_key]

    if params["train_size_dimension"] == "time":
        train_main, test_main, train_size_main, _  = pp.train_test_split_time(
                                        data, params['train_size_fraction'])
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
    """
    Evaluate the performance of the model on the given dataset.

    Parameters:
        model_dawgs (object): The trained model to be evaluated.
        df_dict (dict): Dictionary of dataframes for each HUC to be evaluated.
        params (dict): Dictionary containing the parameters for evaluation.
        epoch (int): The current epoch number.
        selected_keys (list, optional): List of keys to be evaluated. 
            If None, all keys in df_dict are used.

    Returns:
        None
    """
    if selected_keys is None:
        available_keys = list(df_dict.keys())
        random.shuffle(available_keys)

    else:
        available_keys = selected_keys
    metric_names = ["train_mse", "test_mse", "train_kge", "test_kge"]
    metrics_list_dict = {metric: [] for metric in metric_names}

    # Loop through each HUC
    for selected_key in available_keys:
        print(f"evaluating on huc {selected_key}")
        data, y_train_pred, y_test_pred, y_train_true, y_test_true, train_size_main = predict(model_dawgs, df_dict, selected_key, params)

        # Compute MSE
        if params["train_size_dimension"] == "time":
            train_mse = mean_squared_error(y_train_true, y_train_pred)
        else:
            train_mse = -500
        metrics_list_dict["train_mse"].append(train_mse)
        test_mse = mean_squared_error(y_test_true, y_test_pred)
        metrics_list_dict["test_mse"].append(test_mse)

        # Compute KGE
        if params["train_size_dimension"] == "time":
            train_kge, _, _, _ = kling_gupta_efficiency(y_train_true, y_train_pred)
        else:
            train_kge = -500
        metrics_list_dict["train_kge"].append(train_kge)
        test_kge, _, _, _ = kling_gupta_efficiency(y_test_true, y_test_pred)
        metrics_list_dict["test_kge"].append(test_kge)


        metrics = [train_mse, test_mse, train_kge, test_kge]

        # Log, print, & save metrics
        for i, metric in enumerate(metrics):
            mlflow.log_metric(f"{metric_names[i]}_{str(selected_key)}",
                              metric, step=epoch)
            print(f"{metric_names[i]}_{str(selected_key)}: {metric}")
            metrics_list_dict[metric_names[i]].append(metric)
        print("")


        # store plots for final epooch
        if epoch == params["n_epochs"] - 1:
            try:
                plot(data, y_train_pred, y_test_pred, train_size_main, selected_key, params)
            except Exception as e:
                print(f"Error occurred while plotting: {e}")
    
    if len(available_keys) > 1: 
        store_summ_metrics(metric_names, metrics_list_dict, epoch)


def plot(data, y_train_pred, y_test_pred, train_size, huc_id, params):
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
    #plt.savefig(f"{ttle}.png")
    mlflow.log_figure(plt.gcf(), ttle+".png")
    plt.close()
    