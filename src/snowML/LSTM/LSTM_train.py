# pylint: disable=C0103, R0913, R0914, R0917

import random
import torch
import numpy as np
from snowML.LSTM import LSTM_pre_process as pp
#from snowML.LSTM import LSTM_plot
#from snowML.LSTM import LSTM_plot2
from snowML.LSTM import LSTM_plot3 as plot3
from snowML.LSTM import LSTM_metrics as met
from snowML.LSTM import LSTM_predict_recursive as recur

#import importlib
#importlib.reload(recur)


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

    #avg_loss = np.mean(loss_val_list)
    #print(f"Average loss for epoch {epoch} is {avg_loss}")
    #mlflow.log_metric("avg_training_loss", avg_loss, step=epoch)


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


def predict(model_dawgs, data, X_te, params, X_tr=None):
    with torch.no_grad():
        y_te_pred = model_dawgs(X_te).cpu().numpy()

        if X_tr is not None:
            y_tr_pred = model_dawgs(X_tr).cpu().numpy()
        else:
            y_tr_pred = None

        if params["recursive_predict"]:
            # Index of the lagged SWE variable in the input features
            lagged_swe_idx = params['lag_swe_var_idx']

            # Recursive forecast on test set
            y_te_pred_recur = recur.recursive_forecast(
                model_dawgs,
                data,
                lagged_swe_idx,
                params
                )
            y_te_pred_recur = np.array(y_te_pred_recur).reshape(-1, 1)
        else:
            y_te_pred_recur = None
    return y_tr_pred, y_te_pred, y_te_pred_recur


def predict_prep(model_dawgs, df_dict, selected_key, params):
    """
    Generates predictions using the given model for a specified dataset.

    Args:
        model_dawgs (torch.nn.Module): The trained PyTorch model used for predictions.
        df_dict (dict): Dictionary containing multiple datasets, with keys representing 
                        dataset identifiers (e.g., `huc_id`).
        selected_key (str or int): The key used to select the dataset from `df_dict`.
        params (dict): Dictionary containing parameters for data preprocessing and 
                       model input, including:
            - "train_size_dimension" (str): Determines how the dataset is split 
              (e.g., "time" for time-based splits).
            - "train_size_fraction" (float): Fraction of the dataset used for training 
              (only relevant when splitting by time).
            - "lookback" (int): Number of past time steps used as input for predictions.
            - "var_list" (list): List of variable names used as features.

    Returns:
        tuple:
            - data (pd.DataFrame): The full dataset corresponding to `selected_key`.
            - y_train_pred (numpy.ndarray or None): Model predictions on training data 
              (if applicable, else `None`).
            - y_test_pred (numpy.ndarray): Model predictions on test data.
            - y_train (numpy.ndarray or None): Actual target values for training data 
              (if applicable, else `None`).
            - y_test (numpy.ndarray): Actual target values for test data.
            - train_size_main (int or float): The number of training samples (or `0` if no split).

    Notes:
        - If `train_size_dimension` is `"time"`, the dataset is split into training and testing sets 
          based on time, and predictions are made for both.
        - Otherwise, predictions are made on the entire dataset without a train-test split.
        - The function uses PyTorch tensors for predictions and converts outputs to NumPy arrays.
    """

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
    
        y_train = y_train.numpy()
        y_test = y_test.numpy()
        y_tr_pred, y_te_pred, y_te_pred_recur = predict(model_dawgs, data, X_test, params, X_tr=X_train)

    else: # split along entire huc
        X_test, y_test = pp.create_tensor(data, params['lookback'], params['var_list'])
        y_test = y_test.numpy()
        y_tr_pred = None
        y_train = None
        train_size_main = 0
        y_tr_pred, y_te_pred, y_te_pred_recur = predict(model_dawgs, data, X_test, params)
    return data, y_tr_pred, y_te_pred, y_train, y_test, y_te_pred_recur, train_size_main


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

    # Loop through each HUC
    for selected_key in available_keys:
        print(f"evaluating on huc {selected_key}")
        data, y_tr_pred, y_te_pred, y_tr_true, y_te_true, y_te_pred_recur, train_size = (
            predict_prep(model_dawgs, df_dict, selected_key, params))

        # test metrics
        metric_dict_test = met.calc_metrics(y_te_true, y_te_pred, metric_type = "test")

        # test_metrics_recur if avail
        if params["recursive_predict"]:
            metric_dict_te_recur = met.calc_metrics(y_te_true, y_te_pred_recur, metric_type = "test_recur")
        else:
            metric_dict_te_recur = None

        # train metrics if avail
        if params["train_size_dimension"] == "time":
            metric_dict_train = met.calc_metrics(y_tr_true, y_tr_pred, metric_type = "train")
            kge_tr = metric_dict_train["train_kge"]
        else:
            metric_dict_train = None
            kge_tr = None

        #Log and print metrics
        for m_dict in [metric_dict_test, metric_dict_te_recur, metric_dict_train]:
            met.log_print_metrics(m_dict, selected_key, epoch)
        
    return kge_tr, metric_dict_test, metric_dict_te_recur, metric_dict_train, data, y_te_true, y_te_pred, y_te_pred_recur, train_size
