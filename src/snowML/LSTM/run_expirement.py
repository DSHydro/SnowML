
# # pylint: disable=C0103

# Script to run an expiriment
import os
import importlib
import random
import torch
from torch import optim
from torch import nn
import mlflow
from snowML.LSTM import LSTM_pre_process as pp
from snowML.LSTM import snow_LSTM as snow
from snowML.LSTM import set_hyperparams as sh


libs_to_reload = [snow, pp, sh]
for lib in libs_to_reload:
    importlib.reload(lib)


def prep_input_data(params):
    """
    Prepares input data for the experiment.


    Args:
        params (dict): A dictionary containing parameters for data preparation. 
                       Expected keys include var_list (list): List of variables to be used.

    Returns:
        df_dict: A dictionary where keys are HUCs and values are preprocessed dataframes.
    """
    hucs = pp.assemble_huc_list(params)
    df_dict = pp.pre_process(hucs, params["var_list"])
    return df_dict

def split_df_dict(df_dict, train_size_fraction_hucs):
    huc_ids = list(df_dict.keys())
    random.shuffle(huc_ids)  # Shuffle the HUC IDs randomly

    split_idx = int(len(huc_ids) * train_size_fraction_hucs)
    train_hucs = set(huc_ids[:split_idx])

    df_dict_train = {huc_id: df for huc_id, df in df_dict.items() if huc_id in train_hucs}
    df_dict_test = {huc_id: df for huc_id, df in df_dict.items() if huc_id not in train_hucs}
    print(list(df_dict_train.keys())[:5])
    print(list(df_dict_test.keys())[:5])

    return df_dict_train, df_dict_test

def set_ML_server(params):
    """
    Configures the MLflow tracking server and sets the experiment.

    Returns:
        None
    """
    # Set our tracking server uri for logging
    #tracking_uri = "https://t-izowcn0gky2o.us-west-2.experiments.sagemaker.aws"
    tracking_uri = "arn:aws:sagemaker:us-west-2:677276086662:mlflow-tracking-server/dawgsML"
    mlflow.set_tracking_uri(tracking_uri)

    # Define the expirement
    mlflow.set_experiment(params["expirement_name"])

def initialize_model(params):
    """
    Initializes the SnowModel with the given parameters.

    Args:
        params (dict): A dictionary containing the following keys:
            - "var_list" (list): List of variables for the input size.
            - "hidden_size" (int): The number of features in the hidden state.
            - "num_class" (int): The number of output classes.
            - "num_layers" (int): The number of recurrent layers.
            - "dropout" (float): The dropout probability.

    Returns:
        tuple: A tuple containing the initialized model, optimizer, and loss function:
            - model_dawgs (SnowModel): The initialized SnowModel.
            - optimizer_dawgs (torch.optim.Optimizer): The optimizer for the model.
            - loss_fn_dawgs (torch.nn.modules.loss._Loss): The loss function for the model.
    """
    input_size=len(params["var_list"])
    model_dawgs = snow.SnowModel(
        input_size,
        params['hidden_size'],
        params['num_class'],
        params['num_layers'],
        params['dropout']
    )
    optimizer_dawgs = optim.Adam(model_dawgs.parameters())
    loss_fn_dawgs = nn.MSELoss()
    return model_dawgs, optimizer_dawgs, loss_fn_dawgs


def run_expirement(params = None):
    if params is None:
        params = sh.create_hyper_dict()
    df_dict = prep_input_data(params)
    if params["train_size_dimension"] == "huc":
        df_dict_train, df_dict_eval = split_df_dict(df_dict, params["train_size_fraction"])
    else:
        df_dict_train = df_dict
        df_dict_eval = df_dict
    set_ML_server(params)
    model_dawgs_pretrain, optimizer_dawgs, loss_fn_dawgs = initialize_model(params)

    with mlflow.start_run():
        # log all the params
        mlflow.log_params(params)

        # pre-train
        if params["train_size_dimension"] == "time":
            pre_train_epochs = int(params['n_epochs'] * params['pre_train_fraction'])
        else:
            pre_train_epochs = params["n_epochs"]

        for epoch in range(pre_train_epochs):
            print(f"Epoch {epoch}: Pre-training on multiple HUCs")

            # pre-train
            snow.pre_train(
                model_dawgs_pretrain,
                optimizer_dawgs,
                loss_fn_dawgs,
                df_dict_train,
                params
                )

            # evaluate
            snow.evaluate(
                model_dawgs_pretrain,
                df_dict_eval,
                params,
                epoch)

        # log the pre-trained model & save locally
        mlflow.pytorch.log_model(model_dawgs_pretrain, artifact_path="pre_trained_model")
        local_path = "model_dawgs_pretrain.pth"
        torch.save(model_dawgs_pretrain.state_dict(), local_path)

        # fine_tune (if applicable)


        if params["pre_train_fraction"] < 1:
            for i, target_key in enumerate(df_dict_eval.keys(), start=1):

                model_dawgs_ft, optimizer_dawgs_ft, loss_dawgs_ft = initialize_model(params)
                if params["pre_train_fraction"] > 0:
                    model_dawgs_ft.load_state_dict(torch.load(local_path))

                fine_tune_epochs = int(params['n_epochs'] - pre_train_epochs)

                for epoch in range(pre_train_epochs, fine_tune_epochs):

                    print(f"Epoch{epoch}, Fine-tuning on {target_key}, number {i}")
                    snow.fine_tune(
                        model_dawgs_ft,
                        optimizer_dawgs_ft,
                        loss_dawgs_ft,
                        df_dict_eval,
                        target_key,
                        params,
                        epoch,
                    )

                    if ((epoch-pre_train_epochs) % 1 == 0) or (epoch == params["n_epochs"]-1):
                        snow.evaluate(
                            model_dawgs_ft,
                            df_dict_eval,
                            params,
                            epoch,
                            selected_keys = [target_key])

                mlflow.pytorch.log_model(model_dawgs_ft, artifact_path=f"fit_model_{target_key}")
    os.remove(local_path)
