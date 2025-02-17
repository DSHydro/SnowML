
# # pylint: disable=C0103

# Script to run an expiriment
import os
import importlib
import torch
from torch import optim
from torch import nn
import mlflow
from snowML.LSTM import LSTM_pre_process as pp
from snowML.LSTM import snow_LSTM as snow
from snowML.LSTM import set_hyperparams as sh


#input_pairs = [[17020009, '12'], [17110005, '12'], [17030002, '12']]


libs_to_reload = [snow, pp, sh]
for lib in libs_to_reload:
    importlib.reload(lib)


def set_inputs(mode):
    if mode not in {"train", "eval"}:
        raise ValueError(f"Invalid mode: {mode}. Expected 'train' or 'eval'.")
    if mode == "train":
        input_pairs = [[17110005, '12']]
    else: # eval mode
        input_pairs = [[17110005, '12']]
    return input_pairs


def prep_input_data(params, mode):
    """
    Prepares input data for the experiment.


    Args:
        params (dict): A dictionary containing parameters for data preparation. 
                       Expected keys include var_list (list): List of variables to be used.

    Returns:
        df_dict: A dictionary where keys are HUCs and values are preprocessed dataframes.
    """
    if mode not in {"train", "eval"}:
        raise ValueError(f"Invalid mode: {mode}. Expected 'train' or 'eval'.")
    input_pairs = set_inputs(mode)
    hucs = pp.assemble_huc_list(input_pairs)
    df_dict = pp.pre_process(hucs, params["var_list"])
    return df_dict

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
    df_dict_train = prep_input_data(params, "train")
    df_dict_eval = prep_input_data(params, "eval")
    set_ML_server(params)
    model_dawgs_pretrain, optimizer_dawgs, loss_fn_dawgs = initialize_model(params)


    with mlflow.start_run():
        # log all the params
        mlflow.log_params(params)

        # pre-train

        pre_train_epochs = int(params['n_epochs'] * params['pre_train_fraction'])
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
                df_dict_train,
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
