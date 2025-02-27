
# # pylint: disable=C0103

# Script to run an expiriment on multiple HUCs
import torch
import importlib
from torch import optim
import mlflow
from snowML.LSTM import LSTM_train as LSTM_tr
from snowML.LSTM import LSTM_model as LSTM_mod
from snowML.LSTM import set_hyperparams as sh
from snowML.LSTM import LSTM_pre_process as pp

importlib.reload(LSTM_tr)
importlib.reload(sh)


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
    model_dawgs = LSTM_mod.SnowModel(
        input_size,
        params['hidden_size'],
        params['num_class'],
        params['num_layers'],
        params['dropout']
    )
    optimizer_dawgs = optim.Adam(model_dawgs.parameters())

    # Set the loss function based on the loss_type parameter
    if params["loss_type"] == "mse":
        loss_fn_dawgs = torch.nn.MSELoss()
    else: #params["loss_type"] == "hybrid"
        loss_fn_dawgs = LSTM_mod.HybridLoss(initial_lambda=params["mse_lambda"],
                                            final_lambda=params["mse_lambda"],
                                            total_epochs=params["n_epochs"])

    return model_dawgs, optimizer_dawgs, loss_fn_dawgs


def run_expirement(train_hucs, val_hucs, test_hucs, params = None):
    if params is None:
        params = sh.create_hyper_dict()
    df_dict_tr = pp.pre_process(train_hucs, params["var_list"])
    df_dict_val = pp.pre_process(val_hucs, params["var_list"])

    set_ML_server(params)
    model_dawgs_pretrain, optimizer_dawgs, loss_fn_dawgs = initialize_model(params)

    with mlflow.start_run():
        # log all the params
        mlflow.log_params(params)
        # log the hucs
        mlflow.log_param("train_hucs", train_hucs)
        mlflow.log_param("val_hucs", val_hucs)
        mlflow.log_param("val_hucs", val_hucs)


        for epoch in range(params["n_epochs"]):
            print(f"Epoch {epoch}: Pre-training on multiple HUCs")

            # pre-train
            LSTM_tr.pre_train(
                model_dawgs_pretrain,
                optimizer_dawgs,
                loss_fn_dawgs,
                df_dict_tr,
                params,
                epoch
                )

            # validate
            LSTM_tr.evaluate(
                model_dawgs_pretrain,
                df_dict_val,
                params,
                epoch)

            # log the model
            mlflow.pytorch.log_model(model_dawgs_pretrain,
                                     artifact_path=f"epoch{epoch}_model")
