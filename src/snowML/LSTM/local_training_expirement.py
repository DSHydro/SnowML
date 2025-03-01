
""" Script to run an expiriment with local training on target huc(s) only"""

# # pylint: disable=C0103


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


def run_local_exp(hucs, train_size_frac, params = None):
    if params is None:
        params = sh.create_hyper_dict()
    df_dict = pp.pre_process(hucs, params["var_list"])

    set_ML_server(params)
    model_dawgs, optimizer_dawgs, loss_fn_dawgs = initialize_model(params)

    with mlflow.start_run():
        # log all the params
        mlflow.log_params(params)
        # log the hucs & train size fraction
        mlflow.log_param("train_size_fraction", train_size_frac)
        mlflow.log_param("hucs", hucs)

        for huc in df_dict.keys():
            print(f"Training on HUC {huc}")
            df = df_dict[huc]
            df_dict_small = {huc: df}
            df_train, df_test, _, _ = pp.train_test_split_time(df, train_size_frac)

            for epoch in range(params["n_epochs"]):
                print(f"Epoch {epoch}")

                # for local training, call fine_tune instead of pre_train
                LSTM_tr.fine_tune(
                model_dawgs,
                optimizer_dawgs,
                loss_fn_dawgs,
                df_train,
                params,
                epoch
                )

                # validate
                LSTM_tr.evaluate(
                    model_dawgs,
                    df_dict_small,
                    params,
                    epoch)

            # log the model
            mlflow.pytorch.log_model(model_dawgs,
                                     artifact_path=f"model_{huc}")
