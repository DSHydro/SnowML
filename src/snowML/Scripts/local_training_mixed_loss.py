
""" Script to run an expiriment with local training on target huc(s) only"""

# # pylint: disable=C0103



import time
import torch
from torch import optim
import mlflow
#import cloudpickle
from snowML.LSTM import LSTM_train as LSTM_tr
from snowML.LSTM import LSTM_model as LSTM_mod
from snowML.LSTM import set_hyperparams as sh
from snowML.LSTM import LSTM_pre_process as pp
from snowML.datapipe.utils import data_utils as du


def set_ML_server(params):
    """
    Configures the MLflow tracking server and sets the experiment.

    Returns:
        None
    """
    # Set our tracking server uri for logging
    tracking_uri = params["mlflow_tracking_uri"]
    #tracking_uri = "arn:aws:sagemaker:us-west-2:677276086662:mlflow-tracking-server/dawgsML"
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
        print("We are using Mse loss")
    elif params["loss_type"] == "hybrid":
        loss_fn_dawgs = LSTM_mod.HybridLoss(initial_lambda=params["mse_lambda_start"],
                                            final_lambda=params["mse_lambda_end"],
                                            total_epochs=params["n_epochs"])
        print("We are using hybrid loss")
    else: # MAE loss
        loss_fn_dawgs= torch.nn.L1Loss()
        print("We are using MAE loss")

    return model_dawgs, optimizer_dawgs, loss_fn_dawgs


def run_local_exp(hucs, params = None):
    if params is None:
        params = sh.create_hyper_dict()
        sh.val_params(params)

    # normalize each df separately when local training
    df_dict = pp.pre_process_separate(hucs, params["var_list"], filter_dates=params["filter_dates"])
    #print("df_dict is", df_dict)
    train_size_frac = params["train_size_fraction"]

    set_ML_server(params)
    model_dawgs, optimizer_dawgs, loss_fn_dawgs = initialize_model(params)

    with mlflow.start_run():
        # log all the params
        mlflow.log_params(params)
        # log the hucs & train size fraction
        mlflow.log_param("hucs", hucs)

        for huc in df_dict.keys():
            time_start = time.time()
            print(f"Training on HUC {huc}")
            df = df_dict[huc]
            df_dict_small = {huc: df}
            df_train, _, _, _ = pp.train_test_split_time(df, train_size_frac)

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
                if True: 
                #if (epoch % 5 == 0) or (epoch == params["n_epochs"] - 1):
                    LSTM_tr.evaluate(
                        model_dawgs,
                        df_dict_small,
                        params,
                        epoch)

            # log the model
            #mlflow.pytorch.log_model(model_dawgs, artifact_path=f"model_{huc}", pickle_module=cloudpickle)
            mlflow.pytorch.log_model(model_dawgs, artifact_path=f"model_{huc}")
            du.elapsed(time_start)
