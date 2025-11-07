""" Module to Initialize The LSTM Model given the input parameters """ 

import torch
from torch import optim
from snowML.LSTM import LSTM_model as LSTM_mod


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
        #print("We are using Mse loss")
    elif params["loss_type"] == "hybrid":
        loss_fn_dawgs = LSTM_mod.HybridLoss(initial_lambda=params["mse_lambda_start"],
                                            final_lambda=params["mse_lambda_end"],
                                            total_epochs=params["n_epochs"])
        #print("We are using hybrid loss")
    elif params["loss_type"] == "custom":
        loss_fn_dawgs = LSTM_mod.CustomMSEKGE_Loss(delta=params["custom delta"])
        print("We are using custome kge/mse loss with delta", params["custom delta"])
    else: # MAE loss
        loss_fn_dawgs= torch.nn.L1Loss()
        print("We are using MAE loss")

    return model_dawgs, optimizer_dawgs, loss_fn_dawgs