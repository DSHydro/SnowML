
# # pylint: disable=C0103
"""
This module contains functions for setting up and running experiments with the 
SnowML LSTM model. It includes functions to configure the MLflow tracking server, 
initialize the model, train the model, and evaluate the model with validation set.

Functions:
    set_ML_server(params)
        Configures the MLflow tracking server and sets the experiment.
    initialize_model(params)
        Initializes the SnowML model with the given parameters.
    run_expirement(train_hucs, val_hucs, params=None):
        Runs an experiment with the given training and validation HUCs and
          parameters.
"""



import torch
from torch import optim
import mlflow
import os
from pathlib import Path
from snowML.LSTM import LSTM_train as LSTM_tr
from snowML.LSTM import LSTM_model as LSTM_mod
from snowML.LSTM import set_hyperparams as sh
from snowML.LSTM import LSTM_pre_process as pp

import importlib
importlib.reload(sh)


def set_ML_server(params):
    """
    Configures the MLflow tracking server and sets the experiment.

    Returns:
        None
    """
    # Set our tracking server uri for logging
    mlflow.set_tracking_uri(params["mlflow_tracking_uri"])

    # Define the expirement
    mlflow.set_experiment(params["expirement_name"])

def initialize_model(params):
    """
    Initializes the SnowML model with the given parameters.

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


def run_expirement(train_hucs, val_hucs, params = None):
    """
    Runs an experiment by pre-training and validating a model on multiple
    HUCs (Hydrologic Unit Codes).

    Parameters:
        train_hucs (list): List of HUCs to be used for training.
        val_hucs (list): List of HUCs to be used for validation.
        params (dict, optional): Dictionary of parameters for the experiment. If
            None, default parameters are created.

        Returns:
        None
    """
    if params is None:
        params = sh.create_hyper_dict()
        sh.val_params(params)
    tr_and_val_hucs = train_hucs + val_hucs
    #print("finished finding tr and val hucs")
    df_dict, global_means, global_stds = pp.pre_process(tr_and_val_hucs, params["var_list"])
    df_dict_tr = {huc: df_dict[huc] for huc in train_hucs if huc in df_dict}
    df_dict_val = {huc: df_dict[huc] for huc in val_hucs if huc in df_dict}

    # ========== FIX #3: CREATE CHECKPOINT DIRECTORY ==========
    # Create local checkpoint directory if it doesn't exist
    # Auto-detect environment (Notebook Instance vs Studio)
    import os
    if os.path.exists("/home/ec2-user/SageMaker"):
        checkpoint_dir = Path("/home/ec2-user/SageMaker/checkpoints")
    else:
        checkpoint_dir = Path("/home/sagemaker-user/checkpoints")
    checkpoint_dir.mkdir(exist_ok=True, parents=True)

    # Generate unique experiment ID based on params
    exp_id = params.get("expirement_name", "exp").replace(" ", "_")
    # ==========================================================

    set_ML_server(params)
    model_dawgs_pretrain, optimizer_dawgs, loss_fn_dawgs = initialize_model(params)

    # Move model to GPU if available
    model_dawgs_pretrain = model_dawgs_pretrain.to(params['device'])

    with mlflow.start_run():
        # log all the params
        mlflow.log_params(params)
        # log the hucs
        mlflow.log_param("train_hucs", train_hucs)
        mlflow.log_param("val_hucs", val_hucs)
        # log the normalization values
        mlflow.log_param("global_means", global_means)
        mlflow.log_param("global_stds", global_stds)


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
            kge_tr, metric_dict_test, metric_dict_te_recur, metric_dict_train, data, y_te_true, y_te_pred, y_te_pred_recur, train_size = LSTM_tr.evaluate(
                model_dawgs_pretrain,
                df_dict_val,
                params,
                epoch)

            # ========== FIX #3: SAVE LOCAL CHECKPOINT ==========
            # Save checkpoint locally as backup (in addition to MLflow)
            checkpoint_path = checkpoint_dir / f"{exp_id}_epoch{epoch}.pth"

            # Collect metrics for checkpoint
            checkpoint_metrics = {
                'test_kge': metric_dict_test.get('test_kge', None),
                'test_mse': metric_dict_test.get('test_mse', None),
                'test_r2': metric_dict_test.get('test_r2', None),
            }

            LSTM_tr.save_checkpoint(
                model_dawgs_pretrain,
                optimizer_dawgs,
                epoch,
                checkpoint_metrics,
                str(checkpoint_path),
                params=params
            )
            print(f"💾 Checkpoint saved: {checkpoint_path.name}")
            # ===================================================

            # log the model to MLflow
            mlflow.pytorch.log_model(model_dawgs_pretrain,
                                     artifact_path=f"epoch{epoch}_model")
