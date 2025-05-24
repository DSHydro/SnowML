
""" Script to run an expiriment with local training on target huc(s) only
using mixed loss function as specified, and early stop if kge target reached """

# # pylint: disable=C0103



#import time
import importlib
import torch
from torch import optim
import mlflow
from snowML.LSTM import LSTM_train as LSTM_tr
from snowML.LSTM import LSTM_model as LSTM_mod
from snowML.LSTM import set_hyperparams as sh
from snowML.LSTM import LSTM_pre_process as pp
from snowML.LSTM import LSTM_plot3 as plot3



importlib.reload(pp)
importlib.reload(sh)
importlib.reload(LSTM_tr)

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


def run_local_exp(hucs, params = None):
    if params is None:
        params = sh.create_hyper_dict()
        sh.val_params(params)

    # normalize each df separately when local training
    df_dict = pp.pre_process_separate(hucs, params["var_list"], UCLA = params["UCLA"], filter_dates=params["filter_dates"])
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
            #time_start = time.time()
            print(f"Training on HUC {huc}")
            df = df_dict[huc]
            df_dict_small = {huc: df}
            df_train, _, _, _ = pp.train_test_split_time(df, train_size_frac)
            stop = False

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

                # evaluate and inspect train_kge
                kge_tr, metric_dict_test, metric_dict_te_recur, metric_dict_train, data, y_te_true, y_te_pred, y_te_pred_recur, tr_size = LSTM_tr.evaluate(
                    model_dawgs,
                    df_dict_small,
                    params,
                    epoch)

                if kge_tr >= params["KGE_target"]: 
                    stop = True
                    if stop: 
                        print(f"Ending training after epoch {epoch}, training target reached")
                    break
        
            # store plots for final epooch
            if params["recursive_predict"]:
                combined_dict = {**metric_dict_test, **metric_dict_te_recur}
            else:
                print("Plotting . . . ")
                combined_dict = metric_dict_test
                #met.log_print_metrics(combined_dict, selected_key, epoch)
                if params["UCLA"]:
                    plot_dict_true = plot3.assemble_plot_dict(y_te_true, "blue",
                        'SWE Estimates UCLA Data')
                else:
                    plot_dict_true = plot3.assemble_plot_dict(y_te_true, "blue",
                        'SWE Estimates UA Data (Physics Based Model)')
                plot_dict_te = plot3.assemble_plot_dict(y_te_pred, "green",
                    'SWE Estimates Prediction') 
                if params["recursive_predict"]:
                    plot_dict_te_recur = plot3.assemble_plot_dict(y_te_pred_recur, "black",
                        'SWE Estimates Recursive Prediction')
                else:
                    plot_dict_te_recur = None
                y_dict_list = [plot_dict_true, plot_dict_te, plot_dict_te_recur ]
                ttl = f"SWE_Actual_vs_Predicted_for_huc_{huc}"
                x_axis_vals = data.index[tr_size:]
                plot3.plot3(x_axis_vals, y_dict_list, ttl, metrics_dict = combined_dict)


        # log the model
        #mlflow.pytorch.log_model(model_dawgs, artifact_path=f"model_{huc}", pickle_module=cloudpickle)
        mlflow.pytorch.log_model(model_dawgs, artifact_path=f"model_{huc}")
        #du.elapsed(time_start)
