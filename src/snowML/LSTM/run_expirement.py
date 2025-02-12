
# # pylint: disable=C0103

# Script to run an expiriment
import time
import os
import importlib
import pandas as pd
from torch import optim
from torch import nn
import mlflow
from snowML.LSTM import LSTM_pre_process as pp
from snowML.LSTM import snow_LSTM as snow
from snowML.LSTM import set_hyperparams as sh
from snowML import data_utils as du


libs_to_reload = [snow, pp, sh]
for lib in libs_to_reload:
    importlib.reload(lib)


def set_inputs():
    input_pairs = [[17020009, '12'], [17110005, '12'], [17030002, '12']]
    #input_pairs = [[17110005, '12']]
    #input_pairs = [[17020009, '12']]
    return input_pairs


def prep_input_data(params):
    """
    Prepares input data for the experiment.


    Args:
        params (dict): A dictionary containing parameters for data preparation. 
                       Expected keys include var_list (list): List of variables to be used.

    Returns:
        df_dict: A dictionary where keys are HUCs and values are preprocessed dataframes.
    """
    input_pairs = set_inputs()
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
    mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")

    # Create a new MLflow Experiment called "LSTM"
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

def run_expirement():
    params = sh.create_hyper_dict()
    df_dict = prep_input_data(params)
    set_ML_server(params)
    model_dawgs, optimizer_dawgs, loss_fn_dawgs = initialize_model(params)

    with mlflow.start_run():
        # log all the params
        mlflow.log_params(params)
        
        for step_val in range(params["n_steps"]):

            # pre_train
            start_time = time.time()
            snow.pre_train(
                model_dawgs,
                optimizer_dawgs,
                loss_fn_dawgs,
                df_dict,
                params)


            train_mse_list = []
            test_mse_list = []
            train_kg_list = []
            test_kg_list = []

            # for each target key, fine tune (if applicable) and evaluate
            for target_key, data in df_dict.items():

                if params["pre_train_fraction"] < 1:
                    print(f"Fine-tuning on {target_key}")
                    snow.fine_tune(
                        model_dawgs,
                        optimizer_dawgs,
                        loss_fn_dawgs,
                        df_dict,
                        target_key,
                        params)

                print(f"Performing prediction and evlauation on {target_key}")
                train_main, test_main, train_size_main, _ = pp.train_test_split(data, params['train_size_fraction'])
                X_train, y_train = pp.create_tensor(train_main,
                                                    params['lookback'],
                                                    params['var_list'])
                X_test, y_test = pp.create_tensor(test_main,
                                                  params['lookback'],
                                                  params['var_list'])
                
                
                metrics = snow.evaluate_metrics(model_dawgs,
                                      X_train,
                                      y_train,
                                      X_test,
                                      y_test,
                                      target_key,
                                      step_val)
                train_mse_list.append(metrics[0])
                test_mse_list.append(metrics[1])
                train_kg_list.append(metrics[2])
                test_kg_list.append(metrics[3])

                if step_val == params["n_steps"]-1:
                    snow.predict(data,
                                 model_dawgs,
                                 X_train,
                                 X_test,
                                 train_size_main,
                                 int(target_key),
                                 params)

                du.elapsed(start_time)


        # gather and save results
        df = pd.DataFrame({
            "train_mse": train_mse_list,
            "test_mse": test_mse_list,
            "train_kg": train_kg_list,
            "test_kg": test_kg_list
            }, index=df_dict.keys())

        # Log the dataframe in mlflow
        csv_path = "results.csv"
        df.to_csv(csv_path, index=True)
        mlflow.log_artifact(csv_path)
        os.remove(csv_path)
