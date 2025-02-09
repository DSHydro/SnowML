
# # pylint: disable=C0103

# Script to run an expiriment
import sys
import os
import time
from torch import optim
from torch import nn
import LSTM_pre_process as pp
import snow_LSTM as snow
import set_hyperparams as sh
from smdebug.pytorch import Hook

import importlib
importlib.reload(snow)  # TO DO - Remove once stable

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import data_utils as du




def set_inputs():
    input_pairs = [[17110005, '10'], [17020009, '12']]
    #input_pairs = [[17110005, '10']]
    var_list = ["mean_pr", "mean_tair"]
    return input_pairs, var_list

# prepare the data for LSTM
def prep():
    input_pairs, var_list = set_inputs()
    hucs = pp.assemble_huc_list(input_pairs)
    df_dict = pp.pre_process(hucs, var_list)
    return df_dict


def run_expirement():
    #set hyperparams
    params = sh.create_hyper_dict()
    print(params)


    # get_data
    df_dict = prep()
    _ , var_list = set_inputs()
    target_key = "1711000510"  # TO DO make dynamic

    # initalize model
    input_size=len(var_list)
    model_dawgs = snow.SnowModel(
        input_size,
        params['hidden_size'],
        params['num_class'],
        params['num_layers'],
        params['dropout']
    )
    optimizer_dawgs = optim.Adam(model_dawgs.parameters())
    loss_fn_dawgs = nn.MSELoss()
    # Create the hook from your JSON config or default configuration
    #hook = Hook.create_from_json_file()  # Optional: You can specify your own config file
    # Register the model with the hook
    #hook.register_module(model_dawgs)

    # train
    start_time = time.time()
    snow.train_model(
        model_dawgs, 
        optimizer_dawgs, 
        loss_fn_dawgs, 
        df_dict, 
        target_key, 
        var_list, 
        params)
    

    # test  - use selected huc for testing
    data = df_dict[target_key]
    print(f"Huc selected for model prediction and evlauatoin is{target_key}")
    train_main, test_main, train_size_main, _ = pp.train_test_split(data, params['train_size_fraction'])
    X_train, y_train = pp.create_tensor(train_main, params['lookback'], var_list)
    X_test, y_test = pp.create_tensor(test_main, params['lookback'], var_list)

    
    #if step_val == n_steps:
    snow.predict(data, model_dawgs,  X_train, X_test, train_size_main, var_list, int(target_key), params)
    step_val = 1 # TO DO
    dawgs_metrics = snow.evaluate_metrics(model_dawgs, X_train, y_train, X_test, y_test)
    print(dawgs_metrics)

    du.elapsed(start_time)


# Start an MLflow run and log the parameters
#with mlflow.start_run():
    #mlflow.log_params(params)