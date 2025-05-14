"""Module to set the hyperparams for the LSTM model"""



def create_hyper_dict():
    """ Create dictionary of hyperparams with the given values"""
    param_dict = {
        "hidden_size": 2**6,
        "num_class": 1,
        "num_layers": 1,
        "dropout": 0.5,
        "learning_rate": 1e-3, # 3e-3, 3e-4
        "n_epochs": 10,
        "lookback": 180,
        "batch_size": 32,
        "n_steps": 1,
        "num_workers": 8,
        "var_list": ["mean_pr", "mean_tair"],
        "expirement_name": "Mar_Mixed_Loss",
        "loss_type": "custom",
        "mse_lambda_start": 1, 
        "mse_lambda_end": 0.5, 
        "train_size_dimension": "time",
        "train_size_fraction": .67, 
        "mlflow_tracking_uri": 
        "arn:aws:sagemaker:us-west-2:677276086662:mlflow-tracking-server/dawgsML",
        "recursive_predict": False, 
        "lag_days": 30,
        "lag_swe_var_idx": 3,
        "filter_dates": ["1984-10-01", "2021-09-30"]
    }
    return param_dict

def val_params(params):
    if params["recursive_predict"]:
        lag_var_name = params["var_list"][params["lag_swe_var_idx"]]
        
        if "lag" not in lag_var_name:
            raise ValueError("Double check index of lagged variable for recursive predict: 'lag' not in variable name.")
        
        if str(params["lag_days"]) not in lag_var_name:
            raise ValueError("Double check lagged days param matches variable: "
                             f"'{params['lag_days']}' not found in variable name '{lag_var_name}'.")
    
    return True

