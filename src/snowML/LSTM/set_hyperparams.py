"""Module to set the hyperparams for the LSTM model"""



def create_hyper_dict():
    """ Create dictionary of hyperparams with the given values"""
    param_dict = {
        "hidden_size": 2**6,
        "num_class": 1,
        "num_layers": 1,
        "dropout": 0.5,
        "learning_rate":  1e-3, #3e-4, # 3e-3
        "n_epochs": 10,
        "lookback": 180,
        "batch_size": 64,
        "n_steps": 1,
        "num_workers": 8,
        "var_list": ["mean_pr", "mean_tair"],
        "expirement_name": "Skagit_DataSource_Compare",
        "loss_type": "mse",
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
        if not lag_var_name.startswith("lag"):
            print("Double check index of lagged variable for recursive predict")
            return False
    return True
