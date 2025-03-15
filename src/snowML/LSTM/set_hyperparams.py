"""Module to set the hyperparams for the LSTM model"""



def create_hyper_dict():
    """ Create dictionary of hyperparams with the given values"""
    param_dict = {
        "hidden_size": 2**6,
        "num_class": 1,
        "num_layers": 1,
        "dropout": 0.5,
        "learning_rate": 3e-4,  # 3e-3
        "n_epochs": 30,
        "lookback": 180,
        "batch_size": 32,
        "n_steps": 1,
        "num_workers": 8,
        "var_list": ["mean_pr", "mean_tair", "mean_hum", "Mean Elevation"],
        "expirement_name": "Multi_All-2",
        "loss_type": "mse",
        "mse_lambda": 1, 
        "train_size_dimension": "huc",
        "train_size_fraction": 1, 
        "mlflow_tracking_uri": 
        "arn:aws:sagemaker:us-west-2:677276086662:mlflow-tracking-server/dawgsML"
    }
    return param_dict
