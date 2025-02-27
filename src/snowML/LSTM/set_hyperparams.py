# Module to set the hyperparams for the LSTM model



def create_hyper_dict():
    param_dict = {
        "hidden_size": 2**6,
        "num_class": 1,
        "num_layers": 1,
        "dropout": 0.5,
        "learning_rate": 5e-4,  # 3e-3
        "n_epochs": 30,
        "lookback": 180,
        "batch_size": 64,
        "n_steps": 1,
        "num_workers": 8,
        "var_list": ["mean_pr", "mean_tair", "Mean Elevation"],
        "expirement_name": "Multi_All",
        "loss_type": "mse",
        "mse_lambda": 1, 
        "train_size_dimension": "huc"
    }
    return param_dict
