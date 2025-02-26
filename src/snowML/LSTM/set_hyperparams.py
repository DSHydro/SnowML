# Module to set the hyperparams for the LSTM model



def create_hyper_dict():
    param_dict = {
        "hidden_size": 2**6,
        "num_class": 1,
        "num_layers": 1,
        "dropout": 0.5,
        "learning_rate": 1e-3,  # 3e-3
        "n_epochs": 20,
        "lookback": 180,
        "batch_size": 64,
        "n_steps": 1,
        "num_workers": 2,
        "var_list": ["mean_pr", "mean_tair", "Mean_Elevation"],
        "expirement_name": "Prototype_Results",
        "loss_type": "mse",
        "mse_lambda": 1
    }
    return param_dict
