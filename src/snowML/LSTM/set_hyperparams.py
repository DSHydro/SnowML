# Module to set the hyperparams for the LSTM model


def create_hyper_dict():
    param_dict = {
        "hidden_size": 2**6,
        "num_class": 1,
        "num_layers": 1,
        "dropout": 0.2,
        "learning_rate": 1e-3,  # 3e-3
        "n_epochs": 10,
        "pre_train_fraction" : 0,
        "train_size_fraction": 0.67,
        "lookback": 180,
        "batch_size": 16,
        "n_steps": 1,
        "num_workers": 8,
        "var_list": ["mean_pr", "mean_tair"],
        #"var_list": ["mean_pr", "mean_tair", "Maritime", "Ephemeral", "Montane Forest", "Ice"], 
        "expirement_name": "TestingSagemaker"
    }
    return param_dict
