# Module to set the hyperparams for the LSTM model


def create_hyper_dict(param_type=None):
    # set baseline values:
    param_dict = {
        "hidden_size": 2**6,
        "num_class": 1,
        "num_layers": 1,
        "dropout": 0.3,
        "learning_rate": 1e-3,  # 3e-3
        "n_epochs": 5,
        "train_size_fraction": 0.67,
        "lookback": 180,
        "batch_size": 32,
        "n_steps": 1,
        "self_only": False,
    }

    if param_type == "Skagit_orig":
        param_dict["self_only"] = True
        param_dict["dropout"] = 0.5
        param_dict["batch_size"] = 8
        param_dict["n_epochs"] = 30

    return param_dict
