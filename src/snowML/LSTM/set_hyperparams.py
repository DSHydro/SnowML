# Module to set the hyperparams for the LSTM model 


def create_hyper_dict(type=None):
    # set baseline values: 
    dict = {
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
    
    if type == "Skagit_orig": 
        dict["self_only"] = True
        dict["dropout"] = 0.5
        dict["batch_size"] = 8
        dict["n_epochs"] = 30

    
    return dict

