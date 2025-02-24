# Module to set the hyperparams for the LSTM model

#INPUT_PAIRS = [[17110006, '12'], [17110005, '12'], [17110009, '12']]


def create_hyper_dict():
    param_dict = {
        "hidden_size": 2**6,
        "num_class": 1,
        "num_layers": 1,
        "dropout": 0.5,
        "learning_rate": 1e-3,  # 3e-3
        "n_epochs": 20,
        "pre_train_fraction" : 0,
        "train_size_fraction": .67,
        "train_size_dimension": "time",
        "lookback": 180,
        "batch_size": 64,
        "n_steps": 1,
        "num_workers": 2,
        "var_list": ["mean_pr", "mean_tair"],
        #"var_list": ["mean_pr", "mean_tair", "Maritime", "Ephemeral", "Montane Forest", "Ice"],
        "expirement_name": "Prototype_Results",
        #"input_pairs": [[17110005, '12'], [17110006, '12'], [17110009, '12'], [17020009, '12']],
        "input_pairs": [[17110005, '10']],
        "exclude_ephem": False,
        "mse_lambda": 1
    }
    return param_dict
