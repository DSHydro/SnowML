# Module to set the hyperparams for the LSTM model

#INPUT_PAIRS = [[17110006, '12'], [17110005, '12'], [17110009, '12']]


def create_hyper_dict():
    param_dict = {
        "hidden_size": 2**6,
        "num_class": 1,
        "num_layers": 1,
        "dropout": 0.5,
        "learning_rate": 5e-4,  # 3e-3
        "n_epochs": 10,
        "pre_train_fraction" : 1,
        "train_size_fraction": .8,
        "train_size_dimension": "huc",
        "lookback": 180,
        "batch_size": 32,
        "n_steps": 1,
        "num_workers": 8,
        "var_list": ["mean_pr", "mean_tair"],
        #"var_list": ["mean_pr", "mean_tair", "Maritime", "Ephemeral", "Montane Forest", "Ice"], 
        "expirement_name": "Maritime_Multi", 
        "input_pairs": [[17110006, '12'], [17110005, '12'], [17110009, '12']]
    }
    return param_dict
