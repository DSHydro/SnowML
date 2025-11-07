""" Module to ReRun Training on the Same Huc Multiple Times to Compare Results """ 


from snowML.LSTM import LSTM_train as LSTM_tr
from snowML.LSTM import LSTM_initialize as LSTM_init
from snowML.LSTM import set_hyperparams as sh
from snowML.LSTM import LSTM_pre_process as pp



def pre_process(huc, params):
    # normalize the data and create train/test split
    df_dict = pp.pre_process_separate([huc], params["var_list"], UCLA = params["UCLA"], filter_dates=params["filter_dates"])
    train_size_frac = params["train_size_fraction"]
    df = df_dict[huc]
    df_train, _, _, _ = pp.train_test_split_time(df, train_size_frac)
    return df_dict, df_train


def run_multi_exp(huc, params = None):  # Takes only a single huc
    if params is None:
        params = sh.create_hyper_dict()
        sh.val_params(params)

    # pre-process data
    df_dict, df_train = pre_process(huc, params)

    # initialize_model
    model_dawgs, optimizer_dawgs, loss_fn_dawgs = LSTM_init.initialize_model(params)
    stop = False

    for epoch in range(params["n_epochs"]):
        print(f"Epoch {epoch}")

        # for local training, call fine_tune instead of pre_train
        LSTM_tr.fine_tune(
            model_dawgs,
            optimizer_dawgs,
            loss_fn_dawgs,
            df_train,
            params,
            epoch
        )

        # evaluate and inspect train_kge
        kge_tr, metric_dict_test, _, _, _, _, _, _, _ = LSTM_tr.evaluate(
            model_dawgs,
            df_dict,
            params,
            epoch)

        if (kge_tr >= params["KGE_target"] and params["Stop_Loss"]):
            stop = True
        if stop:
            print(f"Ending training after epoch {epoch}, training target reached")
            break

    return  kge_tr, metric_dict_test
