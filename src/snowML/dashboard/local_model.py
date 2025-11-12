""" Module to Locally Train on An Individual Huc """

import streamlit as st
import pandas as pd
from snowML.LSTM import LSTM_initialize as init
from snowML.LSTM import LSTM_train as LSTM_tr
from snowML.LSTM import LSTM_pre_process as pp
from snowML.LSTM import LSTM_plot3 as plot3



def set_params():
    """ Create dictionary of hyperparams with the given values"""
    param_dict = {
        "hidden_size": 2**6,
        "num_class": 1,
        "num_layers": 1,
        "dropout": 0.5,
        "learning_rate": 1e-3, # 3e-3, 3e-4
        "n_epochs": 5,
        "lookback": 180,
        "batch_size": 32,
        "n_steps": 1,
        "num_workers": 8,
        "var_list": ["mean_pr", "mean_tair"],
        "expirement_name": "Multi_Run",
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
        "filter_dates": ["1984-10-01", "2021-09-30"], 
        "custom delta": .04, 
        "UCLA": False,
        "Stop_Loss": False,
        "KGE_target": .9, 
        "MLFLOW_ON": False
    }
    return param_dict


def combine_dict(params, metric_dict_test, metric_dict_train, metric_dict_te_recur):
    """
    Combine metric dictionaries depending on whether recursive prediction is enabled.

    Args:
        params (dict): Parameter dictionary with 'recursive_predict' key.
        metric_dict_test (dict): Metrics for the test set.
        metric_dict_train (dict): Metrics for the training set.
        metric_dict_te_recur (dict): Metrics for recursive prediction (if any).

    Returns:
        dict: Combined metrics dictionary.
    """
    if params.get("recursive_predict", False):
        # Combine test, train, and recursive metrics
        combined_dict = {**metric_dict_test, **metric_dict_train, **metric_dict_te_recur}
    else:
        # Combine only test and train
        combined_dict = {**metric_dict_test, **metric_dict_train}

    return combined_dict



def plot_local(huc12, params, y_te_true, y_te_pred, y_te_pred_recur, data, tr_size, combined_dict, epoch):
    if params["UCLA"]:
        plot_dict_true = plot3.assemble_plot_dict(y_te_true, "blue",
            'SWE Estimates UCLA Data')
    else:
        plot_dict_true = plot3.assemble_plot_dict(y_te_true, "blue",
            'SWE Estimates UA Data (Physics Based Model)')

    plot_dict_te = plot3.assemble_plot_dict(y_te_pred, "green",
            'SWE Estimates Prediction') 

    if params["recursive_predict"]:
        plot_dict_te_recur = plot3.assemble_plot_dict(y_te_pred_recur, "black",
            'SWE Estimates Recursive Prediction')
    else:
        plot_dict_te_recur = None

    y_dict_list = [plot_dict_true, plot_dict_te, plot_dict_te_recur ]
    ttl = f"SWE_Actual_vs_Predicted_for_huc_{huc12}_epoch{epoch}"
    x_axis_vals = data.index[tr_size:]
    fig = plot3.plot3b(x_axis_vals, y_dict_list, ttl, metrics_dict = combined_dict)
    fig_path = f"{ttl}.png"
    fig.savefig(fig_path, bbox_inches="tight", dpi=300)
    return fig


def local_train (huc12):
    params = set_params()
    model_dawgs, optimizer_dawgs, loss_fn_dawgs = init.initialize_model(params)
    df_dict = pp.pre_process_separate([huc12], params["var_list"],
                                      UCLA = params["UCLA"], filter_dates=params["filter_dates"])
    df_train = df_dict[huc12]
    stop = False

    status_placeholder = st.empty()
    status_placeholder.write("Commencing training, epoch 0")

    results_df = pd.DataFrame()

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
        kge_tr, metric_dict_test, metric_dict_te_recur, metric_dict_train, data, y_te_true, y_te_pred, y_te_pred_recur, tr_size = LSTM_tr.evaluate(
            model_dawgs,
            df_dict,
            params,
            epoch)

        if (kge_tr >= params["KGE_target"] and params["Stop_Loss"]):
            stop = True
            if stop:
                print(f"Ending training after epoch {epoch}, training target reached")
                st.write(f"Ending training after epoch {epoch}, training target reached")
            break

        combined_dict = combine_dict(params, metric_dict_test, metric_dict_train, metric_dict_te_recur)
        epoch_df = pd.DataFrame([{**{"Epoch": epoch}, **combined_dict}]).set_index("Epoch")
        results_df = pd.concat([results_df, epoch_df])
        fig = plot_local(huc12, params, y_te_true, y_te_pred, y_te_pred_recur,
                         data, tr_size, combined_dict, epoch)
        status_placeholder.empty()
        if epoch != (params["n_epochs"] -1):
            with status_placeholder.container():
                st.markdown("### Results So Far:")
                st.dataframe(results_df, use_container_width=True)
                st.pyplot(fig)
                st.markdown(f"**Commencing Training Epoch {epoch+1}...**")

    return results_df, fig
