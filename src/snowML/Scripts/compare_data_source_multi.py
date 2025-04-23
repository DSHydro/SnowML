""" Module to compare model performance on diff Skagit data sources.  
Uses Multi-Huc Module """


# pylint: disable=C0103

import re
import mlflow
import mlflow.pytorch
import pandas as pd
from snowML.LSTM import LSTM_evaluate as evaluate
from snowML.LSTM import LSTM_pre_process as pp
from snowML.LSTM import LSTM_plot3 as plot3
from snowML.LSTM import LSTM_metrics as met

# Define some constants
TRACKING_URI = "arn:aws:sagemaker:us-west-2:677276086662:mlflow-tracking-server/dawgsML"
RUN_ID = "a6c611d4c4cf410e9666796e3a8892b7" #deboniar dove
MODEL_URI = "s3://sues-test/298/a6c611d4c4cf410e9666796e3a8892b7/artifacts/epoch8_model"
FILTER_START = "1984-10-01"
FILTER_END = "2021-09-30"

def get_Skagit_list_10():
    return [1711000504, 1711000505, 1711000506, 1711000507, 1711000508, 1711000509, 1711000511]

def extract_globals(s):
    # Normalize key formatting and extract key-value pairs using regex
    pattern = re.compile(r'(.+?)\s+([\d.]+)')
    matches = pattern.findall(s)

    global_vals = {}
    for key, value in matches:
        global_vals[key] = float(value)
    return global_vals

def get_params(tracking_uri, run_id):
    params = evaluate.get_params(tracking_uri, run_id)
    global_means = extract_globals(params['global_means'])
    global_stds = extract_globals(params['global_stds'])
    params["filter_dates"] = [FILTER_START, FILTER_END] # TO DO - GET DYNAMICALLY
    return params, global_means, global_stds

def eval_UA(model_dawgs, df_dict_test, huc, params):
    metric_dict_test, metric_dict_te_recur, _, _, y_te_pred_UA, _, y_te_true_UA, _, _ = evaluate.eval_from_saved_model(
            model_dawgs, df_dict_test, huc, params)
    for m_dict in [metric_dict_test, metric_dict_te_recur]:
        met.log_print_metrics(m_dict, 0)

    # plot
    plot_dict_true = plot3.assemble_plot_dict(y_te_true_UA, "blue",
        'SWE Estimates UA Data (Physics Based Model)')
    plot_dict_te = plot3.assemble_plot_dict(y_te_pred_UA, "green",
        'SWE Estimates Prediction Using UA Data') 
    ttl = f"SWE_Predictions_for_huc_{huc} UA Data"
    df_UA = df_dict_test[huc]
    x_axis_vals = df_UA.index
    plot3.plot3(x_axis_vals, [plot_dict_true, plot_dict_te], ttl, metrics_dict = metric_dict_test)
    return plot_dict_true, plot_dict_te

def eval_snowTEL (huc, model_dawgs, params, df_UA_norm):
    f = f"notebooks/Ex1_MoreData/orig_data/wus-sr-skagit-{huc}-mean-swe.csv"
    df_snowTEL = pd.read_csv(f)
    df_UA_copy = df_UA_norm.copy()
    df_UA_copy["mean_swe"] = df_snowTEL["mean"].values  # replace mean_swe column
    df_dict_snowTEL = {}
    df_dict_snowTEL[huc] = df_UA_copy  # Store updated df
    metric_dict_test, metric_dict_te_recur, _, y_tr_pred_snowTEL, y_te_pred_snowTEL, _, y_te_true_snowTEL, _, train_size = evaluate.eval_from_saved_model(
            model_dawgs, df_dict_snowTEL, huc, params)
    for m_dict in [metric_dict_test, metric_dict_te_recur]:
        met.log_print_metrics(m_dict, 0)

    # plot
    plot_dict_true = plot3.assemble_plot_dict(y_te_true_snowTEL, "black",
        'SWE - Orig Skgit Data')
    plot_dict_te = plot3.assemble_plot_dict(y_te_pred_snowTEL, "red",
        'SWE Estimates Using Orig Skagit Data') 
    x_axis_vals = df_UA_norm.index
    ttl = f"SWE Estimates for {huc} Using Orig Skagit Data"
    plot3.plot3(x_axis_vals, [plot_dict_true, plot_dict_te], ttl, metrics_dict = metric_dict_test)
    return plot_dict_true, plot_dict_te

def eval_all(hucs, tracking_uri=TRACKING_URI, run_id = RUN_ID, model_uri = MODEL_URI):
    params, global_means, global_stds = get_params(tracking_uri, run_id)
    mlflow.set_experiment("Skagit_DataSource_Compare")
    model_dawgs = evaluate.load_model(model_uri)
    with mlflow.start_run():
        mlflow.log_params(params)
        mlflow.log_param("hucs", hucs)
        mlflow.log_param("model_uri", model_uri)
        mlflow.log_param("hucs", hucs)
        for huc in hucs:
            y_dict_list = []
            # load_normalized_UA dataset
            df_UA = pp.load_df(huc, params['var_list'], filter_dates = params["filter_dates"])
            df_UA_norm = pp.z_score_normalize(df_UA, global_means, global_stds)

            # evaluate using UA data
            df_dict_test = {huc: df_UA_norm}
            plot_dict_true_UA, plot_dict_te_UA  = eval_UA(model_dawgs, df_dict_test, huc, params)
            y_dict_list.append(plot_dict_true_UA)
            y_dict_list.append(plot_dict_te_UA)

            # evaluate using snowTEL data
            plot_dict_true, plot_dict_te = eval_snowTEL (huc, model_dawgs, params, df_UA_norm)
            y_dict_list.append(plot_dict_true)
            y_dict_list.append(plot_dict_te)

            # plot all
            ttl = f"SWE_Predictions_for_huc_{huc} multiuple data sources"
            x_axis_vals = df_UA_norm.index
            plot3.plot3(huc, x_axis_vals, y_dict_list, ttl)

            # plot just green
            #ttl = f"SWE_Predictions_for_huc_{huc} just green"
            #plot3(huc, x_axis_vals, [plot_dict_te_UA, plot_dict_true_UA, plot_dict_true], ttl)
   