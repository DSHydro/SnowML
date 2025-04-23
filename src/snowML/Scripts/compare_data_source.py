""" Module to compare model performance on diff Skagit data sources.  
Assumes Single HUC (LOCAL TRAINING) """


# pylint: disable=C0103

import mlflow
import mlflow.pytorch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from snowML.LSTM import LSTM_evaluate as evaluate
from snowML.LSTM import LSTM_pre_process as pp
from snowML.LSTM import LSTM_plot3 as plot3
from snowML.Scripts.load_hucs import select_hucs_local_training as sh


# Define some constants
TRACKING_URI = "arn:aws:sagemaker:us-west-2:677276086662:mlflow-tracking-server/dawgsML"
RUN_ID = "76af96db1ba04141859547bdb6b59320" # intrigued mule
MODEL_PREFIX = "s3://sues-test/463/76af96db1ba04141859547bdb6b59320/artifacts/model"
FILTER_START = "1984-10-01"
FILTER_END = "2021-09-30"


def get_Skagit_list_12():
    Skagit_huc_list = sh.assemble_huc_list(input_pairs=[[17110005, '12']])
    return Skagit_huc_list

def get_Skagit_list_10():
    return [1711000504, 1711000505, 1711000506, 1711000507, 1711000508, 1711000509, 1711000511]


def eval_UA (huc, model_dawgs, params):
    df_dict_test =  pp.pre_process_separate([huc], params["var_list"],
        filter_dates = params["filter_dates"])
    df_UA = df_dict_test[huc]
    metric_dict, data, y_tr_pred_UA, y_te_pred_UA, _, _, train_size = evaluate.eval_from_saved_model(
            model_dawgs, df_dict_test, huc, params)
    for met_nm, metric in metric_dict.items():
        mlflow.log_metric(f"{met_nm}_{str(huc)}_UA", metric)
        print(f"{met_nm}: {metric}")
    return df_UA, y_te_pred_UA

def eval_snowTEL (huc, model_dawgs, params, df_UA):
    f = f"notebooks/Ex1_MoreData/orig_data/wus-sr-skagit-{huc}-mean-swe.csv"
    df_snowTEL = pd.read_csv(f)
    df_UA_copy = df_UA.copy()
    df_UA_copy["mean_swe"] = df_snowTEL["mean"].values  # replae mean_swe column
    df_dict_snowTEL = {}
    df_dict_snowTEL[huc] = df_UA_copy  # Store updated df
    metric_dict, data, y_tr_pred_snowTEL, y_te_pred_snowTEL, _, y_te_true_snowTEL, train_size = evaluate.eval_from_saved_model(
            model_dawgs, df_dict_snowTEL, huc, params)
    # TO DO - UPDATE LABELS IN PLOT 
    plot2.plot(data, y_tr_pred_snowTEL, y_te_pred_snowTEL, train_size, huc, params, metrics_dict = metric_dict)
    for met_nm, metric in metric_dict.items():
        mlflow.log_metric(f"{met_nm}_{str(huc)}_snowTel", metric)
        print(f"{met_nm}: {metric}")
    return y_te_pred_snowTEL, y_te_true_snowTEL, train_size

def plot_test_results(df_UA, y_te_pred_UA, y_te_pred_snowTEL,
        y_te_true_snowTEL, train_size, params, huc_id):
    test_plot_UA = np.full_like(df_UA['mean_swe'].values, np.nan, dtype=float)
    test_plot_UA[train_size + params["lookback"] : df_UA.shape[0]] = y_te_pred_UA.flatten()
    test_plot_snowTEL = np.full_like(df_UA['mean_swe'].values, np.nan, dtype=float)
    test_plot_snowTEL[train_size + params["lookback"] : df_UA.shape[0]] = y_te_pred_snowTEL.flatten()
    snowTEL_true = np.full_like(df_UA['mean_swe'].values, np.nan, dtype=float)
    snowTEL_true[train_size + params["lookback"] : df_UA.shape[0]] = y_te_true_snowTEL.flatten()



    plt.figure(figsize=(12, 6))
    plt.plot(
        df_UA.index[train_size:],
        df_UA['mean_swe'][train_size:],
        c='blue',
        label='SWE Estimates UA Data (Physics Based Model)')
    plt.plot(
        df_UA.index[train_size + params["lookback"]:],
        test_plot_UA[train_size + params["lookback"]:],
        c='green',
        label='LSTM Predictions Forecasting Phase UA Data'
    )
    plt.plot(
        df_UA.index[train_size + params["lookback"]:],
        snowTEL_true[train_size + params["lookback"]:],
        c='black',
        label='Skagit Orig Data True'
    )
    plt.plot(
        df_UA.index[train_size + params["lookback"]:],
        test_plot_snowTEL[train_size + params["lookback"]:],
        c='red',
        label='LSTM Predictions Forecasting Phase Skagit Orig Data'
    )

    plt.legend()
    plt.xlabel('Date')
    plt.ylabel('SWE')
    ttle = f"SWE_Predictions_for_huc_{huc_id} multiuple data sources"
    plt.title(ttle)
    mlflow.log_figure(plt.gcf(), ttle + ".png")

def eval_all(hucs, tracking_uri=TRACKING_URI, run_id = RUN_ID):
    params = evaluate.get_params(tracking_uri, run_id)
    params["filter_dates"] = [FILTER_START, FILTER_END] # TO DO - GET DYNAMICALLY
    mlflow.set_experiment("Skagit_DataSource_Compare")
    with mlflow.start_run():
        mlflow.log_params(params)
        mlflow.log_param("hucs", hucs)  
        

        for huc in hucs: 
            model_uri = f"{MODEL_PREFIX}_{huc}"
            model_dawgs = evaluate.load_model(model_uri)
            mlflow.log_param(f"model_uri_{huc}", model_uri)
            # get prediction metrics original model, return df
            df_UA, y_te_pred_UA = eval_UA(huc, model_dawgs, params)
            

            # get prediction metrics snowTEL data
            y_te_pred_snowTEL, y_te_true_snowTEL, train_size = eval_snowTEL(huc,
                model_dawgs, params, df_UA)

            # plot all
            plot_test_results(df_UA, y_te_pred_UA, y_te_pred_snowTEL, y_te_true_snowTEL, train_size,
                params, huc)

def run_all():
    hucs = get_Skagit_list_10()
    eval_all(hucs)
