""" Module to compare model performance on diff Skagit data sources.  
Uses Multi-Huc Module """


# pylint: disable=C0103

import re
import mlflow
import mlflow.pytorch
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from snowML.LSTM import LSTM_evaluate as evaluate
from snowML.LSTM import LSTM_pre_process as pp
from snowML.LSTM import LSTM_plot2 as plot2
from snowML.Scripts.load_hucs import select_hucs_local_training as sh


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

    globals = {}
    for key, value in matches:
        globals[key] = float(value)
    return globals

def get_params(tracking_uri, run_id):
    params = evaluate.get_params(tracking_uri, run_id)
    global_means = extract_globals(params['global_means'])
    global_stds = extract_globals(params['global_stds'])
    params["filter_dates"] = [FILTER_START, FILTER_END] # TO DO - GET DYNAMICALLY
    return params, global_means, global_stds

def assemble_plot_dict(y_vals, color, label, n_offset=180): 
    plot_dict = {}
    plot_dict["y_axis_vals"] = np.concatenate([np.full(n_offset, np.nan), y_vals.flatten()])
    plot_dict["color"] = color
    plot_dict["label"] = label
    return plot_dict

def eval_UA(model_dawgs, df_dict_test, huc, params):
    metric_dict, data, _, y_te_pred_UA, _, y_te_true_UA, _, train_size = evaluate.eval_from_saved_model(
            model_dawgs, df_dict_test, huc, params)
    for met_nm, metric in metric_dict.items():
        mlflow.log_metric(f"{met_nm}_{str(huc)}_snowTel", metric)
        print(f"{met_nm}: {metric}")
    
    # plot
    plot_dict_true = assemble_plot_dict(y_te_true_UA, "blue", 'SWE Estimates UA Data (Physics Based Model)')
    plot_dict_te = assemble_plot_dict(y_te_pred_UA, "green", 'SWE Estimates Prediction Using UA Data') 
    ttl = f"SWE_Predictions_for_huc_{huc} UA Data"
    df_UA = df_dict_test[huc]
    x_axis_vals = df_UA.index
    plot3(huc, x_axis_vals, [plot_dict_true, plot_dict_te], ttl, metrics_dict = metric_dict)
    return plot_dict_true, plot_dict_te 

def eval_snowTEL (huc, model_dawgs, params, df_UA_norm):
    f = f"notebooks/Ex1_MoreData/orig_data/wus-sr-skagit-{huc}-mean-swe.csv"
    df_snowTEL = pd.read_csv(f)
    df_UA_copy = df_UA_norm.copy()
    df_UA_copy["mean_swe"] = df_snowTEL["mean"].values  # replace mean_swe column
    df_dict_snowTEL = {}
    df_dict_snowTEL[huc] = df_UA_copy  # Store updated df
    metric_dict, data, y_tr_pred_snowTEL, y_te_pred_snowTEL, _, y_te_true_snowTEL, _, train_size = evaluate.eval_from_saved_model(
            model_dawgs, df_dict_snowTEL, huc, params)
    for met_nm, metric in metric_dict.items():
        mlflow.log_metric(f"{met_nm}_{str(huc)}_snowTel", metric)
        print(f"{met_nm}: {metric}")

    # plot
    plot_dict_true = assemble_plot_dict(y_te_true_snowTEL, "black", 'SWE - Orig Skgit Data')
    plot_dict_te = assemble_plot_dict(y_te_pred_snowTEL, "red", 'SWE Estimates Using Orig Skagit Data') 
    x_axis_vals = df_UA_norm.index
    ttl = f"SWE Estimates for {huc} Using Orig Skagit Data"
    plot3(huc, x_axis_vals, [plot_dict_true, plot_dict_te], ttl, metrics_dict = metric_dict)
    return plot_dict_true, plot_dict_te 
    


def eval_all(hucs, tracking_uri=TRACKING_URI, run_id = RUN_ID, model_uri = MODEL_URI):
    params, global_means, global_stds = get_params(tracking_uri, run_id)
    mlflow.set_experiment("Skagit_DataSource_Compare")
    model_dawgs = evaluate.load_model(model_uri)
    with mlflow.start_run():
        mlflow.log_params(params)
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
            plot3(huc, x_axis_vals, y_dict_list, ttl)

            # plot just green 
            #ttl = f"SWE_Predictions_for_huc_{huc} just green"
            #plot3(huc, x_axis_vals, [plot_dict_te_UA, plot_dict_true_UA, plot_dict_true], ttl)


            

    return None
    
def plot3(huc, x_axis_vals, y_dict_list, ttl, metrics_dict = None): 
    plt.figure(figsize=(12, 6))
    for plot_dict in y_dict_list: 
        plt.plot(
            x_axis_vals, 
            plot_dict["y_axis_vals"], 
            c = plot_dict["color"], 
            label = plot_dict["label"])
    plt.legend(loc='upper right')
    plt.xlabel('Date')
    plt.ylabel('SWE')
    plt.title(ttl)

    # Display metrics in the upper-right corner if metrics_dict is not None
    if metrics_dict is not None:
        ax = plt.gca()
        metric_text = "\n".join([f"{key}: {value:.3f}" for key, value in metrics_dict.items()])

        ax.text(
            0.02, 0.98, metric_text, transform=ax.transAxes, ha='left', va='top',
            fontsize=10, color='black', bbox=dict(facecolor='white', alpha=0.7, edgecolor='black')
        )

    mlflow.log_figure(plt.gcf(), ttl + ".png")
    plt.close()
    
       