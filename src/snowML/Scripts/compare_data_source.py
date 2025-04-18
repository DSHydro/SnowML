""" Module to compare model performance on diff Skagit data sources.  
Assumes Single HUC (LOCAL TRAINING) """


# pylint: disable=C0103

import mlflow
import mlflow.pytorch
import pandas as pd
from snowML.LSTM import LSTM_evaluate as evaluate
from snowML.LSTM import LSTM_pre_process as pp
from snowML.LSTM import LSTM_plot2 as plot2
from snowML.Scripts.load_hucs import select_hucs_local_training as sh


# Define some constants 
TRACKING_URI = "arn:aws:sagemaker:us-west-2:677276086662:mlflow-tracking-server/dawgsML"
RUN_ID = "9320981250694bea85107cadcc4714df" 
MODEL_PREFIX = "s3://sues-test/463/9320981250694bea85107cadcc4714df/artifacts/model"
FILTER_START = "1984-10-01"
FILTER_END = "2021-09-30"


def get_Skagit_list_12():
    Skagit_huc_list = sh.assemble_huc_list(input_pairs=[[17110005, '12']])
    return Skagit_huc_list

def get_Skagit_list_10():
    return [1711000504, 1711000505, 1711000506, 1711000507, 1711000508, 1711000509, 1711000511]


def eval_all(huc, tracking_uri=TRACKING_URI, run_id = RUN_ID):
    model_uri = f"{MODEL_PREFIX}_{huc}"
    model_dawgs = evaluate.load_model(model_uri)
    params = evaluate.get_params(tracking_uri, run_id)
    params["filter_dates"] = [FILTER_START, FILTER_END] # TO DO - GET DYNAMICALLY 
    mlflow.set_experiment("Skagit_DataSource_Compare")
    with mlflow.start_run():
        mlflow.log_params(params)
        mlflow.log_param("huc", huc)  # TO DO - LOG THE LIST
        mlflow.log_param("model_uri", model_uri)


        # get prediction metrics original model, return df
        df_UA = eval_UA(huc, model_dawgs, params)

        # get prediction metrics snowTEL data
        eval_snowTEL(huc, model_dawgs, params, df_UA)


def eval_UA (huc, model_dawgs, params):
    df_dict_test =  pp.pre_process_separate([huc], params["var_list"], filter_dates = params["filter_dates"])
    df_UA = df_dict_test[huc]
    metric_dict, data, y_tr_pred_UA, y_te_pred_UA, _, _, train_size = evaluate.eval_from_saved_model(
            model_dawgs, df_dict_test, huc, params)
    plot2.plot(data, y_tr_pred_UA, y_te_pred_UA, train_size, huc, params, metrics_dict = metric_dict)
    for met_nm, metric in metric_dict.items():
        mlflow.log_metric(f"{met_nm}_{str(huc)}_UA", metric)
        print(f"{met_nm}: {metric}")
    return df_UA

def eval_snowTEL (huc, model_dawgs, params, df_UA):
    f = f"notebooks/Ex1_MoreData/orig_data/wus-sr-skagit-{huc}-mean-swe.csv"
    df_snowTEL = pd.read_csv(f)
    df_UA["mean_swe"] = df_snowTEL["mean"].values  # replae mean_swe column
    print(df_UA.head())
    df_dict_snowTEL = {}
    df_dict_snowTEL[huc] = df_UA  # Store updated df
    metric_dict, data, y_tr_pred_snowTEL, y_te_pred_snowTEL, _, _, train_size = evaluate.eval_from_saved_model(
            model_dawgs, df_dict_snowTEL, huc, params)
    plot2.plot(data, y_tr_pred_snowTEL, y_te_pred_snowTEL, train_size, huc, params, metrics_dict = metric_dict)
    for met_nm, metric in metric_dict.items():
        mlflow.log_metric(f"{met_nm}_{str(huc)}_snowTel", metric)
        print(f"{met_nm}: {metric}")

def run_all(): 
    hucs = get_Skagit_list_10()
    for huc in hucs: 
        eval_all(huc)

