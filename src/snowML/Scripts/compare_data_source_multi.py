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
    # TO DO - GET GLOBAL MEANS AND STD FROM MLFLOW
    #params["filter_dates"] = [FILTER_START, FILTER_END] # TO DO - GET DYNAMICALLY 
    mlflow.set_experiment("Skagit_DataSource_Compare")
    with mlflow.start_run():
        mlflow.log_params(params)
        mlflow.log_param("huc", huc)  # TO DO - LOG THE LIST
        mlflow.log_param("model_uri", model_uri)


        # get prediction metrics original model, return df
        df_UA = eval_UA(huc, model_dawgs, params)

        # get prediction metrics snowTEL data
        eval_snowTEL(huc, model_dawgs, params, df_UA)