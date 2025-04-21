""" Module to compare model performance on diff Skagit data sources.  
Uses Multi-Huc Module """


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
RUN_ID = "a6c611d4c4cf410e9666796e3a8892b7" #deboniar dove
MODEL_URI = "s3://sues-test/298/a6c611d4c4cf410e9666796e3a8892b7/artifacts/epoch8_model"

def eval_all(hucs, tracking_uri=TRACKING_URI, run_id = RUN_ID):
    params = evaluate.get_params(tracking_uri, run_id)
    mlflow.set_experiment("Skagit_DataSource_Compare")
    return params