"""Module to Save PreTrained Model Data to S3 for Use in Dashabord"""
# pylint: disable=C0103

import os
import json
#import mlflow
#import mlflow.pytorch
import torch
import boto3
#from snowML.datapipe import to_model_ready as mr
from snowML.LSTM import LSTM_evaluate as evaluate
from snowML.LSTM import LSTM_pre_process as pp

s3 = boto3.client("s3")

def model_from_MLflow(uri):
    model = evaluate.load_model(uri)
    return model

def model_to_s3(model, model_name, bucket_name = "snowml-dashboard"):
    file_name = f"{model_name}.pth"
    torch.save(model.state_dict(), file_name)
    s3.upload_file(file_name, bucket_name, f"models/{file_name}")
    os.remove(file_name)
    return file_name

def params_to_s3(params, model_name, bucket_name = "snowml-dashboard"):
    params_file = f"{model_name}_params.json"
    with open(params_file, "w") as f:
        json.dump(params, f, indent=2)
    s3.upload_file(params_file, bucket_name, f"models/{params_file}")
    os.remove(params_file)

def norms_to_s3(g_means, g_stds, model_name, bucket_name = "snowml-dashboard"):
    g_means = g_means.to_dict()
    g_stds = g_stds.to_dict()
    means_file = f"{model_name}_means.json"
    with open(means_file, "w") as f:
        json.dump(g_means, f, indent=2)
    s3.upload_file(means_file, bucket_name, f"models/{means_file}")
    os.remove(means_file)
    std_file = means_file = f"{model_name}_stds.json"
    with open(std_file, "w") as f:
        json.dump(g_stds, f, indent=2)
    s3.upload_file(std_file, bucket_name, f"models/{std_file}")
    os.remove(std_file)
    

def get_norm(params):
    huc_list_all_tr = params["train_hucs"] + params["val_hucs"]
    _, global_means, global_stds = pp.pre_process(huc_list_all_tr, params["var_list"])
    return global_means, global_stds


def save_all_model_data(model_uri, model_name, tracking_uri, run_id):
    model = evaluate.load_model(model_uri)
    model_to_s3(model, model_name)
    params = evaluate.get_params(tracking_uri, run_id)
    params_to_s3(params, model_name)
    return params


