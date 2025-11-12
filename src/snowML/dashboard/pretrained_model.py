"""Module to run the pretrrained model on the dashboard """
# pylint: disable=C0103

import os
import json
import torch
import boto3
from snowML.LSTM.LSTM_model import SnowModel
from snowML.LSTM import LSTM_evaluate as evaluate
from snowML.LSTM import LSTM_pre_process as pp
from snowML.LSTM import LSTM_plot3 as plot3

s3 = boto3.client("s3")

def get_json_dict(model_name, suffix, bucket_name="snowml-dashboard"):
    """
    Load the saved model parameters (JSON) from S3.

    Parameters
    ----------
    model_name : str
        Base name used when saving (without '_{suffix}.json').
    bucket_name : str
        S3 bucket name.

    Returns
    -------
    params : dict
        Dictionary of model parameters (e.g., input_size, hidden_size, etc.)
    """
    key = f"models/{model_name}_{suffix}.json"
    print(key)
    local_path = f"/tmp/{model_name}_{suffix}.json"

    s3.download_file(bucket_name, key, local_path)
    with open(local_path, "r") as f:
        saved_dict = json.load(f)
    os.remove(local_path)
    return saved_dict



#@st.cache_data
def load_model_from_s3(model_name, bucket_name="snowml-dashboard", map_location=None):
    """
    Load a SnowModel from S3 using parameters stored in <model_name>_params.json
    and weights stored in <model_name>.pth.

    Parameters
    ----------
    model_name : str
        Base name used when saving (without '.pth' or '_params.json').
    bucket_name : str
        S3 bucket name.
    map_location : str or torch.device, optional
        Where to map the model (e.g., 'cpu' or 'cuda').

    Returns
    -------
    model : SnowModel
        The reloaded model with weights restored.
    params : dict
        The model’s parameter dictionary.
    """

    # --- 1️⃣ Load parameters ---
    params = get_json_dict(model_name, "params", bucket_name)
    global_means = get_json_dict(model_name, "means", bucket_name)
    global_stds = get_json_dict(model_name, "stds", bucket_name)

    # Compute input size dynamically
    input_size = len(params["var_list"])

    # --- 2️⃣ Download model weights ---
    weights_key = f"models/{model_name}.pth"
    local_path = f"/tmp/{model_name}.pth"
    s3.download_file(bucket_name, weights_key, local_path)

    # --- 3️⃣ Initialize model ---
    model = SnowModel(
        input_size=input_size,
        hidden_size=int(params["hidden_size"]),
        num_class=int(params["num_class"]),
        num_layers=int(params["num_layers"]),
        dropout=float(params["dropout"])
    )

    # --- 4️⃣ Load weights ---
    state_dict = torch.load(local_path, map_location=map_location or torch.device("cpu"))
    model.load_state_dict(state_dict)

    # --- 5️⃣ Cleanup ---
    os.remove(local_path)
    model.eval()

    return model, params, global_means, global_stds


def create_df_dict(huc12, params, global_means, global_stds):
    df_dict = evaluate.assemble_df_dict([huc12], params["var_list"], bucket_dict=None)
    # renormalize with the global_means and global_std used in training
    for huc, df in df_dict.items():
        df = pp.z_score_normalize(df, global_means, global_stds)
        df_dict[huc] = df  # Store normalized DataFrame
    return df_dict

def plot(y_te_true, y_te_pred, data, metrics_dict, train_size, huc12):
    plot_dict_true = plot3.assemble_plot_dict(y_te_true, "blue",
            'SWE Estimates UA Data (Physics Based Model)')
    plot_dict_te = plot3.assemble_plot_dict(y_te_pred, "green",
            'SWE Estimates Prediction')
    y_dict_list = [plot_dict_true, plot_dict_te]
    ttl = f"SWE_Actual_vs_Predicted_for_huc_{huc12}"
    x_axis_vals = data.index[train_size:]
    fig = plot3.plot3b(x_axis_vals, y_dict_list, ttl, metrics_dict=metrics_dict)

    #fig_path = f"{ttl}.png"
    #fig.savefig(fig_path, bbox_inches="tight", dpi=300)
    return fig

def eval_huc(huc12):
    model_name = 'Multi_Trained_Base'
    model_dawgs, params, global_means, global_stds = load_model_from_s3(model_name)
    params["recursive_predict"] = False
    df_dict_test = create_df_dict(huc12, params, global_means, global_stds)
    metric_dict_test, _, data, _, y_te_pred, _, y_te_true, _, train_size = evaluate.eval_from_saved_model(
        model_dawgs, df_dict_test, huc12, params)
    fig = plot(y_te_true, y_te_pred, data, metric_dict_test, train_size, huc12)
    return metric_dict_test, fig



#for run base_1e-3 the highest median test kge achieved was 0.811 in epoch 8
# run_id =	"a6c611d4c4cf410e9666796e3a8892b7"
# debonair_dove
# Expirement Name is "Multi-ALl-2"
