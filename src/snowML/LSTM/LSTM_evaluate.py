""" Module to evaluate model results from a saved model """
# pylint: disable=C0103

import ast
import mlflow
import mlflow.pytorch
import pandas as pd
from snowML.LSTM import LSTM_pre_process as pp
from snowML.LSTM import LSTM_train
from snowML.LSTM import LSTM_plot3 as plot3
from snowML.LSTM import LSTM_metrics as met
from snowML.datapipe import data_utils as du
from snowML.datapipe import set_data_constants as sdc


def load_model(model_uri):
    """
    Load a PyTorch model from the given URI using MLflow.

    Args:
        model_uri (str): The URI of the model to load.

    Returns:
        torch.nn.Module: The loaded PyTorch model.
    """
    print(model_uri)
    model = mlflow.pytorch.load_model(model_uri)
    print(model)
    return model

def get_params(tracking_uri, run_id):
    """
    Retrieve parameters from an MLflow run.

    Args:
        tracking_uri (str): The URI of the MLflow tracking server.
        run_id (str): The ID of the MLflow run.
        
    Returns:
        dict: A dictionary containing the parameters of the specified MLflow run.
    """
    mlflow.set_tracking_uri(tracking_uri)
    run = mlflow.get_run(run_id)
    params = run.data.params
      # reformat some lists that got converted to string literals
    for key in ["var_list", "train_hucs", "val_hucs"]:
        if params.get(key):
            params[key] = ast.literal_eval(params[key])
    # convert some strings back to int or float that we need for predict and plotting
    for key in ['lookback']:
        params[key] = int(params[key])
    for key in ['train_size_fraction']:
        params[key] = float(params[key])
    params["lag_swe_var_idx"] =  2 # TO DO - MAKE DYNAMIC
    params["lag_days"] = 30 # TO DO - MAKE DYNAMIC
    return params


def assemble_df_dict(huc_list, var_list, bucket_dict=None):
    """
    Assembles a dictionary of DataFrames for given HUCs (Hydrologic Unit Codes) and variables.

    Parameters:
    huc_list (list): List of HUCs for which to assemble DataFrames.
    var_list (list): List of variables to include in the DataFrames.
    bucket_dict (dict, optional): Dictionary containing bucket information. 
        If None, a default bucket dictionary is created.

    Returns:
    dict: A dictionary where keys are HUCs and values are DataFrames containing
         the specified variables and 'mean_swe'.
    """
    df_dict = {}  # Initialize dictionary
    if bucket_dict is None:
        bucket_dict = sdc.create_bucket_dict("prod")
    bucket_name = bucket_dict["model-ready"]

    for huc in huc_list:
        file_name = f"model_ready_huc{huc}.csv"
        df = du.s3_to_df(file_name, bucket_name)
        df['day'] = pd.to_datetime(df['day'])
        df.set_index('day', inplace=True)  # Set 'day' as the index
        # Collect only the columns of interest
        col_to_keep = var_list + ["mean_swe"]

        for col in col_to_keep:
            if col not in df.columns:
                print(f"huf{huc} is missing col {col}")

        df = df[col_to_keep]
        df_dict[huc] = df  # Store DataFrame in dictionary

    return df_dict


def renorm(train_hucs, val_hucs, test_hucs, var_list):
    """
    Renormalizes the test HUCs (Hydrologic Unit Codes) using the global means
    and standard deviations computed from the training and validation HUCs.

    Parameters:
    train_hucs (list): List of training HUCs.
    val_hucs (list): List of validation HUCs.
    test_hucs (list): List of test HUCs to be renormalized.
    var_list (list): List of variables to be considered for normalization.

    Returns:
    dict: A dictionary where keys are HUCs and values are the renormalized DataFrames.
    """
    # compute global_means and std used in training
    huc_list_all_tr = train_hucs + val_hucs
    _, global_means, global_stds = pp.pre_process(huc_list_all_tr, var_list)
    # create dictionary of of hucs to test
    df_dict = assemble_df_dict(test_hucs, var_list, bucket_dict=None)
    # renormalize with the global_means and global_std used in training
    for huc, df in df_dict.items():
        df = pp.z_score_normalize(df, global_means, global_stds)
        df_dict[huc] = df  # Store normalized DataFrame

    return df_dict



def eval_from_saved_model (model_dawgs, df_dict, huc, params):
    print(f"evaluating on huc {huc}")

    if params["train_size_dimension"] == "huc":
        # all data is "test" data
        params["train_size_fraction"] = 0
        data, y_tr_pred, y_te_pred, y_tr_true, y_te_true, y_te_pred_recur, train_size = LSTM_train.predict(
            model_dawgs, df_dict, huc, params)
    else: # else train/test split is time
        if params.get("train_size_fraction") in {0, 1}:
            raise ValueError("Train_size_fraction cannot be 0 or 1 if training dimension is time")
        data, y_tr_pred, y_te_pred, y_tr_true, y_te_true,  y_te_pred_recur, train_size, = LSTM_train.predict(model_dawgs,
            df_dict, huc, params)

        #print("Last few elements of y_te_true", y_te_true[-10])
        #print("Last few elements of y_te_pred", y_te_pred[-10])

    metric_dict_test = met.calc_metrics(y_te_true, y_te_pred, metric_type = "test")
    metric_dict_test_recur = met.calc_metrics(y_te_true, y_te_pred_recur, metric_type = "test_recur")
    return metric_dict_test, metric_dict_test_recur, data, y_tr_pred, \
        y_te_pred, y_tr_true, y_te_true, y_te_pred_recur, train_size


def predict_from_pretrain(test_hucs, run_id, model_uri, mlflow_tracking_uri,
    mlflow_log_now = True, recur_predict = False):

    # retrieve model details from mlflow
    model_dawgs = load_model(model_uri)
    params = get_params(mlflow_tracking_uri, run_id)
    if recur_predict:
        params["recursive_predict"] = True
    else:
        params["recursive_predict"] = False

    # assemble test data

    if params["train_size_dimension"] == "huc":
        df_dict_test = assemble_df_dict(test_hucs, params["var_list"])
        # normalize test data using same means/standard dev used in training
        df_dict_test = renorm(params["train_hucs"],  params["val_hucs"],
            test_hucs, params["var_list"])
    else:
        # normalize test huc against itself only (as in training)
        df_dict_test =  pp.pre_process_separate(test_hucs, params["var_list"])

    if mlflow_log_now:
        mlflow.set_experiment("Predict_From_Pretrain")
        with mlflow.start_run():
            mlflow.log_params(params)
            mlflow.log_param("test_hucs", test_hucs)
            mlflow.log_param("model_uri", model_uri)

            for huc in test_hucs:
                metric_dict_test, metric_dict_test_recur, data, y_tr_pred, y_te_pred, _, y_te_true, y_te_pred_recur, train_size = eval_from_saved_model(
                    model_dawgs, df_dict_test, huc, params)
                for m_dict in [metric_dict_test, metric_dict_test_recur]:
                    met.log_print_metrics(m_dict, huc, 0)

                print("y_te_pred", y_te_pred.flatten()[400:420])
                print("y_te_pred_recur", y_te_pred_recur.flatten()[400:420])

                plot_dict_true = plot3.assemble_plot_dict(y_te_true, "blue",
                        'SWE Estimates UA Data (Physics Based Model)')
                plot_dict_te = plot3.assemble_plot_dict(y_te_pred, "green",
                        'SWE Estimates Prediction')     
                plot_dict_te_recur = plot3.assemble_plot_dict(y_te_pred_recur, "black",
                        'SWE Estimates Recursive Prediction')
                y_dict_list = [plot_dict_true, plot_dict_te, plot_dict_te_recur ]
                ttl = "SWE_Actual_vs_Predicted_for_huc_{huc}"
                x_axis_vals = data.index[9523:]  # TO DO - MAKE DYNAMIC
                plot3.plot3(x_axis_vals, y_dict_list, ttl)
    else:
        for huc in test_hucs:
            metric_dict_test, metric_dict_test_recur, data, y_tr_pred, y_te_pred, _, y_te_true, y_te_pred_recur, train_size = eval_from_saved_model(
                model_dawgs, df_dict_test, huc, params)
            for m_dict in [metric_dict_test, metric_dict_test_recur]:
                met.log_print_metrics(m_dict, 0)

            plot_dict_true = plot3.assemble_plot_dict(y_te_true, "blue",
                    'SWE Estimates UA Data (Physics Based Model)')
            plot_dict_te = plot3.assemble_plot_dict(y_te_pred, "green",
                    'SWE Estimates Prediction')     
            plot_dict_te_recur = plot3.assemble_plot_dict(y_te_pred_recur, "black",
                    'SWE Estimates Recursive Prediction')
            y_dict_list = [plot_dict_true, plot_dict_te, plot_dict_te_recur]
            ttl = "SWE_Actual_vs_Predicted_for_huc_{huc}"
            x_axis_vals = data.index
            plot3.plot3(x_axis_vals, y_dict_list, ttl)
