# Module to evaluate model results from a saved model
# pylint: disable=C0103, R0913, R0914, R0917

import mlflow
import mlflow.pytorch
import pandas as pd
from snowML.LSTM import LSTM_pre_process as pp
from snowML.LSTM import LSTM_train
from snowML.LSTM import set_hyperparams as sh
from snowML.datapipe import data_utils as du
from snowML.datapipe import set_data_constants as sdc

def load_model(model_uri):
    print(model_uri)
    model = mlflow.pytorch.load_model(model_uri)
    print(model)
    return model

def set_ML_server(expirement_name):
    """
    Configures the MLflow tracking server and sets the experiment.

    Returns:
        None
    """
    tracking_uri = "arn:aws:sagemaker:us-west-2:677276086662:mlflow-tracking-server/dawgsML"
    mlflow.set_tracking_uri(tracking_uri)
    # Define the expirement
    mlflow.set_experiment(expirement_name)


def reset_params(var_list,
                learning_rate,
                batch_size,
                expirement_name =  "Predict From PreTrained",
                train_size_dimension = "huc",
                train_size_fraction = 0,
                epochs = -500):
    params = sh.create_hyper_dict()
    params["batch_size"] = batch_size
    params["var_list"] = var_list
    params["learning_rate"] = learning_rate
    params["expirement_name"] = expirement_name
    params["train_size_dimension"] = train_size_dimension
    params["train_size_fraction"] = train_size_fraction
    params["epochs"] = epochs
    return params


def assemble_df_dict(huc_list, var_list, bucket_dict=None):
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


# TO DO: add option to load in global_means and std from MLflow instead of recalc
def renorm(train_hucs, val_hucs, test_hucs, var_list):
    # compute global_means and std used in training
    huc_list_all_tr = train_hucs + val_hucs
    _, global_means, global_stds = pp.pre_process(huc_list_all_tr, var_list)
    print(f"global means were {global_means}, global_stds were {global_stds}")
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
        data, y_train_pred, y_test_pred, y_train_true, y_test_true, train_size_main = LSTM_train.predict(model_dawgs, df_dict, huc, params)
        test_mse = LSTM_train.mean_squared_error(y_test_true, y_test_pred)
        test_kge, _, _, _ = LSTM_train.kling_gupta_efficiency(y_test_true, y_test_pred)
        LSTM_train.plot(data, y_train_pred, y_test_pred, train_size_main, huc, params)
        return test_mse, test_kge

    # else train/test split is time
    print("still working on this branch")
    return -500, -500


def predict_from_pretrain (train_hucs,
                        val_hucs,
                        test_hucs,
                        model_uri,
                        var_list,
                        learning_rate,
                        batch_size,
                        ):

    params = reset_params(var_list, learning_rate,
                    batch_size)

    model_dawgs = load_model(model_uri)

    #df_dict_test = assemble_df_dict(test_hucs, params["var_list"])
    df_dict_test = renorm(train_hucs, val_hucs, test_hucs, var_list)
    print(df_dict_test)

    # initialize ML server
    set_ML_server(params["expirement_name"])

    with mlflow.start_run():
        mlflow.log_params(params)
        mlflow.log_param("test_hucs", test_hucs)
        mlflow.log_param("model_uri", model_uri)

        for huc in test_hucs:
            test_mse, test_kge = eval_from_saved_model(model_dawgs, df_dict_test, huc, params)
            print (f"for huc {huc}, test_mse was {test_mse}, test_kge was {test_kge}")
               