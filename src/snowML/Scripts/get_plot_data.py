import mlflow
from snowML.LSTM import LSTM_evaluate as eval 

# define constants 
T_URI = "arn:aws:sagemaker:us-west-2:677276086662:mlflow-tracking-server/dawgsML" 
M_URI = "s3://sues-test/298/51884b406ec545ec96763d9eefd38c36/artifacts/epoch27_model" 
RUN_ID = "d71b47a8db534a059578162b9a8808b7" 
#HUC = '170300010701'
HUC = '170300010402'


def get_plot_data(model_uri=M_URI, mlflow_tracking_uri = T_URI, run_id=RUN_ID, huc_to_plot=HUC):
    test_hucs = [huc_to_plot]
    model_dawgs = eval.load_model(model_uri)
    params = eval.get_params(mlflow_tracking_uri, run_id)
    df_dict_test = eval.assemble_df_dict(test_hucs, params["var_list"])
    df_dict_test = eval.renorm(params["train_hucs"],  params["val_hucs"], test_hucs, params["var_list"])
    metric_dict, data, y_tr_pred, y_te_pred, y_tr_true, y_te_true, train_size_main= eval.eval_from_saved_model(model_dawgs,
         df_dict_test, huc_to_plot, params)
    return data, y_tr_pred, y_te_pred, train_size_main, huc_to_plot, params, metric_dict



