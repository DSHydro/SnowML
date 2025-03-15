"""Script to retrieve data needed to create SWE prediction plots for a single huc"""

import mlflow
from snowML.LSTM import LSTM_evaluate as eval 
from snowML.LSTM import LSTM_pre_process as pp

# define constants 
T_URI = "arn:aws:sagemaker:us-west-2:677276086662:mlflow-tracking-server/dawgsML" 

#M_URI = "s3://sues-test/298/51884b406ec545ec96763d9eefd38c36/artifacts/epoch27_model" 
#RUN_ID = "d71b47a8db534a059578162b9a8808b7" 
#HUC = '170300010701'

M_URI = "s3://sues-test/200/07a71994891e487ba36bfd3675a860a8/artifacts/model_170300010402"
RUN_ID = "07a71994891e487ba36bfd3675a860a8"
HUC = '170300010402'


def get_plot_data(model_uri=M_URI, mlflow_tracking_uri = T_URI, run_id=RUN_ID, huc_to_plot=HUC):
    test_hucs = [huc_to_plot]
    model_dawgs = eval.load_model(model_uri)
    params = eval.get_params(mlflow_tracking_uri, run_id)
    
    if params["train_size_dimension"] == "huc": 
        df_dict_test = assemble_df_dict(test_hucs, params["var_list"])
        # normalize test data using same means/standard dev used in training
        df_dict_test = renorm(params["train_hucs"],  params["val_hucs"], test_hucs, params["var_list"])
    else: 
        # normalize test huc against itself only (as in training)
        df_dict_test, _, _ =  pp.pre_process(test_hucs, params["var_list"])
   
    metric_dict, data, y_tr_pred, y_te_pred, y_tr_true, y_te_true, train_size = eval.eval_from_saved_model(model_dawgs,
         df_dict_test, huc_to_plot, params)
    return data, y_tr_pred, y_te_pred, train_size, huc_to_plot, params, metric_dict



