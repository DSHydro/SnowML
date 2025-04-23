""" Module to download metrics from MLflow """
# pylint: disable=C0103

import os
import mlflow
import pandas as pd

def create_run_dict_Ex3():
    """ Create a dictionary of run_ids by recognizable short names """

    run_dict = {}
    run_dict["base_1e-3"] = "a6c611d4c4cf410e9666796e3a8892b7" # debonair_dove (all 30)
    run_dict["hum_1e-3"] = "d71b47a8db534a059578162b9a8808b7" #peaceful stork (all 30)
    run_dict["srad_1e-3"] = "deed782fda71472fb47cf8670b668473" # enchanting-roo (aborted at 27)
    run_dict["vs_1e-3"] = "4653005687094d9ba54c295b943a4667" # puzzled cow (all 30)

    run_dict["base_3e-4"] = "e989030c272d4de59c84aff739d8063c" # spiffy whale (all 30)
    run_dict["hum_3e-4"] = "51884b406ec545ec96763d9eefd38c36" # capricious snipe (all 30)
    run_dict["srad_3e-4"] = "2b49d6cce3844ede8a66821ae9aec27b" # judicious_mare (all 30)
    run_dict["vs_3e-4"] = "bc031cafad7445adb73173adc43b63c6" # placid-croc (all 30)

    return run_dict


def create_run_dict_Ex3Eval():
    run_dict = {}
    run_dict["Test_Set_A"] = "ccc1af0a3007412bb23cbd4a7cb0d431" # invincable snail
    run_dict["Test_Set_B"] = "148f7dddc0814c4f86bb93edbc425c4c" # bittersweet croc
    return run_dict

def create_run_dict_Ex1():
    run_dict = {}
    run_dict["new10_rerun"] = "bfdd8164ce1f46d8a9418be41a70ffdf"
    run_dict["orig200_rerun"] = "02a508daab8e42b2a5f1baab194cd744"
    return run_dict

def create_run_dict_Ex4(): 
    run_dict = {}
    run_dict["hum30"] = "f76d3fe92f0a479da0e75b9141564287"
    run_dict["hum_mixed_loss30"] = "ba300bba68fd451bbb684283fd3b3eab"
    run_dict["no_hum_mixed_loss10"] = "f64daf3d9751406ab4c054804c51c340"
    run_dict["orig_10_low_lr"] = "8dcf79313b6a43be8b779b2a927714e6" # Aborted early
    run_dict["orig10_ml_05"] = "018b5d59203a49c6bbdb06324eca434f" # Aborted early
    run_dict["hum10"] = "9e1a4f60d2924717be25e33ee065a74c"
    return run_dict

def create_run_dict_Ex4_Prairie():
    run_dict = {}
    run_dict["prairie10"] = "033984f1ce27482090740030fa25af9d" # done
    run_dict["prairie10_mixed"] = "4dde4b91d28349c9b88688a860a6cf34" #done
    run_dict["prairie30_mixed"] = "6842189b35964aff8aa815cc517c5890" #done 
    return run_dict

def create_run_dict_Ex5():
    run_dict = {}
    run_dict["Maritime_DI_low_batch"] = "d8e9971eb89f4ce087fddb766aa85ef1" # stylish ant
    run_dict["Maritime_DI_mse"] = "5d8480525f5246f3a9cad0948daf9fde" # sincere snake
    run_dict["Maritime_DI_mae"] = "c11ded8d42c24b5aa23746e8c7eb0121" # adaptable cat 
    #run_dict["Maritime_DI_hybrid_2"] = "f9118f415b37408ba4d4dd2a6dc3e8e6" # omniscient fish (aborted)
    run_dict["Maritime_DI_hybrid"] = "be0e7e4963a94309b7447ceb70ee8bb2"
    
    #run_dict["Ephemeral_DI_hybrid"]  = "5c31e266ce404aa7a5e5716d04e78b93" # kindly cow
    
    run_dict["Ephemeral_DI_hybrid"] = "ec1bc8b9064841cba401607a5c3866bc" # abrasive finch
    run_dict["Ephemeral_DI_mse"] = "d462d25d0aac4ae1952e502ee435ccd0" # valuable foal
    run_dict["Ephemeral_DI_mae"] = "5d4eae0991c545e2a3a9739cfd1d7268" # righteous snake 
    
    run_dict["Montane_DI_MSE"] = "f56504d6c33447c095615c2167d9b5d2" # grandiose-bee 
    run_dict["Maritime_DI_2_MAE"] = "56d60b81d63e4b81a88e89e344b39fa9" # orderly stoat
    run_dict["Maritime_DI_mae"] = "bebeee98ccb74137b3c629ef6e38b8dc" 
    
    run_dict["Maritime_DI_MSE_recur"] = "586bfeb7a5cc4b769205503c113e7528" # upbeat smelt (from righteous snake)
    

    return run_dict

def create_run_dict_Ex6(): 
    run_dict = {}
    run_dict["Maritime_90Lookback"] = "8d14cdd4d15045b183347211b1e20751"  # Trained on entire HUC by accident
    run_dict["Montane_90Lookback"] = "f82e931b60cd4d21863729305ff312a8"  # Dashing sloth
    return run_dict


# function to retrieve metrics from ML server
def load_ml_metrics(
    run_id,
    save_local=True,
    folder = "mlflow_data/",
    tracking_uri = "arn:aws:sagemaker:us-west-2:677276086662:mlflow-tracking-server/dawgsML"):

    """ Function that retreives all the logged ML metrics for a give run_id"""

    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.MlflowClient()
    run_data = client.get_run(run_id).data
    metric_keys = run_data.metrics.keys()
    # Retrieve full metric history for each key
    all_metrics = []
    for metric in metric_keys:
        history = client.get_metric_history(run_id, metric)
        for record in history:
            all_metrics.append({
                "Metric": metric,
                "Step": record.step,
                "Value": record.value
            })

    # Convert to DataFrame
    metrics_df = pd.DataFrame(all_metrics)

    # Save to CSV if needed
    if save_local:
        f_out = f"{folder}/run_id_data/metrics_from_{run_id}.csv"
        metrics_df.to_csv(f_out, index=False)

    return metrics_df

def download_all(run_dict, folder ="data/", overwrite = False):
    for run_id in run_dict.values():
        f_out =  f"{folder}/run_id_data/metrics_from_{run_id}.csv"
        print(f"processing file {f_out}")
        if (not os.path.exists(f_out) or overwrite):
            load_ml_metrics(run_id)
        else:
            print(f"File already exists: {f_out}")
