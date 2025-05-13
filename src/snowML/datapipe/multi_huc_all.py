# Module to run end to end pipeline
# pylint: disable=C0103


import pandas as pd
import geopandas as gpd
import sys
from contextlib import redirect_stdout, redirect_stderr
from snowML.datapipe.utils import data_utils as du
from snowML.datapipe.utils import set_data_constants as sdc
from snowML.datapipe.utils import get_geos as gg
from snowML.datapipe import bronze_to_gold as btg
from snowML.datapipe import to_model_ready as gtm

def process_multi_huc (geos,
                       bucket_dict = None,
                       var_list = None,
                       overwrite_gold = False,
                       overwrite_mod = False):
    
    # some set up
    if bucket_dict is None:
        bucket_dict = sdc.create_bucket_dict("prod")
    if var_list is None:
        var_dict = sdc.create_var_dict()
        var_list = list(var_dict.keys())

    # get geos
    num_geos = geos.shape[0]
    print(f"Number of geos to process is {num_geos}")

    # create gold files if needed
    b_gold = bucket_dict.get("gold")

    for i in range(geos.shape[0]):
        row = geos.iloc[i]
        huc_id = row["huc_id"]
        if overwrite_gold:
            #print(f"Creating necessary gold files for {var}")
            btg.process_geos(geos, var)  # create gold files for all geos while var bronze open
        for var in var_list:
            f =  f"mean_{var}_in_{huc_id}.csv"
            if not du.isin_s3(b_gold, f):
                #print(f"Creating necessary gold files for {var}")
                btg.process_geos(geos, var)  # create gold files for all geos while var bronze open

        #create model ready data
        try:
            model_df = gtm.huc_model(huc_id, overwrite_mod= overwrite_mod)
        except Exception as e:
            print(f"Error processing huc{huc_id}: {e}, omitting from dataset")
    return model_df


def compile_geos(huc_list):
    # Create an empty GeoDataFrame with appropriate columns and CRS
    results_gdf = gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs="EPSG:4326")

    for huc_id in huc_list:
        level = str(len(str(huc_id))).zfill(2)
        new_gdf = gg.get_geos(huc_id, level)
        results_gdf = pd.concat([results_gdf, new_gdf], ignore_index=True)

    return results_gdf



def run_and_log(func, f_print, f_err, *args, **kwargs):
    with open(f_print, 'w') as out, open(f_err, 'w') as err:
        with redirect_stdout(out), redirect_stderr(err):
            func(*args, **kwargs)


def process_multi_huc_quiet(geos): 
    run_and_log(process_multi_huc, "model_ready.txt", "model_ready_err.txt", geos)