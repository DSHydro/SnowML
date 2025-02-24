# Model to run end to end pipeline
# pylint: disable=C0103

import importlib
import s3fs
import data_utils as du
import set_data_constants as sdc
import get_bronze as gb
import bronze_to_gold as btg
import gold_to_model as gtm
import get_geos as gg

def process_one_huc (huc_id, bucket_dict = None, var_list = None, overwrite = False):
    # verify inputs
    huc_lev = str(len(str(huc_id))).zfill(2)

    huc_lev_permitted = ['10', '12']  # TO DO: FIX TO ALLOW 4/6/8 ETC
    assert huc_lev in huc_lev_permitted, f"Type must be one of {huc_lev_permitted}"

    # some set up
    if bucket_dict is None:
        bucket_dict = sdc.create_bucket_dict("prod")
    if var_list is None:
        var_dict = sdc.create_var_dict()
        var_list = list(var_dict.keys())

    # get geos
    geos = gg.get_geos(huc_id, huc_lev)
    num_geos = geos.shape[0]
    print(f"Number of geos to process is {num_geos}")

    # create gold files if needed
    b_gold = bucket_dict.get("gold")

    for var in var_list:
        f =  f"mean_{var}_in_{huc_id}.csv" 
        if overwrite or (not du.isin_s3(b_gold, f)):
            print(f"Creating gold file {f}")
            btg.bronze_to_gold (geos, var)

    #create model ready data
    model_df = gtm.huc_gold(huc_id) 
    return model_df
