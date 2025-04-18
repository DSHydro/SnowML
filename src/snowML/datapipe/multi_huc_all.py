# Model to run end to end pipeline
# pylint: disable=C0103

from snowML.datapipe import data_utils as du
from snowML.datapipe import set_data_constants as sdc
from snowML.datapipe import bronze_to_gold as btg
from snowML.datapipe import gold_to_model as gtm
from snowML.datapipe import get_geos as gg


def process_multi_huc (huc_id_start,
                       final_huc_lev,
                       bucket_dict = None,
                       var_list = None,
                       overwrite_gold = False,
                       overwrite_mod = False):
    # verify inputs
    huc_lev_permitted = ['10', '12']
    assert final_huc_lev in huc_lev_permitted, f"Type must be one of {huc_lev_permitted}"

    # some set up
    if bucket_dict is None:
        bucket_dict = sdc.create_bucket_dict("prod")
    if var_list is None:
        var_dict = sdc.create_var_dict()
        var_list = list(var_dict.keys())

    # get geos
    geos = gg.get_geos(huc_id_start, final_huc_lev)
    num_geos = geos.shape[0]
    print(f"Number of geos to process is {num_geos}")

    # create gold files if needed
    b_gold = bucket_dict.get("gold")

    for i in range(geos.shape[0]):
        row = geos.iloc[i]
        huc_id = row["huc_id"]
        if overwrite_gold:
            print(f"Creating necessary gold files for {var}")
            btg.process_geos(geos, var)  # create gold files for all geos while var bronze open
        for var in var_list:
            f =  f"mean_{var}_in_{huc_id}.csv"
            if not du.isin_s3(b_gold, f):
                print(f"Creating necessary gold files for {var}")
                btg.process_geos(geos, var)  # create gold files for all geos while var bronze open

        #create model ready data
        try:
            model_df = gtm.huc_model(huc_id, overwrite_mod= overwrite_mod)
        except Exception as e:
            print(f"Error processing huc{huc_id}: {e}, omitting from dataset")
    return model_df
    