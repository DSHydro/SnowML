# Check for mismatch 

import pandas as pd
import time
from snowML.datapipe import set_data_constants as sdc
from snowML.datapipe import bronze_to_gold as btg
from snowML.datapipe import get_geos as gg
from snowML.datapipe import data_utils as du 


def check_for_mismatch(huc_id, huc_lev, var_list = ["pr", "swe"], var_dict = None):
    time_start = time.time()
    if var_dict is None: 
        var_dict = sdc.create_var_dict()
    geos = gg.get_geos(huc_id, huc_lev) 
    counts_df = pd.DataFrame()
    for var in var_list: 
        counts_var = []
        ds = btg.prep_bronze(var, bucket_dict = None)
        for i in range(geos.shape[0]): 
            row = geos.iloc[i]
            ds_small = btg.create_mask(ds, row, geos.crs)
            var_name = var_dict[var]
            non_nan_count = ds_small[var_name].count().compute().item()
            counts_var.append(non_nan_count)
            print(f"non_nan counts for {var}, {i} is {non_nan_count}")
        counts_df[var] = counts_var 
    counts_df = calc_mismatch(counts_df)
    threshold = 25 
    warnings = counts_df[counts_df["%mismatch"] > threshold]
    du.elapsed(time_start)
    return counts_df, warnings  

def calc_mismatch(counts_df): 
    counts_df["min"] = counts_df.min(axis=1)
    counts_df["max"] = counts_df.max(axis=1)
    counts_df["%mismatch"] = 100*(counts_df["max"] - counts_df["min"])/counts_df["min"]
    return counts_df

