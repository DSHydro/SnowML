"""Module to update model_ready data with lagged variables"""

import time
import pandas 
from snowML.datapipe import set_data_constants as sdc
from snowML.datapipe import data_utils as du
from snowML.datapipe import get_geos as gg


def load_model_ready(huc_id, bucket_nm = None): 
    f_in = f"model_ready_huc{huc_id}.csv"
    if bucket_nm is None:
        bucket_dict = sdc.create_bucket_dict("prod")
        bucket_nm = bucket_dict["model-ready"]
        print(bucket_nm)
    df = du.s3_to_df(f_in, bucket_nm)
    return df

def add_lagged_swe(df, num_list): 
    col_names = [f"mean_swe_lag_{num}" for num in num_list]
    for num, col in zip(num_list, col_names):
        df[col] = df["mean_swe"].shift(num)
    return df
  
def update_model_ready(huc_08_list, num_list = [7, 30, 60], bucket_nm = None):
    if bucket_nm is None:
        bucket_dict = sdc.create_bucket_dict("prod")
        bucket_nm = bucket_dict["model-ready"]
    for huc in huc_08_list:
        time_start = time.time()
        geos =  gg.get_geos(huc, '12')
        hucs = list(geos["huc_id"])
        for huc in hucs: 
            df = load_model_ready(huc, bucket_nm)
            df_new = add_lagged_swe(df, num_list)
            f_out = f"model_ready_huc{huc}"
            du.dat_to_s3(df_new, bucket_nm, f_out, file_type = "csv")
        du.elapsed(time_start)

    