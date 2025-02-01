# pylint: disable=C0103


import s3fs
import pandas as pd
import data_utils as du
import set_data_constants as sdc


def gather_gold_files(huc_id, var_list = None, bucket_dict = None):
    # some set up
    if bucket_dict is None:
        bucket_dict = sdc.create_bucket_dict("prod")
    if var_list is None:
        var_dict = sdc.create_var_dict()
        var_list = list(var_dict.keys())

    # gather files
    pattern = "mean_{var}_in_{huc_id}.csv"
    gold_files = [pattern.format(var=var, huc_id=huc_id) for var in var_list]
    bucket_nm = bucket_dict.get("gold")
    gold_files_long = [f"{bucket_nm}/{file}" for file in gold_files]

    return gold_files_long

def clean_and_filter(df, start_date = "1983-10-01", end_date = "2022-09-30"):
    df['day'] = pd.to_datetime(df['day'])
    df_filtered = df[(df["day"] >= start_date) & (df["day"] < end_date)]
    return df_filtered

def huc_gold(huc_id, var_list = None, bucket_dict = None):

    # some set up
    if bucket_dict  is None:
        bucket_dict = sdc.create_bucket_dict("prod")
    if var_list is None:
        var_dict = sdc.create_var_dict()
        var_list = list(var_dict.keys())

    
    files = gather_gold_files(huc_id, var_list = var_list, bucket_dict = bucket_dict)
   

    # open all vars and merge into one df
    fs = s3fs.S3FileSystem()
    dfs = [pd.read_csv(fs.open(file_path)) for file_path in files]
    dfs_clean = [clean_and_filter(df) for df in dfs]

    model_df = dfs_clean[0]
    for df in dfs_clean[1:]:
        df = df[df.columns.drop("huc_id")]
        model_df = pd.merge(model_df, df, on="day", how="outer")

    # reorder columns
    new_order = ['day', 'huc_id'] + [col for col in model_df.columns if col not in ['day', 'huc_id']]
    model_df = model_df[new_order]

    # reset index
    model_df.reset_index(drop=True, inplace=True)  
    model_df.set_index("day", inplace=True)

    f_out = f"model_ready_huc{huc_id}"
    du.dat_to_s3(model_df, bucket_dict.get("model-ready"), f_out, file_type = "csv")

    return model_df



