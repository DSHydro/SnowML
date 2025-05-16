""" Module to create model_ready data using UCLA SWE instead of UA SWE """


import pandas as pd
from snowML.datapipe.utils import data_utils as du

def drop_swe_columns(df): 
    df_slim = df.loc[:, ~df.columns.str.contains('swe', case=False)]
    return df_slim

def add_lagged_swe(df, num_list): 
    col_names = [f"mean_swe_lag_{num}" for num in num_list]
    for num, col in zip(num_list, col_names):
        df[col] = df["mean_swe"].shift(num)
    return df

def load_UA(huc):
    f_UA = f"model_ready_huc{huc}.csv"
    b_mr = "snowml-model-ready"  # TO DO - Make Dynamic 
    df_UA = du.s3_to_df(f_UA, b_mr)
    df_UA.set_index("day", inplace=True)
    col_order = list(df_UA.columns)
    return df_UA, col_order

def load_UCLA(huc):
    f_UCLA = f"mean_swe_ucla_2_in_{huc}.csv"
    b_s = "snowml-gold" # TO DO - Make Dynamic 
    df_UCLA = du.s3_to_df(f_UCLA, b_s)
    df_UCLA.set_index("day", inplace=True)
    df_UCLA.rename(columns={"SWE_Post": "mean_swe"}, inplace=True)
    return df_UCLA

def process_one_huc(huc): 
    df_UA, col_order = load_UA(huc)
    df_model_slim = drop_swe_columns(df_UA)
    df_UCLA = load_UCLA(huc)
    df_model = df_UCLA.join(df_model_slim, how="inner")
    num_list = [7, 30, 60]
    df_model = add_lagged_swe(df_model, num_list)
    df_model_final = df_model[col_order]
    f_out = f"model_ready_huc{huc}_ucla"
    b_mr = "snowml-model-ready"  # TO DO - Make Dynamic 
    du.dat_to_s3(df_model_final, b_mr, f_out, file_type="csv")
    return df_model_final