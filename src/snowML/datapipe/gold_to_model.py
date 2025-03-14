# pylint: disable=C0103
""" 
This module provides functions to gather, clean, and process hydrologic data 
for modeling purposes. 

Functions:
    gather_gold_files(huc_id, var_list=None, bucket_dict=None)
        Generates a list of file paths for gold data files based 
        on the provided HUC ID and variable list.

    clean_and_filter(df, start_date="1983-10-01", end_date="2022-09-30")
        Cleans and filters the input DataFrame based on the 
        specified date range.

     huc_model_wrf(huc_id, bucket_dict, var_list=None):
            Generates a DataFrame for a given HUC by merging and processing data 
            from multiple sources related to meteorological data.

    huc_model(huc_id, var_list=None, bucket_dict=None, overwrite_mod=False):
            Processes data for a given HUC and prepares it for modeling.
"""

import s3fs
import pandas as pd
from snowML.datapipe import data_utils as du
from snowML.datapipe import set_data_constants as sdc
from snowML.datapipe import snow_types as st
from snowML.datapipe import get_dem as gd


def gather_gold_files(huc_id, var_list = None, bucket_dict = None):
    """
    Gather gold files based on HUC ID and variable list.

    This function generates a list of file paths for gold data files based on 
    the provided HUC ID and variable list. If no variable list or bucket 
    dictionary is provided, default values are used.

    Args:
        huc_id (str): The Hydrologic Unit Code (HUC) ID to identify the region.
        var_list (list, optional): A list of variables to include in the file names. 
                Defaults to None, which will use a default variable list.
        bucket_dict (dict, optional): A dictionary mapping bucket names to their paths. 
                Defaults to None, which will use a default bucket dictionary.

    Returns:
        list: A list of file paths for the gold data files.
    """
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
    """
    Cleans and filters the input DataFrame based on the specified date range.

    Parameters:
        df (pandas.DataFrame): The input DataFrame containing a 'day' column 
            with date values.
        start_date (str): The start date for filtering the DataFrame 
            in the format 'YYYY-MM-DD'. Default is "1983-10-01".
        end_date (str): The end date for filtering the DataFrame 
            in the format 'YYYY-MM-DD'. Default is "2022-09-30".

        Returns:
            pandas.DataFrame: The cleaned and filtered DataFrame. The 'day' 
            column is converted to datetime format, and the DataFrame include 
            only rows where the 'day' value falls within the specified date 
            range. The 'huc_id' column is dropped from the DataFrame. 
    """
    df['day'] = pd.to_datetime(df['day'])
    df =  df[(df["day"] >= start_date) & (df["day"] < end_date)]
    df = df[df.columns.drop("huc_id")]
    return df


def huc_model_wrf(huc_id, bucket_dict, var_list = None):
    """
    Generate a DataFrame for a given HUC (Hydrologic Unit Code) by merging 
    and processing data from multiple sources.

    Parameters:
        huc_id (str): The Hydrologic Unit Code identifier.
        bucket_dict (dict): Dictionary containing bucket information for 
            accessing data.
        var_list (list, optional): List of variables to include in the model. If 
            None, a default list is created.

        Returns:
        pd.DataFrame: A DataFrame containing the processed and merged data 
            for the specified HUC.
    """

    # some set up
    if var_list is None:
        var_dict = sdc.create_var_dict()
        var_list = list(var_dict.keys())


    files = gather_gold_files(
        huc_id,
        var_list = var_list,
        bucket_dict = bucket_dict)
    #print(files)

    # open all vars and merge into one df
    fs = s3fs.S3FileSystem()
    dfs = [pd.read_csv(fs.open(file_path)) for file_path in files]
    dfs_clean = [clean_and_filter(df) for df in dfs]

    model_df = dfs_clean[0]
    for df in dfs_clean[1:]:
        model_df = pd.merge(model_df, df, on="day", how="outer")

    # update units & columns
    model_df["mean_swe"] = model_df["mean_swe"] / 1000  #set units to be mm
    model_df["mean_tair"] = model_df["mean_tmmx"]/2 +  model_df["mean_tmmn"] / 2
    model_df["mean_tair"] = model_df["mean_tair"] - 273.15  # set units to be C
    model_df = model_df[model_df.columns.drop(["mean_tmmx", "mean_tmmn"])]
    model_df["mean_rmax"] = model_df["mean_rmax"] / 100  # set units to be %
    model_df["mean_rmin"] = model_df["mean_rmin"] / 100  # set units to be %
    model_df["mean_hum"] = model_df["mean_rmax"]/2 +  model_df["mean_rmin"] / 2


    # reset index
    model_df.reset_index(drop=True, inplace=True)
    model_df.set_index("day", inplace=True)

    # select and reorder columns
    new_order = ["mean_pr", "mean_tair", "mean_vs", "mean_srad", "mean_hum", "mean_swe"]
    model_df = model_df[new_order]

    return model_df

def huc_model(huc_id, var_list = None, bucket_dict = None, overwrite_mod = False):
    """
    Processes data for a given HUC (Hydrologic Unit Code) and prepares it 
        for modeling.

    Parameters:
        huc_id (int): The Hydrologic Unit Code identifier.
        var_list (list, optional): List of variables to include in the model. 
            Defaults to None.
        bucket_dict (dict, optional): Dictionary containing S3 bucket information. Defaults to None.
        overwrite_mod (bool, optional): Flag to overwrite existing model data. 
            Defaults to False.

    Returns:
        pd.DataFrame: The final DataFrame ready for modeling, or None if the 
            data already exists and overwrite_mod is False.
     """
    # some set up
    if bucket_dict  is None:
        bucket_dict = sdc.create_bucket_dict("prod")

    f_out = f"model_ready_huc{huc_id}"
    if du.isin_s3(bucket_dict.get("model-ready"), f"{f_out}.csv") and not overwrite_mod:
        print(f"{f_out} already exists, skipping processing")
        return None

    model_df = huc_model_wrf(huc_id, bucket_dict, var_list = var_list)
    huc_lev = str(len(str(huc_id))).zfill(2)

    # add mean elevation for huc to model_df
    mean_elevation = gd.process_dem_all(huc_id, huc_lev, plot = False)
    model_df['Mean Elevation'] = mean_elevation

    # add snow_types for huc
    snow_types, _, _ = st.process_all(huc_id, huc_lev)
    snow_types = snow_types.loc[[0]].copy()
    # Broadcasting the values from snow_types to model_df
    snow_types_broadcasted = pd.DataFrame([snow_types.iloc[0]] * len(snow_types),
            columns=snow_types.columns, index=model_df.index)
    df_final = pd.concat([model_df, snow_types_broadcasted], axis=1)
    du.dat_to_s3(df_final, bucket_dict.get("model-ready"), f_out, file_type = "csv")
    return df_final
