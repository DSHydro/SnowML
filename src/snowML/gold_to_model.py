# pylint: disable=C0103

import re
import s3fs
import pandas as pd
import data_utils as du
import set_data_constants as sdc

def gather_s3_files(var, huc_id_list, bucket_dict = None):
    # define bucket
    if bucket_dict is None:
        bucket_dict = sdc.create_bucket_dict("prod")
    bucket_nm = bucket_dict.get("gold")
    
    # files we are expecting
    f_out_dict = sdc.create_f_out_dict()
    f_out_pat = f_out_dict.get("gold")
    gold_files = [f_out_pat.format(var=var, huc_id=huc_id) for huc_id in huc_id_list]
    gold_files_long = [f"{bucket_nm}/{file}" for file in gold_files]
    s3 = s3fs.S3FileSystem(anon=False)
    s3_files = s3.ls(bucket_nm)
    matching_files = [f for f in gold_files_long if f in s3_files]
    #print(f"Matching files are: {matching_files}")
    missing_files = [f for f in gold_files_long if f not in s3_files]
    if missing_files:
        raise ValueError(f"Missing gold files: {missing_files}")
    return matching_files


def get_gold_var (files):
    """
    Reads and concatenates CSV files from S3 into a single DataFrame.

    Parameters:
    files (list): A list of S3 file URIs to read.

    Returns:
    pd.DataFrame: A DataFrame containing the concatenated data from all 
    the CSV files.
    """
    results = pd.DataFrame()
    s3 = s3fs.S3FileSystem(anon=False)
    for f in files:
        #s3_path = f"s3://{f}"
        with s3.open(f, 'rb') as file:
            df = pd.read_csv(file)
        results = pd.concat([results, df])
    return results

def merge_data(df_list):
    """
    Merges a list of DataFrames on the 'day' and 'huc_id' columns using 
    an outer join.

    Parameters:
    df_list (list): A list of DataFrames to merge.

    Returns:
    pd.DataFrame: A single DataFrame resulting from the merge.
    """
    merged_df = df_list[0]
    for df in df_list[1:]:
        merged_df = pd.merge(merged_df, df, on=['day', 'huc_id'], how='outer')
    return merged_df

def get_gold_all_df(var_list, huc_id_list, bucket_dict):
    df_list = []
    for var in var_list:
        print(f"gathering data for {var} . . .")
        files = gather_s3_files(var, huc_id_list, bucket_dict)
        new_df = get_gold_var(files)
        df_list.append(new_df)
        # print(new_df.head())    
    gold_all_df = merge_data(df_list)
    return gold_all_df

def clean_and_filter(df, start_date):
    if not isinstance(start_date, str) or not re.match(r"\d{4}-\d{2}-\d{2}", start_date):
        raise ValueError("start_date must be a string in the format 'YYYY-MM-DD'")
    if pd.to_datetime(start_date) < pd.to_datetime("1996-10-01"):
        raise ValueError("start_date must be on or after '1996-10-01'")
    # Ensure the 'time' column is in datetime format and make it the index
    # Also filter by star date
    df['day'] = pd.to_datetime(df['day'])
    df = df[df['day'] >= start_date]
    #df = df.reset_index('day')
    return df


def get_model_ready(huc_id, huc_lev, var_list, bucket_dict = None, start_date = "1996-10-01"):
    """
    Gathers, processes, and merges gold data from S3 for the specified variables, 
    HUC level, and HUC ID, and uploads the resulting DataFrame to S3.

    Parameters:
    huc_id (str): The HUC ID to filter files by.
    huc_lev (str): The HUC level as a string (e.g., "HUC12").
    var_list (list): A list of variables to gather data for (e.g., ["swe", "precip"]).
    bucket_dict (dict): A dictionary mapping variable names to S3 bucket names.

    Returns:
    pd.DataFrame: The merged DataFrame ready for modeling.

    """
    if bucket_dict is None:
        bucket_dict = sdc.create_bucket_dict("prod")

    geos = du.get_basin_geos(huc_lev, huc_id)
    huc_id_list = geos["huc_id"].unique().tolist()
    gold_all_df = get_gold_all_df(var_list, huc_id_list, bucket_dict)
    gold_all_df = clean_and_filter(gold_all_df, start_date)
    f_out = f"model_ready_{huc_lev}_in_{huc_id}"
    du.dat_to_s3(gold_all_df, bucket_dict.get("model-ready"), f_out, file_type = "csv")
    return gold_all_df
