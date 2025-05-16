""" Pre-Process Module for Local Trained LSTM Model """ 

import pandas as pd
from snowML.datapipe.utils import data_utils as du


# Move to shared 
def load_df(huc, var_list, UCLA = False, filter_dates = None, bucket_dict=None):
    if bucket_dict is None:
        bucket_dict = sdc.create_bucket_dict("prod")
    bucket_name = bucket_dict["model-ready"]

    if UCLA: 
        file_name = f"model_ready_huc{huc}_ucla.csv"
    else: 
        file_name = f"model_ready_huc{huc}.csv"
    df = du.s3_to_df(file_name, bucket_name)
    df['day'] = pd.to_datetime(df['day'])
    df.set_index('day', inplace=True)  # Set 'day' as the index

    # Collect only the columns of interest
    col_to_keep = var_list + ["mean_swe"]

    for col in col_to_keep:
        if col not in df.columns:
            print(f"huf{huc} is missing col {col}")

    df = df[col_to_keep]

    startrow = df.shape[0]
    df = df[col_to_keep].dropna()
    num_dropped = startrow - df.shape[0]
    if num_dropped > 0:
        print(f"Number of rows dropped: {num_dropped}")

    if filter_dates is not None:
        df = df.loc[filter_dates[0]:filter_dates[1]]

    return df
    
# Move to shared 
def z_score_normalize(df, global_means, global_stds):
    """
    Normalize the specified columns of a DataFrame using global z-score normalization.

    Parameters:
    df (pandas.DataFrame): The input DataFrame containing the data to be normalized.
    global_means (pandas.Series): The global means for each column to be normalized.
    global_stds (pandas.Series): The global standard deviations for each column to be normalized.

    Returns:
    pandas.DataFrame: A new DataFrame with the specified columns normalized using global 
                    z-score normalization.
    """
    normalized_df = df.copy()
    df_cols = df.columns

    columns_to_normalize = ["mean_pr", "mean_tair", "mean_vs",
            "mean_srad", "mean_hum", "Mean Elevation", "Mean Forest Cover"]

    for column in columns_to_normalize:
        if column in df_cols:
            normalized_df[column] = (df[column] - global_means[column]) / global_stds[column]

    return normalized_df


# normalize the data and create train/test split - one huc for local training
def pre_process (huc, params, bucket_dict=None):
    df = load_df(huc, params["var_list"], UCLA = params["UCLA"], filter_dates = params["filter_dates"], bucket_dict = bucket_dict)
    mean = df.mean()
    std = df.std()
    df = z_score_normalize(df, mean, std)
    train_size_frac = params["train_size_fraction"]
    df_train, _, _, _ = train_test_split_time(df, train_size_frac)
    return df_dict