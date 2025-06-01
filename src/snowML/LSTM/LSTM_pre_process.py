
# pylint: disable=C0103
""" 
This module provides functions to preprocess data for training, validation, 
and testing of an LSTM model. It includes functions for z-score normalization,
data preprocessing, tensor creation, and time-based train-test splitting.

Functions:
- z_score_normalize(df, global_means, global_stds): Normalize specified 
    columns of a DataFrame using global z-score normalization.
- pre_process(huc_list, var_list, bucket_dict=None): Preprocess data by 
    loading, normalizing, and organizing it into a dictionary.
- create_tensor(dataset, lookback, var_list): Transform time series 
    data into tensor objects for LSTM input.
- train_test_split_time(data, train_size_fraction): Split time series data 
    into training and testing sets based on time dimension.
"""

import pandas as pd
import numpy as np
import torch
from snowML.datapipe.utils import data_utils as du
from snowML.datapipe.utils import set_data_constants as sdc


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

   

    return df

def pre_process(huc_list, var_list, UCLA = False, filter_dates = None, bucket_dict=None):
    """
    Pre-processes data for LSTM model training by loading, normalizing, and 
    organizing data from multiple HUCs.

    Args:
        huc_list (list): List of Hydrologic Unit Codes (HUCs) to process.
        var_list (list): List of variable names to keep in the DataFrames.
        bucket_dict (dict, optional): Dictionary containing bucket information.
          If None, a default bucket dictionary is created.

    Returns:
        tuple: A tuple containing:
            - df_dict (dict): Dictionary where keys are HUCs and values are 
                normalized DataFrames.
            - global_means (pd.Series): Series containing the global mean 
                of each variable.
            - global_stds (pd.Series): Series containing the global standard 
                deviation of each variable.
    """
    df_dict = {}  # Initialize dictionary
    # Initialize an empty list to collect all DataFrames for global statistics calculation
    all_dfs = []

    # Step 1: Load all dataframes and collect them for global mean and std computation
    for huc in huc_list:
        df = load_df(huc, var_list, UCLA = UCLA, filter_dates = filter_dates, bucket_dict = bucket_dict)
        all_dfs.append(df)  # Collect DataFrames for global normalization
        df_dict[huc] = df  # Store DataFrame in dictionary

    #print("finished making dictionary")

    # Step 2: Calculate global mean and std for each column of interest across all HUCs
    combined_df = pd.concat(all_dfs)
    global_means = combined_df.mean()
    global_stds = combined_df.std()

    # Step 3: Normalize each DataFrame using the global mean and std
    for huc, df in df_dict.items():
        # Pass global mean and std for normalization
        df = z_score_normalize(df, global_means, global_stds)
        df_dict[huc] = df  # Store normalized DataFrame

    print(f"number of sub units for training is {len(df_dict)}")
    return df_dict, global_means, global_stds


def pre_process_separate(huc_list, var_list, UCLA = False, filter_dates = None, bucket_dict=None):
    """
    Pre-processes data for LSTM model training by loading, normalizing, and 
    organizing data from multiple HUCs. Normalize each huc only against itself.

    Args:
        huc_list (list): List of Hydrologic Unit Codes (HUCs) to process.
        var_list (list): List of variable names to keep in the DataFrames.
        bucket_dict (dict, optional): Dictionary containing bucket information.
          If None, a default bucket dictionary is created.

    Returns:
        - df_dict (dict): Dictionary where keys are HUCs and values are 
            normalized DataFrames.
    """

    df_dict = {}  # Initialize dictionary
    # Initialize an empty list to collect all DataFrames for global statistics calculation
    all_dfs = []

    # Step 1: Load all dataframes
    for huc in huc_list:
        df = load_df(huc, var_list, UCLA = UCLA, filter_dates = filter_dates, bucket_dict = bucket_dict)
        all_dfs.append(df)  # Collect DataFrames for global normalization
        df_dict[huc] = df  # Store DataFrame in dictionary

    # Step 2: Normalize each df individually
    for huc, df in df_dict.items():
        mean = df.mean()
        std = df.std()
        df = z_score_normalize(df, mean, std)
        df_dict[huc] = df  # Store normalized DataFrame

    return df_dict


def create_tensor(dataset, lookback, var_list):
    """Transform the time series into a tensor object.

    Args:
        dataset: A pandas DataFrame of time series data
        lookback: Size of window for prediction
        var_list: List of column names to be used as features
    """
    #time_start = time.time()
    X, y = [], []

    for i in range(len(dataset) - lookback):
        feature = dataset.iloc[i:(i + lookback)][var_list].values  # Select the columns in var_list
        target = np.array([dataset.iloc[i + lookback]["mean_swe"]])  # Selects "mean_swe" as target
        X.append(feature)
        y.append(target)


    # Ensure X and y are numpy arrays before converting to torch tensors
    X = np.array(X) if isinstance(X, list) else X
    y = np.array(y) if isinstance(y, list) else y

    # Convert to torch tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    #du.elapsed(time_start)
    return X_tensor, y_tensor


def train_test_split_time(data, train_size_fraction):
    """
    Splits the given time series data into training and testing sets along the 
    time dimension.

    Parameters:
        data (iterable): The time series data to be split.
        train_size_fraction (float): The fraction of the data to be used for 
            training. Should be between 0 and 1.

    Returns
        tuple: A tuple containing:
            - train_main (iterable): The training set.
            - test_main (iterable): The testing set.
            - train_size_main (int): The size of the training set.
            - test_size_main (int): The size of the testing set.
    """
    train_size_main = int(len(data) * train_size_fraction)
    test_size_main = len(data) - train_size_main
    train_main, test_main = data[:train_size_main], data[train_size_main:]
    return train_main, test_main, train_size_main, test_size_main
   