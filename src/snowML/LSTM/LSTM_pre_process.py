
# pylint: disable=C0103
# module to assemble the test, validation, and training data for the LSTM module

import pandas as pd
import numpy as np
import torch
from snowML.datapipe import data_utils as du
from snowML.datapipe import get_geos as gg
from snowML.datapipe import set_data_constants as sdc
from snowML.datapipe import snow_types as st



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

    columns_to_normalize = ["mean_pr", "mean_tair", "mean_vs", "mean_srad", "mean_hum", "Mean Elevation"]

    for column in columns_to_normalize:
        if column in df_cols:
            normalized_df[column] = (df[column] - global_means[column]) / global_stds[column]

    return normalized_df

def pre_process(huc_list, var_list, bucket_dict=None):
    df_dict = {}  # Initialize dictionary
    if bucket_dict is None:
        bucket_dict = sdc.create_bucket_dict("prod")
    bucket_name = bucket_dict["model-ready"]

    # Initialize an empty list to collect all DataFrames for global statistics calculation
    all_dfs = []

    # Step 1: Load all dataframes and collect them for global mean and std computation
    for huc in huc_list:
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
        all_dfs.append(df)  # Collect DataFrames for global normalization
        df_dict[huc] = df  # Store DataFrame in dictionary

    # Step 2: Calculate global mean and std for each column of interest across all HUCs
    combined_df = pd.concat(all_dfs)
    global_means = combined_df.mean()
    global_stds = combined_df.std()

    # Step 3: Normalize each DataFrame using the global mean and std
    for huc, df in df_dict.items():
        df = z_score_normalize(df, global_means, global_stds)  # Pass global mean and std for normalization
        df_dict[huc] = df  # Store normalized DataFrame

    print(f"number of sub units for training is {len(df_dict)}")
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
   