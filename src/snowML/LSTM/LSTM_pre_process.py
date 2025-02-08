
# pylint: disable=C0103

import sys
import os
import time
import pandas as pd
import numpy as np
import torch


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import data_utils as du


# Excluded Hucs Due to Missing SWE Data (Canada)
EXCLUDED_HUCS = ["1711000501", "1711000502", "1711000503"]
# Excluded Hucs Due to > 50% Ephemeral -- TO DO



def assemble_huc_list(input_pairs):
    hucs = []
    bucket_name = "shape-bronze"  # TO DO MAKE DYNAMIC
    for pair in input_pairs:
        f_name = f"Huc{pair[1]}_in_{pair[0]}.geojson"
        geos = du.s3_to_gdf(bucket_name, f_name)
        hucs.extend(geos["huc_id"].to_list())  # Use extend() to flatten
    print(f"number of sub units to process is {len(hucs)}")
    return hucs


def z_score_normalize(df):
    normalized_df = df.copy()

    for column in ["mean_pr", "mean_tair", "mean_vs", "mean_srad", "mean_rmax", "mean_rmin"]:
        column_mean = df[column].mean()
        column_std = df[column].std()
        normalized_df[column] = (df[column] - column_mean) / column_std

    return normalized_df

def pre_process (huc_list, var_list):
    df_dict = {}  # Initialize dictionary
    bucket_name = "dawgs-model-ready"  # TO DO make dynamic
    for huc in huc_list:
        if huc not in EXCLUDED_HUCS:
            file_name = f"model_ready_huc{huc}.csv"
            df = du.s3_to_df(file_name, bucket_name)
            df['day'] = pd.to_datetime(df['day'])
            df.set_index('day', inplace=True)  # Set 'day' as the index
            col_to_keep = var_list + ["mean_swe"]
            df = z_score_normalize(df)
            df = df[col_to_keep]
            df_dict[huc] = df  # Store DataFrame in dictionary
    return df_dict

def train_test_split(data, train_size_fraction):
    train_size_main = int(len(data) * train_size_fraction)
    test_size_main = len(data) - train_size_main
    train_main, test_main = data[:train_size_main], data[train_size_main:]
    return train_main, test_main, train_size_main, test_size_main

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
   