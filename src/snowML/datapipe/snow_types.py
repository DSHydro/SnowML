# pylint: disable=C0103

"""
This module provides functions for fetching, processing, and analyzing snow 
classification data. The data is sourced from a remote NetCDF file and can be 
clipped to specific geographic regions. 

The module includes functionality to save the processed data to an S3 bucket in 
Zarr format, retrieve it from S3, and calculate snow class statistics. 
Additionally, it provides utilities to map snow class values to their 
corresponding names and classify hydrologic unit codes (HUCs) based on 
predominant snow type.

Functions:
    get_snow_class_data(geos=None): Fetches and processes snow classification 
        data from a remote NetCDF file.
    save_snow_class_data(ds, bucket_dict=None): Saves the snow class data to 
        an S3 bucket in Zarr format.
    snow_class_data_from_s3(geos=None, bucket_dict=None): Fetches snow 
        classification data from an S3 bucket and optionally clips it to a 
        specified geographic region.
    map_snow_class_names(): Maps snow class values to their corresponding names.
    calc_snow_class(ds, snow_class_names): Calculates the percentage of each 
        snow class in the dataset.
    snow_class(geos): Classifies snow data for given geographical regions.
    display_df(df): Appends an average row to the DataFrame and reorders columns.
    classify_hucs(df): Classifies HUCs based on predominant snow type.
    save_snow_types(df, huc_id): Saves a DataFrame as a markdown table to a file.
    process_all(huc_id, huc_lev, save=False): Processes snow types for a given 
        hydrologic unit code (HUC) and level.
"""

import io
import requests
import xarray as xr
import pandas as pd
import numpy as np
import s3fs
from snowML.datapipe import get_geos as gg
from snowML.datapipe import set_data_constants as sdc


def get_snow_class_data(geos = None):
    """
    Fetches and processes snow classification data from a remote NetCDF file.

    Parameters:
        geos (geopandas.GeoDataFrame, optional): A GeoDataFrame containing 
            geometries to clip the dataset to. If None, the function returns 
            data for the contiguous United States (CONUS).

    Returns:
        xarray.Dataset: The processed snow classification dataset, either 
            clipped to the provided geometries or to the CONUS region if no 
            geometries are provided.

    Notes:
        - The dataset is fetched from the NSIDC DAAC data repository.
        - The dataset is expected to be in NetCDF format and is opened 
            using the h5netcdf engine.
        - The coordinate reference system (CRS) is set to "EPSG:4326".
        - If `geos` is provided, it is reprojected to match the dataset's CRS
             before clipping.
    """

    url = "https://daacdata.apps.nsidc.org/pub/DATASETS/nsidc0768_global_seasonal_snow_classification_v01/SnowClass_NA_05km_2.50arcmin_2021_v01.0.nc"

    response = requests.get(url, timeout=10)
    ds = xr.open_dataset(io.BytesIO(response.content), engine="h5netcdf")
    if geos is None: # return the data for CONUS
        lat_min, lat_max = 24.396308, 49.384358
        lon_min, lon_max = -125.0, -66.93457
        ds_conus = ds.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
        # properly set the crs (from metadata, should be "EPSG:4326")
        ds_conus = ds_conus.rio.write_crs("EPSG:4326")
        return ds_conus
    # else return all data witin the geo
    ds = ds.rio.write_crs("EPSG:4326")
    geos = geos.to_crs(ds.rio.crs)
    ds_final = ds.rio.clip(geos.geometry, geos.crs, drop=True)
    ds.close()
    return ds_final

def save_snow_class_data(ds, bucket_dict = None):
    """
    Save the snow class data to an S3 bucket in Zarr format.

    Parameters:
        ds (xarray.Dataset): The dataset to be saved.
        bucket_dict (dict, optional): A dictionary containing S3 bucket 
                information. If None, a default bucket dictionary for "prod" 
                is created.

    Returns:
        None

    Notes:
        - Checks if the Zarr file already exists in the specified S3 path.
        - If the file exists, prints a message indicating the file's existence.
        - If the file does not exist, saves the dataset to Zarr 
    """

    if bucket_dict is None:
        bucket_dict = sdc.create_bucket_dict("prod")
    bucket = bucket_dict["bronze"]
    file_name = "snow_class_data.zarr"
    s3_path = f"s3://{bucket}/{file_name}"
    fs = s3fs.S3FileSystem()
    if fs.exists(s3_path):
        print(f"Zarr file already exists at s3://{bucket}/{s3_path}")
    else:
        ds.to_zarr(s3_path, mode="w", consolidated=True)
        print(f"Created new Zarr file at s3://{bucket}/{s3_path}")

def snow_class_data_from_s3(geos = None, bucket_dict = None):
    """
    Fetch snow classification data from an S3 bucket and optionally clip it to 
    a specified geographic region.

    Parameters:
        geos (geopandas.GeoDataFrame, optional): A GeoDataFrame containing the 
            geographic region to clip the data to. If None, the full dataset for
            CONUS (Continental United States) is returned.
        bucket_dict (dict, optional): A dictionary containing S3 bucket info. 
            If None, a default bucket dictionary for the "prod" environment is 
            created using `sdc.create_bucket_dict("prod")`.

    Returns:
        xarray.Dataset: The snow classification dataset, either for the full 
        CONUS region or clipped to the specified geographic region.
    """

    if bucket_dict is None:
        bucket_dict = sdc.create_bucket_dict("prod")
    bucket = bucket_dict["bronze"]
    zarr_store_url = f's3://{bucket}/snow_class_data.zarr'
    ds_conus = xr.open_zarr(store=zarr_store_url, consolidated=True)
    ds_conus = ds_conus.rio.write_crs("EPSG:4326")
    ds_conus.close()
    if geos is None: # return the full data for CONUS
        return ds_conus
    # if not None return all data witin the geo
    geos = geos.to_crs(ds_conus.rio.crs)
    ds_clipped = ds_conus.rio.clip(geos.geometry, geos.crs, drop=True)
    return ds_clipped

def map_snow_class_names():
    """ Function to map snow class values to their corresponding names.
    Returns:
        dict: A dictionary mapping snow class values to their names.
    """
    snow_class_names = {
        1: "Tundra",
        2: "Boreal Forest",
        3: "Maritime",
        4: "Ephemeral",
        5: "Prairie",
        6: "Montane Forest",
        7: "Ice",
        8: "Ocean"
    }
    return snow_class_names

def calc_snow_class(ds, snow_class_names):
    """
    Calculate the percentage of each snow class in the dataset.

    Parameters:
        ds (xarray.Dataset): The dataset containing the "SnowClass" array.
         snow_class_names (dict): A dictionary mapping snow class integer values 
         to their corresponding names.

    Returns:
        pd.DataFrame: A DataFrame with snow class names as columns and their 
            corresponding percentages as values.
    """

    # Flatten the SnowClass array and remove NaN values if any
    valid_pixels = ds["SnowClass"].values.flatten()
    valid_pixels = valid_pixels[~np.isnan(valid_pixels)]  # Remove NaNs if present

    # Get unique class values and their counts
    unique_classes, counts = np.unique(valid_pixels, return_counts=True)

    # Compute percentage for each class
    total_pixels = counts.sum()
    percentages = {int(cls): np.round((count / total_pixels) * 100).astype(int) for cls, count in zip(unique_classes, counts)}

    # Ensure all snow classes are included, setting missing ones to 0%
    full_percentages = {name: [percentages.get(cls, 0)] for cls, name in snow_class_names.items()}

    # Convert dictionary to a DataFrame
    df_snow_classes = pd.DataFrame(full_percentages)

    return df_snow_classes

def snow_class(geos):
    """
    Classifies snow data for given geographical regions.

    This function processes geographical data to classify snow types within 
    specified regions.It retrieves snow classification data, clips it to the 
    provided geographical regions, and calculates snow class statistics.

    Args:
        geos (GeoDataFrame): A GeoDataFrame containing geographical regions 
            with a 'huc_id' column.

    Returns:
        DataFrame: A DataFrame containing snow class statistics for each 
            region, including the 'huc_id'.

    Raises:
        Exception: If there is an error processing a specific region, it will 
            be omitted from the dataset and an error message will be printed.
    """
    results = pd.DataFrame()
    snow_class_names = map_snow_class_names()
    #ds_conus = get_snow_class_data(geos = None)
    ds_conus = snow_class_data_from_s3(geos = None)
    for i in range(geos.shape[0]):
        #print(f"processing geos {i+1} of {geos.shape[0]}")
        row = geos.iloc[[i]]
        try:
            row = row.to_crs(ds_conus.rio.crs)
            ds = ds_conus.rio.clip(row.geometry, row.crs, drop=True)
            df_snow_classes = calc_snow_class(ds, snow_class_names)
            df_snow_classes["huc_id"] = row["huc_id"].values[0]
            results = pd.concat([results, df_snow_classes], ignore_index=True)
        except Exception as e:
            print(f"Error processing HUC ID {row['huc_id'].values[0]}: {e}, omitting from dataset")
    return results

def display_df(df):
    """
    Appends an average row to the DataFrame and reorders columns.

    This function calculates the average of all columns except 'huc_id' in the 
    given DataFrame,appends this average as a new row with 'huc_id' set to 
    "Average", and reorders the columns so that 'huc_id' is the first column.

    Parameters:
        df (pandas.DataFrame): The input DataFrame with a column named 'huc_id'.

    Returns:
        pandas.DataFrame: The modified DataFrame with an appended average row 
            and reordered columns.
    """
    ave_row = df.drop(columns=['huc_id']).mean().round(1).to_frame().T
    # Add 'huc_id' to the ave_row after filtering
    ave_row['huc_id'] = "Average"
    # Concatenate the filtered average row
    df = pd.concat([df, ave_row], ignore_index=True)
    # Reordering so 'huc_id' is the first column
    df = df[['huc_id'] + [col for col in df.columns if col != 'huc_id']]
    return df


def classify_hucs(df):
    """
    Classifies HUCs (Hydrologic Unit Codes) based on predominant snow type.

    This function takes a DataFrame containing snow data for various HUCs and 
    determines the predominant snow type for each HUC. It excludes the last row 
    (assumed to be an average row) from the classification process. The function 
    returns an updated DataFrame with the predominant snow type for each HUC and 
    a dictionary containing the count of each snow class.

    Parameters:
    df (pandas.DataFrame): A DataFrame where each row represents a HUC and each 
                           column (except the first) represents a snow class.

    Returns:
    tuple: A tuple containing:
        - pandas.DataFrame: The updated DataFrame with an additional column 
                            "Predominant_Snow" indicating the predominant snow 
                            type for each HUC.
        - dict: A dictionary with snow classes as keys and their respective 
                counts as values.
    """
    # Exclude the last row (average row)
    df_without_avg = df.iloc[:-1].copy()

    # List of snow classes (excluding huc_id)
    snow_classes = df.columns[1:]

    # Determine predominant snow type for each huc_id
    df_without_avg["Predominant_Snow"] = df_without_avg[snow_classes].idxmax(axis=1)

    # Count occurrences of each snow class and convert to dictionary
    snow_class_counts = df_without_avg["Predominant_Snow"].value_counts().to_dict()

    return df_without_avg, snow_class_counts  # Return updated DataFrame and counts as a dictionary

def save_snow_types(df, huc_id):
    """
    Save a DataFrame as a markdown table to a file.

    Parameters:
    df (pandas.DataFrame): The DataFrame containing snow types data.
    huc_id (str): The HUC (Hydrologic Unit Code) id to be used in the filename.

    Returns:
    None

    The function converts the DataFrame to a markdown table and saves it to a 
    file located at '../../docs/tables/snow_types{huc_id}.md'. It also prints a 
    message indicating the location of the saved file.
    """
    markdown_table = df.to_markdown(index=False)
    with open(f'../../docs/tables/snow_types{huc_id}.md', 'w',
              encoding='utf-8') as f:
        f.write(markdown_table)
    print(f"Markdown table saved to ../../docs/tables/snow_types{huc_id}.md")

def process_all(huc_id, huc_lev, save = False):
    """
    Processes snow types for a given hydrologic unit code (HUC) and level.

    Args:
        huc_id (str): The hydrologic unit code identifier.
        huc_lev (int): The level of the hydrologic unit code.
        save (bool, optional): If True, saves the predominant snow types. 
            Defaults to False.

    Returns:
        tuple: A tuple containing:
            - df_snow_types (DataFrame): DataFrame containing snow types.
            - snow_class_counts (DataFrame): DataFrame containing counts of 
                each snow class.
            - df_predominant (DataFrame): DataFrame containing predominant 
                snow types.
    """
    #geos = du.get_basin_geos(f"Huc{huc_lev}", huc_id)
    geos = gg.get_geos(huc_id, huc_lev)
    df_snow_types = snow_class(geos)
    df_snow_types = display_df(df_snow_types)
    df_predominant, snow_class_counts = classify_hucs(df_snow_types)
    if save:
        save_snow_types(df_predominant, huc_id)
    return df_snow_types, snow_class_counts, df_predominant

def color_map_standard(): 
    # Create a color map for the "Predominant_Snow" column with specific colors
    color_map_snow = {
        "Montane Forest": "darkgreen",  
        "Maritime": "blue",
        "Ephemeral": "#E6E6FA",  # Hex code for lavender
        "Prairie": "lightgreen", 
        "Tundra": "gray"
    }
    return color_map_snow


