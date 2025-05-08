""" Module to Retrieve Lidar Data """
# pylint: disable=C0103

import re
from datetime import datetime
import rioxarray as rxr
import xarray as xr
import earthaccess
import numpy as np
import pandas as pd
import s3fs
import matplotlib.pyplot as plt
from snowML.datapipe.utils import set_data_constants as sdc
from snowML.datapipe.utils import get_geos as gg


def get_files():
    """
    Searches for and downloads LiDAR data files using the Earthdata API.

    This function searches for data files with the specified short name 
    ("ASO_50M_SWE") using the Earthdata API, downloads the results into 
    a specified folder ("lidar_data"), and returns the list of downloaded 
    files.

    Returns:
        list: A list of file paths to the downloaded LiDAR data files.
    """
    data_name = "ASO_50M_SWE"
    folder_name = 'lidar_data'
    results = earthaccess.search_data(short_name=data_name)
    files = earthaccess.download(results, folder_name)
    return files

def get_one_timeslice(file):
    """
    Extracts a single time slice from a given raster file and converts it 
    into an xarray Dataset.

    Args:
        file (str): Path to the raster file to be processed.

    Returns:
        xarray.Dataset: A dataset containing the single time slice with 
        the variable named "SWE".
    """
    print("Warning - have you verified the EPSG for this prefix?")
    swe_xr = rxr.open_rasterio(file, masked=True).squeeze()
    swe_ds = swe_xr.to_dataset(name="SWE")
    # reproject and sort
    #swe_ds.rio.write_crs("EPSG:32611", inplace=True)  # TO DO - dynamically modify per prefix
    ds_re = swe_ds.rio.reproject("EPSG:4326")
    ds_re = ds_re.sortby(['x', 'y'])
    return ds_re


def create_data_dict_2(files, prefix):
    """
    Creates a dictionary mapping date strings to file paths from a list of file
    names using a specified prefix.

    This function processes a list of file names, extracts date strings in the 
    format 'YYYYMMDD' from file names containing the pattern '<prefix>_<YYYYMMDD>',
    and stores them as keys in a dictionary with the corresponding file paths
    as values.

    Args:
        files (list of str): A list of file paths or file names to process.
        prefix (str): The prefix to search for before the date string.

    Returns:
        dict: A dictionary where the keys are date strings (YYYYMMDD) extracted 
            from the file names, and the values are the file paths.
    """
    date_dict = {}
    pattern = rf'{re.escape(prefix)}_(\d{{8}})'  # dynamically build regex with escaped prefix
    for file in files:
        match = re.search(pattern, file)
        if match:
            date_string = match.group(1)
            date_dict[date_string] = file
    return date_dict



def create_data_dict(files):
    """
    Creates a dictionary mapping date strings to file paths from a list of file
    names.

    This function processes a list of file names, extracts date strings in the 
    format'YYYYMMDD' from file names containing the pattern 'USCATB_<YYYYMMDD>',
    and stores them as keys in a dictionary with the corresponding file paths
    as values.

    Args:
        files (list of str): A list of file paths or file names to process.

    Returns:
        dict: A dictionary where the keys are date strings (YYYYMMDD) extracted 
            from the file names, and the values are the file paths.
    """
    date_dict = {}
    for file in files:
        # Use a regular expression to extract the date string after "USCATB_"
        match = re.search(r'USCATB_(\d{8})', file)
        if match:
            date_string = match.group(1)  # The date string (YYYYMMDD)
            date_dict[date_string] = file  # Store the date as key and file path as value
    return date_dict


def concatenate_timeslices(date_dict):
    datasets = []
    for date, file in date_dict.items():
        swe_ds = get_one_timeslice(file)
        swe_ds = swe_ds.expand_dims(day=[date])
        datasets.append(swe_ds)

    # Concatenate all the datasets along the 'day' dimension
    final_dataset = xr.concat(datasets, dim="day")

    return final_dataset


def get_lidar_all(prefix, bucket_name = None):
    """
    Retrieves, processes, and stores LiDAR data as a Zarr file in an S3 
        bucket.

    This function logs into the Earthdata system, retrieves LiDAR data 
    files, organizes them by date, concatenates the data into a single 
    dataset, and  saves the dataset as a Zarr file in the specified S3 
    bucket. If no bucket name is provided, it defaults to using the 
    "bronze" bucket in the "prod" environment.

    Args:
        bucket_name (str, optional): The name of the S3 bucket where the 
            Zarr file will be stored. Defaults to None, in which case the 
            "bronze" bucket is used.

    Returns:
        xarray.Dataset: The concatenated LiDAR dataset.

    Raises:
        Exception: If there are issues with logging in, retrieving files, 
            processing data, or saving to S3.

    Side Effects:
        - Logs into the Earthdata system.
        - Creates a new Zarr file in the specified S3 bucket.
        - Prints the location of the created Zarr file.
    """
    earthaccess.login()
    files = get_files()
    date_dict = create_data_dict_2(files, prefix)
    print(date_dict)
    ds = concatenate_timeslices(date_dict)
    s3_path = f"lidar_{prefix}.zarr"
    if bucket_name is None:
        bucket_dict = sdc.create_bucket_dict("prod")
        bucket_name = bucket_dict["bronze"]

    fs = s3fs.S3FileSystem()
    file_path = f"s3://{bucket_name}/{s3_path}"
    if not fs.exists(file_path):
        ds.to_zarr(file_path, mode="w", consolidated=True)
        print(f"Created new Zarr file at {file_path}")
    else:
        print(f"The file already exists at {file_path}. Skipping the save.")
    return ds

def subset_by_row(ds, row): 
    ds.rio.write_crs(row.crs, inplace=True)
    clipped_data = ds.rio.clip(row.geometry, drop=True) 
    return clipped_data

def subset_by_huc(ds, huc_id): 
    huc_lev = str(len(str(huc_id))).zfill(2)
    row = gg.get_geos(huc_id, huc_lev)
    clipped_data = subset_by_row(ds, row)
    return clipped_data
    
def count_na(ds, day = "All", quiet = False):
    total_values = np.prod(ds['SWE'].shape)
    nan_count = np.isnan(ds['SWE'].values).sum()
    nan_percent = (nan_count / total_values) * 100
    if not quiet: 
        print(f"for day {day}:")
        print(f"Number of NaN values: {nan_count}")
        print(f"Percent of NaN values: {nan_percent:.2f}%")
    return nan_percent 

def plot_valid_pixels(ds):
    """
    Plot grid of valid SWE pixels in red on a white background.

    Parameters:
    -----------
    ds : xarray.Dataset
        Input Dataset with variables including 'SWE' and dimensions ('day', 'y', 'x').
    """
    # Step 1: Create a mask where SWE is not NaN for any day
    valid_mask = ds['SWE'].notnull().any(dim='day')

    # Step 2: Create meshgrid for lon and lat
    lon, lat = np.meshgrid(ds['x'].values, ds['y'].values)

    # Step 3: Plot
    plt.figure(figsize=(10, 10))
    plt.pcolormesh(lon, lat, valid_mask, cmap='Reds', shading='auto', vmin=0, vmax=1)
    plt.gca().set_facecolor('white')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Pixels with SWE Observations (Red)')
    plt.axis('equal')  # Keep aspect ratio
    plt.xticks([])
    plt.yticks([])
    plt.show()

def calc_mean(ds):
    """
    Calculate mean SWE for each day from an xarray Dataset.

    Parameters:
    -----------
    ds : xarray.Dataset
        Input Dataset with variables including 'SWE' and dimensions ('day', 'y', 'x').

    Returns:
    --------
    pd.DataFrame
        DataFrame with 'day' as datetime index and one column 'mean_swe'.
    """
    # Take mean over 'y' and 'x', skipping NaNs
    mean_swe = ds['SWE'].mean(dim=['y', 'x'], skipna=True)

    # Step 2: Convert to DataFrame & Clean Up 
    df = mean_swe.to_dataframe().reset_index()
    df = df.rename(columns={'SWE': 'mean_swe_lidar'})
    df['day'] = pd.to_datetime(df['day'], format='%Y%m%d')
    df = df.drop(columns=[col for col in ['band', 'spatial_ref'] if col in df.columns])
    df = df.set_index('day')

    return df


def add_day_dimension(ds: xr.DataArray | xr.Dataset, day_str: str) -> xr.DataArray | xr.Dataset:
    
    # Expand with a new dimension called "day"
    day = np.datetime64(pd.to_datetime(day_str), 'ns')
    
    # Expand with a new dimension called "day"
    ds_expanded = ds.expand_dims({"day": [day]})
    
    return ds_expanded

def extract_date_from_filename(filename: str) -> str:
    # Look for a pattern like '2020May21' in the filename
    match = re.search(r'(\d{4}[A-Za-z]{3}\d{2})', filename)
    if not match:
        raise ValueError("Date string in format 'YYYYMonDD' not found in filename.")

    date_str = match.group(1)
    # Convert to datetime object
    date_obj = datetime.strptime(date_str, "%Y%b%d")
    
    # Return in 'YYYY-MM-DD' format
    return date_obj.strftime("%Y-%m-%d")


