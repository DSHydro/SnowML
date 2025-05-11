

# pylint: disable=C0103
"""
Module provides functions to download and process climate data from a
specified URL and save it to an S3 bucket in Zarr format. Includes functions to 
handle authentication, process datasets, and manage progress tracking. 

Functions:
    url_to_ds(url, requires_auth=False, username=None, password=None, timeout=60):

    download_year(var, year):
        Downloads data for a given variable and year.

    process_year(ds, var):

    download_multiple_years(start_year, end_year, var, s3_bucket, 
                append_to=False):
        Downloads and processes data for multiple years, saving the results to a 
        Zarr file on S3.

    get_bronze(var, bronze_bucket_nm, year_start=1995, year_end=2023, 
         append_to=False):
"""

import warnings
import time
import io
import os
import json
import s3fs
import requests
import xarray as xr
from snowML.datapipe.utils import data_utils as du
from snowML.datapipe.utils import set_data_constants as sdc

# define constants
VAR_DICT = sdc.create_var_dict()


def url_to_ds(url, requires_auth=False, username=None, password=None, timeout=60):
    """
    Load data from a URL into an xarray Dataset.

    Parameters:
    url (str): The URL to fetch the data from.
    requires_auth (bool): Whether authentication is required. Default is False.
    username (str): The username for authentication, if required. Default is None.
    password (str): The password for authentication, if required. Default is None.
    timeout (int): The timeout for the request in seconds. Default is 60.

    Returns:
    xarray.Dataset: The dataset loaded from the URL, or None if the request failed.
    """
    # Prepare authentication if required
    auth = (username, password) if requires_auth else None

    try:
        response = requests.get(url, auth=auth, timeout=timeout)

        # Check if the request was successful
        if response.status_code == 200:
            # Convert the raw response content to a file-like object
            file_like_object = io.BytesIO(response.content)

            # Open the dataset from the file-like object
            ds = xr.open_dataset(file_like_object, engine="h5netcdf", chunks={"day": -1, "lat": None, "lon": None})

            return ds

        print(f"Failed to fetch data. Status code: {response.status_code}")
    except requests.exceptions.Timeout:
        print(f"Request timed out after {timeout} seconds.")
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")

    return None


def download_year(var, year):
    """
    Downloads data for a given variable and year. Fetches the relevant
    url pattern from data_utils module based on the specified var. Review dawgs
    pipeline documentation for an explanation of datasources."

    Args:
        var (str): The variable for which data is to be downloaded.
        year (int): The year for which data is to be downloaded.

    Returns:
        xarray.Dataset: The dataset containing the downloaded data.
    """
    url_pattern = du.get_url_pattern(var)
    url = url_pattern.format(year=year)
    print(f"Downloading {url}")
    ds = url_to_ds(url)
    print(ds)
    return ds

def process_year(ds, var):
    """
    Processes a dataset for a given year and variable.

    Parameters:
        ds (xarray.Dataset): The dataset to be processed.
        var (str): The variable to process. 

    Returns:
        xarray.Dataset: The processed dataset with sorted latand lon coords.
        If the variable is "swe", the "SWE" var long name will be reset to "swe"
        and the "DEPTH" variable will be dropped from the dataset.
    """
    if var == "swe":
        ds = ds.rename({"time": "day"})
        ds = ds["SWE"] # drop DEPTH variable from SWE Dataset
    if not ds['lat'].to_index().is_monotonic_increasing:
        ds = ds.sortby("lat")
    if not ds['lon'].to_index().is_monotonic_increasing:
        ds = ds.sortby("lon")

    return ds

def download_multiple_years(
        start_year,
        end_year,
        var,
        s3_bucket,
        append_to=False):

    """
    Downloads and processes data for multiple years, saving the results to a 
    Zarr file on S3.

    Parameters:
        start_year (int): The starting year of the range to download.
        end_year (int): The ending year of the range to download.
        var (str): The variable name to download and process.
        s3_bucket (str): The name of the S3 bucket to store Zarr file.
        append_to (bool, optional): If True, append to an existing Zarr file. 
            If False, create a new Zarr file. Default is False.

        Returns:
            str: The S3 path to the saved Zarr file.

        Raises:
            ValueError: If the S3 Zarr path exists and append_to is False.

        Notes:
        - The function tracks completed years using a local progress file.
        - Data is processed and saved year by year, either creating a new Zarr 
                file or appending to an existing one.
        """

    time_start = time.time()
    s3_path = f"{var}_all.zarr"

    # Initialize the S3 filesystem
    fs = s3fs.S3FileSystem()

    # Check if the S3 Zarr path already exists
    if fs.exists(f"s3://{s3_bucket}/{s3_path}") and not append_to:
        print(f"Warning: The path s3://{s3_bucket}/{s3_path} already exists.")
        print("Skipping file. Set append_to = True if you intended to append.")
        return "no path created"

    # define some data-specific attributes
    dim_to_concat = "day"

    # Load progress from a local file to keep track of completed years
    progress_file = f"{var}_progress.json"
    completed_years = set()
    if os.path.exists(progress_file):
        with open(progress_file, "r", encoding="utf-8") as f:
            completed_years = set(json.load(f))
    print(f"Resuming with completed years: {sorted(completed_years)}")

    # Process years sequentially
    for year in range(start_year, end_year + 1):

        if year in completed_years:
            print(f"Skipping year {year} (already processed)")
            continue

        print(f"Processing year: {year}")
        ds = download_year(var, year)
        ds = process_year(ds, var)
        ds = ds.chunk({dim_to_concat: -1, "lat": None, "lon": None})

        # Append to the existing Zarr file on S3
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            if not fs.exists(f"s3://{s3_bucket}/{s3_path}"):
                # Create a new Zarr file for the first year
                ds.to_zarr(f"s3://{s3_bucket}/{s3_path}", mode="w", consolidated=True)
                print(f"Created new Zarr file at s3://{s3_bucket}/{s3_path}")
                completed_years.add(year)
            else:
                # Append data to the existing Zarr file
                ds.to_zarr(f"s3://{s3_bucket}/{s3_path}", mode="a",
                           append_dim=dim_to_concat, consolidated=True)
                print(f"Appended year {year} to s3://{s3_bucket}/{s3_path}")
                completed_years.add(year)
                with open(progress_file, "w", encoding="utf-8") as f:
                    json.dump(sorted(completed_years), f)
                du.elapsed(time_start)

    #du.elapsed(time_start)
    print(f"Final dataset saved to s3://{s3_bucket}/{s3_path}")
    return s3_path

def get_bronze(var,
               bronze_bucket_nm,
               year_start = 1995,
               year_end =  2024,
               append_to = False):
    """
    Downloads raw data for the specified variable and saves it to an S3 bucket.

    Parameters:
    var (str): The variable for which data is to be downloaded.
    bronze_bucket_nm (str): The name of the S3 bucket where the data will
        be saved.
    year_start (int, optional): The starting year for the data download. 
        Must be >= 1983. Default is 1995.
    year_end (int, optional): The ending year for the data download. 
        Must be <= 2025. Default is 2023.
    append_to (bool, optional): If True, appends data to the 
        existing data in the S3 bucket. Default is False.

    Returns:
    str: The S3 path where the data has been saved.

    Raises:
    ValueError: If year_start is less than 1983 or year_end is greater than 2025.
    """
    # Validate year input
    if year_start < 1983 or year_end > 2025:
        raise ValueError("Year start must be >= 1983; year end must be < 2025")

    # download raw and save to S3 directly
    s3_path = download_multiple_years(year_start,
                                      year_end,
                                      var,
                                      bronze_bucket_nm,
                                      append_to = append_to)

    return s3_path


def add_partial_year(year, var, bronze_bucket_nm): 
    ds = download_year(var, year)
    ds = process_year(ds, var)
    dim_to_concat = "day"
    ds = ds.chunk({dim_to_concat: 365, "lat": 97, "lon": 231})
    s3_path = f"{var}_all.zarr"
    ds.to_zarr(f"s3://{bronze_bucket_nm}/{s3_path}", mode="a",
                           append_dim=dim_to_concat, consolidated=True)
    print(f"Appended year {year} to s3://{bronze_bucket_nm}/{s3_path}")
    return ds

# update bronze data with year 2024 and 2025 
def update_bronze(var, bucket_nm = None): 
    if bucket_nm is None: 
        bucket_dict = sdc.create_bucket_dict("prod")
        bucket_nm = bucket_dict["bronze"]
    # Add data for 2024
    get_bronze(var, bucket_nm, year_start = 2024, year_end = 2024, append_to=True)
    # Add data for 2025 partial year 
    add_partial_year(2025, var, bucket_nm) 

    