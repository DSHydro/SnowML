# Module to download raw data to bronze bucket
# pylint: disable=C0103

import warnings
import time
import shutil
import io
import os
import s3fs
import requests
import xarray as xr
import data_utils as du

def url_to_ds(url,requires_auth=False, username=None, password=None):
    """ Direct load from url to xarray"""

    # Prepare authentication if required
    auth = (username, password) if requires_auth else None

    response = requests.get(url, auth=auth)

    # Check if the request was successful
    if response.status_code == 200:
        # Convert the raw response content to a file-like object
        file_like_object = io.BytesIO(response.content)

        # Open the dataset from the file-like object
        ds = xr.open_dataset(file_like_object)

        return ds

    print(f"Failed to fetch data. Status code: {response.status_code}")
    return None


def download_year(var, year):
    url_pattern = du.get_url_pattern(var)
    url = url_pattern.format(year=year)
    print(f"Downloading {url}")
    ds = url_to_ds(url)
    return ds

def download_multiple_years(start_year, end_year, var):  
    time_start = time.time()
    zarr_store = f"{var}_all.zarr"

    #zarr_store = f"s3://{bucket}/{var}_all.zarr"

    if os.path.exists(zarr_store):
        raise ValueError(f"Zarr file {zarr_store} already exists. Please delete it first.")

    # define some data-specific attributes
    if var == "swe":
        dim_to_concat = "time"
    else:
        dim_to_concat = "day"

    # Process years sequentially
    for year in range(start_year, end_year + 1):
        print(f"Processing year: {year}")
        # download the file
        ds = download_year(var, year)
        if var == "swe":
            ds = ds["SWE"] # drop DEPTH variable from SWE Dataset
        ds_rechunked = ds.chunk({dim_to_concat: -1, "lat": 50, "lon": 50})
        # Append to the existing Zarr file
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            if not os.path.exists(zarr_store):
                # Create a new Zarr file for the first year
                ds_rechunked.to_zarr(zarr_store, mode="w")
                print(f"Created new Zarr file at {zarr_store}")
            else:
                # Append data to the existing Zarr file
                ds_rechunked.to_zarr(zarr_store, mode="a", append_dim=dim_to_concat)
                print(f"Appended year {year} to {zarr_store}")
        du.elapsed(time_start)

    print(f"Final dataset saved to {zarr_store}")
    return zarr_store

def upload_zarr_to_s3(zarr_path, s3_bucket, s3_path=None):
    """
    Upload a Zarr dataset to an S3 bucket.
    
    Args:
        zarr_path (str): Path to the local Zarr directory or file.
        s3_bucket (str): Name of the S3 bucket.
        s3_path (str, optional): Path in the S3 bucket where the Zarr file will be uploaded. 
                                  If None, it uses the name of the local Zarr file.
    """
    # Initialize the S3 filesystem
    fs = s3fs.S3FileSystem()

    # If no s3_path is specified, use the base name of the local zarr_path
    if s3_path is None:
        s3_path = os.path.basename(zarr_path)

    # Check if the local Zarr path exists
    if not os.path.exists(zarr_path):
        raise FileNotFoundError(f"The local Zarr directory {zarr_path} does not exist.")

    # Upload the Zarr directory to S3
    try:
        # Using s3fs to copy the entire directory
        fs.put(zarr_path, f"s3://{s3_bucket}/{s3_path}", recursive=True)
        print(f"Zarr data uploaded successfully to s3://{s3_bucket}/{s3_path}")

        # If upload is successful, delete the local Zarr directory
        if os.path.isdir(zarr_path):
            shutil.rmtree(zarr_path)  # Use shutil.rmtree for directories with contents
            print(f"Local Zarr directory {zarr_path} has been deleted.")
        else:
            os.remove(zarr_path)  # If it's a file, remove it
            print(f"Local Zarr file {zarr_path} has been deleted.")
    except Exception as e:
        print(f"Error uploading Zarr to S3: {e}")
    return s3_path

def get_bronze (year_start, year_end, var, bronze_bucket_nm):
    # TO DO - Validate year and var input

    # download raw and save to local directory
    local_zarr = download_multiple_years(year_start, year_end, var)


    # upload to bronze bucket
    time_start = time.time()
    s3_path = upload_zarr_to_s3(local_zarr, bronze_bucket_nm, s3_path=None)
    du.elapsed(time_start)
    return s3_path
