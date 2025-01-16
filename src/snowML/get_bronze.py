# Module to download raw data to bronze bucket 

import requests
import os
import data_utils as du
import s3fs
import time
import warnings
import xarray as xr
from tqdm import tqdm


BRONZE_BUCKET_NM = "wrf-prebronze"

def elapsed(time_start):
    elapsed_time = time.time() - time_start
    print(f"elapsed time is {elapsed_time}")

def get_url_pattern(var):
    if var.__eq__("swe"):
        root = "https://daacdata.apps.nsidc.org/pub/DATASETS/nsidc0719_SWE_Snow_Depth_v1/"
        file_name_pattern = "4km_SWE_Depth_WY{year}_v01.nc"
        url_pattern = root+file_name_pattern
    elif var in ["pr", "tmmn", "vs"]: 
        url_pattern = "http://www.northwestknowledge.net/metdata/data/pr_{year}.nc"
    else: 
        print("var not regognized")
        url_pattern = ""
    return url_pattern 

def netcdf_to_zarr(years, var, zarr_output="combined_data.zarr", batch_size=1):
    """
    Downloads NetCDF files for the specified years, combines them into a Zarr file in batches, 
    and deletes the local NetCDF files.

    Parameters:
        years (list of int): List of years to download and process.
        url_pattern (str): URL pattern with a placeholder for the year (e.g., "https://example.com/data_{year}.nc").
        zarr_output (str): Path to save the combined Zarr file (default: "combined_data.zarr").
        batch_size (int): Number of years to process in each batch (default: 10).
    """
    time_start = time.time()
    output_dir = "downloaded_files"
    os.makedirs(output_dir, exist_ok=True)

    # Function to download a file
    def download_file(year):
        url_pattern = get_url_pattern(var)
        url = url_pattern.format(year=year)
        local_file = os.path.join(output_dir, f"WY{year}_v01.nc")
        if not os.path.exists(local_file):
            try:
                with requests.get(url, stream=True) as response:
                    response.raise_for_status()
                    with open(local_file, "wb") as f:
                        for chunk in tqdm(response.iter_content(chunk_size=8192), desc=f"Downloading WY{year}"):
                            f.write(chunk)
            except requests.RequestException as e:
                print(f"Failed to download {url}: {e}")
                return None
        else:
            print(f"File for year {year} already exists.")
        return local_file

    # Process files in batches
    for i in range(0, len(years), batch_size):
        batch_years = years[i:i + batch_size]
        print(f"Processing batch: {batch_years}")

        datasets = []
        for year in batch_years:
            file_path = download_file(year)
            if file_path and os.path.exists(file_path):
                ds = xr.open_dataset(file_path)
                ds_sorted = ds.sortby("lat")
                datasets.append(ds_sorted)

        if datasets:
            print(f"Writing batch {batch_years} to Zarr...")
            combined_batch = xr.concat(datasets, dim="day") # TODO: make dim dynamic
            # Append the batch to the Zarr file
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                if not os.path.exists(zarr_output):
                    combined_batch.to_zarr(zarr_output, mode="w")  # Use mode="w" to write the first datase
                else: 
                    combined_batch.to_zarr(zarr_output, mode="a", append_dim="day")
            print(f"Batch {batch_years} successfully written to {zarr_output}.")

        # Clean up memory
        datasets.clear()

    # Delete local NetCDF files
    print("Deleting downloaded NetCDF files...")
    for file in os.listdir(output_dir):
        os.remove(os.path.join(output_dir, file))
    os.rmdir(output_dir)
    print("All NetCDF files deleted.")

    elapsed_time = time.time() - time_start
    print(f"Elapsed time : {elapsed_time:.2f} seconds")
    return zarr_output

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
    except Exception as e:
        print(f"Error uploading Zarr to S3: {e}")  
    return s3_path 

def get_bronze (years, var, bronze_bucket_nm, batch_size=1): 
    # TO DO - Validate year input, batch size input

    # download raw and save to local directory 
    local_zarr = netcdf_to_zarr(years, var, zarr_output=f"combined_{var}.zarr", batch_size=batch_size)

    # upload to bronze bucket
    s3_path = upload_zarr_to_s3(local_zarr, bronze_bucket_nm, s3_path=None)

    return s3_path


