""" Module to Retrieve Lidar Data """
# pylint: disable=C0103

import re
import rioxarray as rxr
import xarray as xr
import earthaccess
import s3fs
from snowML.datapipe import set_data_constants as sdc


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
    swe_xr = rxr.open_rasterio(file, masked=True).squeeze()
    swe_ds = swe_xr.to_dataset(name="SWE")
    return swe_ds

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


def get_lidar_all(bucket_name = None):
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
    date_dict = create_data_dict(files)
    ds = concatenate_timeslices(date_dict)
    s3_path = "lidar_all.zarr"
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