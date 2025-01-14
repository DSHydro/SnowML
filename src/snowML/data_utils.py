""" 
Module with functions to download data and save in S3, as well as geo
masking and taking a basin mean. Requires that you have set Amazon Credentials
as Environment Variables.
"""

# pylint: disable=C0103


import io
import boto3
import os
import requests
import s3fs
import re
import zarr
import xarray as xr
import pandas as pd
import geopandas as gpd
from botocore.exceptions import NoCredentialsError, ClientError
from io import StringIO
from s3fs.core import S3FileSystem
import rasterio
from rasterio.transform import from_bounds
from affine import Affine


# use calc_transform instead of Affine if the data is normally sorted
def calc_transform(ds):
    transform = from_bounds(
        west=ds.lon.min().item(),
        south=ds.lat.min().item(),
        east=ds.lon.max().item(),
        north=ds.lat.max().item(),
        width=ds.dims["lon"],
        height=ds.dims["lat"],
    )
    return transform


def calc_Affine(ds):
    """
    Calculates the affine transformation matrix for datasets with latitude
    values listed from largest to smallest.
    """
    lon_min = ds.lon.min().values
    lat_max = ds.lat.max().values
    lon_res = (ds.lon[1] - ds.lon[0]).values  # Longitude resolution
    lat_res = (ds.lat[1] - ds.lat[0]).values  # Latitude resolution (negative)

    # Construct and return the affine matrix
    return Affine(lon_res, 0, lon_min, 0, lat_res, lat_max)


def filter_by_geo(ds, geo):
    """
    Filter an Xarray file by a geographical mask.

    Args:
        ds(Xarray): The input data to mask.
        geo(geopandas df): The geometry to filter by.

    Returns:
       ds_final: A clipped Xarray
    """

    geo = geo.to_crs(ds.rio.crs)
    ds_final = ds.rio.clip(geo.geometry, geo.crs, drop=True)
    return ds_final


def ds_mean(ds):
    ds_mean = ds.mean(dim=["lat", "lon"])
    ds_mean = ds_mean.rename({var: f"mean_{var}" for var in ds_mean.data_vars})
    return ds_mean


def url_to_ds(root, file_name, requires_auth=False, username=None, password=None):
    """Direct load from url to netcdf file"""

    # Prepare authentication if required
    auth = (username, password) if requires_auth else None

    url = root + file_name
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


def url_to_ds_muliti(
    root, file_names, requires_auth=False, username=None, password=None
):
    """Load multiple NetCDF files from URLs and combine them into a single dataset.

    Parameters:
        root (str): The root URL where files are located.
        file_names (list of str): List of file names to load.
        requires_auth (bool): Whether authentication is required.
        username (str): Username for authentication (if required).
        password (str): Password for authentication (if required).

    Returns:
        xarray.Dataset: Combined dataset from all files.
    """
    # Prepare authentication if required
    auth = (username, password) if requires_auth else None

    file_like_objects = []

    for file_name in file_names:
        url = root + file_name
        response = requests.get(url, auth=auth)

        # Check if the request was successful
        if response.status_code == 200:
            # Convert the raw response content to a file-like object
            file_like_objects.append(io.BytesIO(response.content))
        else:
            print(f"Failed to fetch {file_name}. Status code: {response.status_code}")

    if not file_like_objects:
        print("No files could be loaded. Returning None.")
        return None

    # Open multiple file-like objects as a single dataset
    ds = xr.open_mfdataset(file_like_objects, combine="by_coords")

    return ds


def url_to_s3(
    root,
    file_name,
    bucket_name,
    region_name="us-east-1",
    requires_auth=False,
    username=None,
    password=None,
    quiet=True,
):
    """
    Download a data file from a given URL and save it to an S3 bucket.

    Args:
        root (str): The first part of the URL from which to download.
        file_name (str): The name of the file to save to S3.
        bucket_name (str): The name of the S3 bucket to save the file.
        region_name (str): The AWS region of S3 bucket. Default is us-east-1.
        requires_auth (bool): Indicates if the URL requires authentication.
                 Default is False.
        username (str): Username for authentication (required if
                requires_auth is True).
        password (str): Password for authentication (required if
                requires_auth is True).

    Returns:
        str: The name of the uploaded file, or None if the operation fails.
    """

    url = root + file_name

    # Check if the file already exists in the bucket
    if isin_s3(bucket_name, file_name):
        if not quiet:
            print(
                f"File '{file_name}' already exists in \
                  bucket '{bucket_name}', skipping download."
            )
        return None

    # Prepare authentication if required
    auth = (username, password) if requires_auth else None

    # Download the file and upload to S3
    s3_client = boto3.client("s3", region_name=region_name)
    try:
        with requests.get(url, stream=True, auth=auth) as response:
            response.raise_for_status()

            # Stream directly to S3 using boto3
            with response.raw as data_stream:
                s3_client.upload_fileobj(data_stream, bucket_name, file_name)

        if not quiet:
            print(f"File {file_name} uploaded to S3 bucket '{bucket_name}'.")
        return file_name

    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {e}")
    except NoCredentialsError:
        print("AWS credentials not found.")
    except ClientError as e:
        print(f"Error uploading file to S3: {e}")

    return None


def s3_to_ds(bucket_name, file_name):
    s3_path = f"s3://{bucket_name}/{file_name}"
    fs = s3fs.S3FileSystem(anon=False)
    # fs = s3fs.S3FileSystem(cache_regions=False)
    # fs.invalidate_cache
    with fs.open(s3_path) as f:
        ds = xr.open_dataset(f, engine="h5netcdf")
        ds.load()
    return ds


def s3_to_gdf(bucket_name, file_name, region_name="us-east-1"):
    """
    Download a file from S3 and load it into a GeoDataFrame.

    Args:
        bucket_name (str): Name of the S3 bucket.
        s3_key (str): The S3 key (path) to the file.
        region_name (str): AWS region of the S3 bucket. Default is 'us-east-1'.

    Returns:
        geos: A GeoDataFrame containing the data from the file.
    """

    s3_client = boto3.client("s3", region_name=region_name)

    # Download the file from S3 into a bytes buffer
    try:
        buffer = io.BytesIO()
        s3_client.download_fileobj(bucket_name, file_name, buffer)
        buffer.seek(0)  # Reset buffer position to the beginning

        # Read the file into a GeoDataFrame (assuming GeoJSON format)
        geos = gpd.read_file(buffer)

        return geos

    except Exception as e:
        print(f"Error downloading or reading file from S3: {e}")
        return None


def dat_to_s3(dat, bucket_name, f_out, file_type="netcdf", region_name="us-east-1"):
    """
    Save a Dataset to an S3 bucket in the specified format.

    Args:
        dat: The Dataset to save. Can be an xarray.Dataset (for netcdf or zarr)
        or a DataFrame (for csv or parquet).
        bucket_name (str): The S3 bucket name to save to.
        bucket_name (str): The S3 bucket name to save to.
        f_out (str): The base name of the file to save in the S3 bucket.
        file_type (str): The format to save the file ('csv', 'parquet', or 'netcdf').
        region_name (str): AWS region of the S3 bucket (optional).

    Returns:
        None
    """
    valid_file_types = ["csv", "parquet", "netcdf"]
    if file_type not in valid_file_types:
        raise ValueError(
            f"Invalid file_type '{file_type}'. Supported types: {valid_file_types}"
        )

    # Adjust file name based on file type
    file_extension_map = {
        "csv": ".csv",
        "parquet": ".parquet",
        "netcdf": ".nc",
        "zarr": "",
    }

    file_extension = file_extension_map[file_type]
    output_file = f"{f_out}{file_extension}"

    # Save the dataset in the specified format

    # if file_type == "zarr":
    #     # Save Zarr directly to S3
    #     dat.to_zarr(f"s3://{bucket_name}/{output_file}/", mode="w")
    #     print(f"Zarr dataset successfully uploaded to s3://{bucket_name}/{output_file}")
    #     return

    if file_type == "csv":
        dat.to_csv(output_file)
    elif file_type == "parquet":
        dat.to_dataframe().to_parquet(output_file)
    elif file_type == "netcdf":
        dat.to_netcdf(output_file)

    # Upload to S3
    s3_client = boto3.client("s3", region_name=region_name)
    s3_client.upload_file(output_file, bucket_name, output_file)

    # Cleanup local files
    os.remove(output_file)

    print(f"File {output_file} successfully uploaded to {bucket_name}")


def isin_s3(bucket_name, file_name):
    """Check if a file exists in an S3 bucket."""
    s3 = boto3.client("s3")
    try:
        s3.head_object(Bucket=bucket_name, Key=file_name)
        return True
    except s3.exceptions.ClientError:
        return False


def s3_to_df(file_name, bucket_name):
    """
    Loads a CSV file from an S3 bucket into a pandas DataFrame.

    Parameters:
        file_name (str): The name of the CSV file in the S3 bucket.
        bucket_name (str): The name of the S3 bucket.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the data from the CSV file.
    """
    s3 = boto3.client("s3")
    response = s3.get_object(Bucket=bucket_name, Key=file_name)
    content = response["Body"].read().decode("utf-8")
    df = pd.read_csv(StringIO(content))
    return df


# Returns a geo dataframe of geometries from the shape bornze bucket
def get_basin_geos(huc_lev, huc_no, bucket_nm="shape-bronze"):
    file_nm = f"{huc_lev}_in_{huc_no}.geojson"
    if not isin_s3(bucket_nm, file_nm):
        raise ValueError(f"No shape file found for {file_nm} in {bucket_nm}")
    basin_gdf = s3_to_gdf(bucket_nm, file_nm)
    print(f"Shapefile {file_nm} uploaded from {bucket_nm}")
    return basin_gdf
