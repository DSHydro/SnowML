""" 
Module with functions to download data and save in S3, as well as geo
masking and taking a basin mean. Requires that you have set Amazon Credentials
as Environment Variables.
"""

# pylint: disable=C0103


import io
import os
import time
from io import StringIO
import warnings
import s3fs
import boto3
import xarray as xr
import pandas as pd
import geopandas as gpd
import requests
from rasterio.transform import from_bounds
from affine import Affine
import pyproj
import numpy as np



# TO DO - fix the future warning issue
def calc_transform(ds):
    """
    Calculates the affine transformation matrix for datasets with latitude
    values listed from smallest to largest.

    Args:
        ds (xarray.Dataset): Input dataset containing 'lon' and 'lat' dims.

    Returns:
        affine.Affine: The affine transformation matrix.
    
    Note:   Thus function assumes that the latitude values are listed from 
            smallest to largest. If the latitude values are listed from largest 
            to smallest, use the calc_Affine function instead.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
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


def filter_by_geo (ds, geo):
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


def get_url_pattern(var):
    """
    Generates a URL pattern based on the variable type.

    Parameters:
    var (str): The variable type. Accepted values are:
               ["swe", "pr", "tmmn", "sph", "vs", "srad", "tmmx", "rmin","rmax"]
    Returns:
    str: The URL pattern for the specified variable type. If the variable type 
            is not recognized, returns an empty string.
    """
    if var == "swe":
        root = "https://daacdata.apps.nsidc.org/pub/DATASETS/nsidc0719_SWE_Snow_Depth_v1/"
        file_name_pattern = "4km_SWE_Depth_WY{year}_v01.nc"
        url_pattern = root+file_name_pattern
    elif var == "swe_ucla": 
        url_pattern = "https://n5eil01u.ecs.nsidc.org/SNOWEX/WUS_UCLA_SR.001/{Yr}.10.01/WUS_UCLA_SR_v01_N{north}_0W{west}_0_agg_16_WY{Yr}_{Yr_end}_SWE_SCA_POST.nc"
   
    
    elif var in ["pr", "tmmn", "sph", "vs", "srad", "tmmx", "rmin", "rmax"]:
        url_p = f"http://www.northwestknowledge.net/metdata/data/{var}"
        url_pattern = url_p + "_{year}.nc"
    else:
        print("var not regognized")
        url_pattern = ""
    return url_pattern

def url_to_ds(root, file_name,requires_auth=False, username=None, password=None):
    """ Direct load from url to netcdf file """

    # Prepare authentication if required
    auth = (username, password) if requires_auth else None

    url = root + file_name
    response = requests.get(url, auth=auth, timeout=60)

    # Check if the request was successful
    if response.status_code == 200:
        # Convert the raw response content to a file-like object
        file_like_object = io.BytesIO(response.content)

        # Open the dataset from the file-like object
        ds = xr.open_dataset(file_like_object)

        return ds

    print(f"Failed to fetch data. Status code: {response.status_code}")
    return None


def elapsed(time_start):
    """
    Calculate and print the elapsed time since the given start time.

    Args:
        time_start (float): The start time in seconds since the epoch.

    Returns:
        None
    """
    elapsed_time = time.time() - time_start
    print(f"______Elapsed time is {int(elapsed_time)} seconds")


def s3_to_ds(bucket_name, file_name):
    """
    Load a dataset from an S3 bucket into an xarray Dataset.

    Parameters:
    bucket_name (str): The name of the S3 bucket.
    file_name (str): The name of the file in the S3 bucket.

    Returns:
    xarray.Dataset: The loaded dataset.
    """
    s3_path = f"s3://{bucket_name}/{file_name}"
    fs = s3fs.S3FileSystem(anon=False)
    with fs.open(s3_path) as f:
        ds = xr.open_dataset(f, engine="h5netcdf")
        ds.load()
    return ds



def s3_to_gdf(bucket_name, file_name, region_name="us-west-2"):
    """
    Download a file from S3 and load it into a GeoDataFrame.

    Args:
        bucket_name (str): Name of the S3 bucket.
        s3_key (str): The S3 key (path) to the file.
        region_name (str): AWS region of the S3 bucket. Default is 'us-west-2'.
    
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



def dat_to_s3(dat, bucket_name, f_out, file_type="netcdf", region_name="us-west-2"):
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
        raise ValueError(f"Invalid file_type '{file_type}'. Supported types: {valid_file_types}")

    # Adjust file name based on file type
    file_extension_map = {
        "csv": ".csv",
        "parquet": ".parquet",
        "netcdf": ".nc", 
        "zarr": ""
    }

    file_extension = file_extension_map[file_type]
    output_file = f"{f_out}{file_extension}"

    if file_type == "csv":
        dat.to_csv(output_file, index=True)
    elif file_type == "parquet":
        dat.to_dataframe().to_parquet(output_file)
    elif file_type == "netcdf":
        dat.to_netcdf(output_file)


    # Upload to S3
    s3_client = boto3.client('s3', region_name=region_name)
    s3_client.upload_file(output_file, bucket_name, output_file)

    # Cleanup local files
    os.remove(output_file)

    print(f"File {output_file} successfully uploaded to {bucket_name}")


def isin_s3(bucket_name, file_name):
    """Check if a file exists in an S3 bucket."""
    s3 = boto3.client('s3')
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
    s3 = boto3.client('s3')
    response = s3.get_object(Bucket=bucket_name, Key=file_name)
    content = response['Body'].read().decode('utf-8')
    df = pd.read_csv(StringIO(content))
    return df

# Returns a geo dataframe of geometries from the shape bornze bucket
def get_basin_geos (huc_lev, huc_no, bucket_nm = "shape-bronze"):
    """
    Retrieve a basin's geographical data from an S3 bucket.

    Parameters:
        huc_lev (str): The hydrologic unit code level.
        huc_no (str): The hydrologic unit code number.
        bucket_nm (str, optional): The name of the S3 bucket where the 
        shapefile is stored. Defaults to "shape-bronze".

        Returns:
            GeoDataFrame: A gpd with geographical data of the basin.

        Raises:
            ValueError: If the shapefile not found in the specified S3 bucket.
        """
    file_nm = f"{huc_lev}_in_{huc_no}.geojson"
    if not isin_s3(bucket_nm, file_nm):
        raise ValueError(f"No shape file found for {file_nm} in {bucket_nm}")
    basin_gdf = s3_to_gdf (bucket_nm, file_nm)
    print(f"Shapefile {file_nm} uploaded from {bucket_nm}")
    return basin_gdf


def s3_to_ds_zarr (bucket_name, zarr_path):
    """
    Create an xarray by opening a Zarr store on S3.

    Parameters:
    - bucket_name (str): The name of the S3 bucket.
    - zarr_path (str): The path to the Zarr file within the bucket.
    - anon (bool): Whether to access the bucket anonymously (default: True).

    Returns:
    - Dataset(xarray): The loaded dataset.

    Example:
    >>> ds = load_zarr_from_s3('my-bucket', 'my-data.zarr')
    """
    s3_url = f"s3://{bucket_name}/{zarr_path}"
    dat = xr.open_zarr(store=s3_url, chunks={}, consolidated=True)

    return dat



def reproject_to_latlon(ds_lidar):
    """
    Convert an xarray dataset from UTM (EPSG:32611) to lat/lon (EPSG:4326),
    drop x and y, and reindex using lat and lon.
    
    Parameters:
    -----------
    ds_lidar : xarray.Dataset
        Input dataset with 'x' and 'y' coordinates in UTM Zone 11N.

    Returns:
    --------
    xarray.Dataset
        Dataset with 'lat' and 'lon' coordinates instead of 'x' and 'y'.
    """
    # Set up transformer from UTM Zone 11N to WGS84 (lat/lon)
    transformer = pyproj.Transformer.from_crs("EPSG:32611", "EPSG:4326", always_xy=True)

    # Extract x and y coordinate values
    x = ds_lidar['x'].values
    y = ds_lidar['y'].values

    # Create meshgrid of x and y to match dataset dimensions
    xx, yy = np.meshgrid(x, y)

    # Transform the meshgrid to lon and lat
    lon, lat = transformer.transform(xx, yy)

    # Create new DataArray for lat and lon
    lat_da = xr.DataArray(lat, dims=("y", "x"))
    lon_da = xr.DataArray(lon, dims=("y", "x"))

    # Assign new coordinates
    ds_lidar = ds_lidar.assign_coords(lat=lat_da, lon=lon_da)

    # Drop old x and y coords
    ds_lidar = ds_lidar.drop_vars(['x', 'y'])

    return ds_lidar
