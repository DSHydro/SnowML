""" 
Module with functions to download data and save in S3, as well as geo
masking and taking a basin mean.Requires that you have set Amazon Credentials
as Environment Variables.
"""

import io, os, rioxarray
import requests
import s3fs
import xarray as xr
import boto3
import geopandas as gpd
from botocore.exceptions import NoCredentialsError, ClientError




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

def ds_mean(ds):
    ds_mean = ds.mean(dim=['lat','lon'])
    ds_mean = ds_mean.rename({var: f"mean_{var}" for var in ds_mean.data_vars})
    return ds_mean


def url_to_s3(root, file_name, bucket_name, region_name="us-east-1",
               requires_auth=False, username=None, password=None,
               quiet = True):
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
            print(f"File '{file_name}' already exists in \
                  bucket '{bucket_name}', skipping download.")
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
    with fs.open(s3_path) as f:
        ds = xr.open_dataset(f)
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
        return 
    
def ds_to_s3(ds, bucket_name, f_out, region_name="us-east-1"):
    """
    Save an Xarray Dataset to an S3 bucket.

    Args:
        ds (xr.Dataset): The Xarray Dataset to save.
        bucket_name (str): The S3 bucket name.
        file_name (str): The name of the file to save in the S3 bucket.
        region_name (str): AWS region of the S3 bucket (optional).
    
    Returns:
        None
    """
    ds.to_netcdf(f_out)
    s3_client = boto3.client('s3')
    s3_client.upload_file(f_out, bucket_name, f_out)
    os.remove(f_out)
    
    print(f"File {f_out} successfully uploaded to {bucket_name}")

def isin_s3(bucket_name, file_name):
    """Check if a file exists in an S3 bucket."""
    s3 = boto3.client('s3')
    try:
        s3.head_object(Bucket=bucket_name, Key=file_name)
        return True
    except s3.exceptions.ClientError:
        return False