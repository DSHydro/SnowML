import xarray as xr
import pandas as pd
import requests, io
import geopandas as gpd
import boto3
import numpy as np
import s3fs
import time
import data_utils as du 


def get_snow_class_data(geos = None):
    url = "https://daacdata.apps.nsidc.org/pub/DATASETS/nsidc0768_global_seasonal_snow_classification_v01/SnowCla>
    response = requests.get(url)
    ds = xr.open_dataset(io.BytesIO(response.content))
    if geos is None: # return the data for CONUS
        lat_min, lat_max = 24.396308, 49.384358
        lon_min, lon_max = -125.0, -66.93457
        ds_conus = ds.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
        # properly set the crs (from metadata, should be "EPSG:4326")
        ds_conus = ds_conus.rio.write_crs("EPSG:4326")    
        return ds_conus
    else: # return all data witin the geo 
        # properly set the crs (from metadata, should be "EPSG:4326")
        ds = ds.rio.write_crs("EPSG:4326")  
        geos = geos.to_crs(ds.rio.crs)
        ds_final = ds.rio.clip(geos.geometry, geos.crs, drop=True)
        return ds_final
    
def save_snow_class_data(ds):
    bucket = "dawgs-bronze" # TO DO: make this a parameter
    file_name = "snow_class_data.zarr"
    s3_path = f"s3://{bucket}/{file_name}"
    fs = s3fs.S3FileSystem()
    if fs.exists(s3_path):
        print(f"Zarr file already exists at s3://{bucket}/{s3_path}")
    else:
        ds.to_zarr(s3_path, mode="w", consolidated=True