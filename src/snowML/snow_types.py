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
    url = "https://daacdata.apps.nsidc.org/pub/DATASETS/nsidc0768_global_seasonal_snow_classification_v01/SnowClass_NA_05km_2.50arcmin_2021_v01.0.nc"
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
        ds.to_zarr(s3_path, mode="w", consolidated=True)
        print(f"Created new Zarr file at s3://{bucket}/{s3_path}")

def snow_class_data_from_s3(geos = None):
    bucket  = "dawgs-bronze"# TO DO: make this a parameter
    zarr_store_url = f's3://{bucket}/snow_class_data.zarr'
    ds_conus = xr.open_zarr(store=zarr_store_url, consolidated=True)
    if geos is None: # return the full data for CONUS
        return ds_conus
    else: # return all data witin the geo 
        geos = geos.to_crs(ds.rio.crs)
        ds_clipped = ds_conus.rio.clip(geos.geometry, geos.crs, drop=True)
        return ds_clipped
    
def map_snow_class_names(): 
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
    time_start = time.time()
    results = pd.DataFrame()
    snow_class_names = map_snow_class_names()
    ds_conus = get_snow_class_data(geos = None)
    for i in range(geos.shape[0]):
        print(f"processing geos {i+1} of {geos.shape[0]}")
        row = geos.iloc[[i]]
        row = row.to_crs(ds_conus.rio.crs)
        ds = ds_conus.rio.clip(row.geometry, row.crs, drop=True)
        df_snow_classes = calc_snow_class(ds, snow_class_names)
        df_snow_classes["huc_id"] = row["huc_id"].values[0]
        results = pd.concat([results, df_snow_classes], ignore_index=True)
    du.elapsed(time_start)
    return results