# pylint: disable=C0103,C0116

import os
import time
import pandas as pd
import xarray as xr
import data_utils as du
import boto3
import xarray as xr
import re
from s3fs.core import S3FileSystem


# CONSTANTS
ROOT = "https://daacdata.apps.nsidc.org/pub/DATASETS/nsidc0719_SWE_Snow_Depth_v1/"
USERNAME = os.environ["EARTHDATA_USER"]
PASSWORD = os.environ["EARTHDATA_PASS"]
QUIET = False
BRONZE_BUCKET_NM = "swe-bronze"
SILVER_BUCKET_NM = "swe-silver"
GOLD_BUCKET_NM =  "swe-gold"
YEAR_LIST = [range(1982, 1990), range(1990, 2000), range(2000, 2010), \
              range(2010, 2020), range(2020, 2024)]

# download file from url
def get_raw(year, VARS_TO_KP = "SWE"):
    file_name = f"4km_SWE_Depth_WY{year}_v01.nc"
    ds = du.url_to_ds(ROOT, file_name, requires_auth=True, \
                 username=USERNAME, password=PASSWORD)
    ds = ds[VARS_TO_KP]
    return ds

# def get_raw_multi(years, VARS_TO_KP = "SWE"):
#     time_start = time.time()
#     files = [f"4km_SWE_Depth_WY{year}_v01.nc" for year in years]
#     ds = du.url_to_ds_muliti(ROOT, files, requires_auth=True, \
#                  username=USERNAME, password=PASSWORD)
#     ds = ds[VARS_TO_KP]
#     elapsed = time.time() - time_start
#     print(f"Elapsed time is {elapsed}")
#     return ds

# crude filter of data based on bounding box
def crude_filter(ds, min_lon, min_lat, max_lon, max_lat):
    filtered_ds = ds.sel(lat=slice(min_lat, max_lat), lon=slice(min_lon, max_lon))
    return filtered_ds

def raw_to_bronze(geos, year):
    ds = get_raw(year)
    min_lon, min_lat, max_lon, max_lat = geos.unary_union.bounds
    filtered_ds = crude_filter(ds, min_lon, min_lat, max_lon, max_lat)
    return filtered_ds

def raw_to_bronze_multi(geos, years):
    yr = years[0]
    if not QUIET:
        print(f"processing raw data year {yr}")
    results = raw_to_bronze(geos, yr)
    for yr in years[1:]:
        if not QUIET:
            print(f"processing raw data year {yr}")
        ds = raw_to_bronze(geos, yr)
        results = xr.concat([ds, results], dim="time")
    return results

def bronze_to_silver(row):
    results = pd.DataFrame()
    huc_id = row.iloc[0]["huc_id"][:8] # TO DO - DYNAMIC UPPER LEVEL 

    # gather files 
    s3 = S3FileSystem(anon=False)
    s3path = f"s3://{BRONZE_BUCKET_NM}/raw_swe_unmasked_in_{huc_id}*"
    files = s3.glob(s3path)
    if not files:
        raise ValueError(f"No bronze data ready for silver processing for huc {huc_id}")
    fileset = [s3.open(f) for f in files]
    
    # open dataset, filter by geo, save to df
    for f in fileset:
        ds = xr.open_dataset(f)
        ds.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace = True)
        ds = ds.rio.write_crs("EPSG:4326", inplace=True)
        ds_filter = du.filter_by_geo (ds, row)
        df = ds_filter.to_dataframe()
        df = df[df.columns.drop(["spatial_ref", "crs"])]
        df["huc_id"] = huc_id
        results = pd.concat([results, df])

    return results

def process_bronze_all (geos, huc_id, overwrite = False):
    for years in YEAR_LIST:
        f_bronze = f"raw_swe_unmasked_in_{huc_id}_{min(years)}_to_{max(years)}"
        if du.isin_s3(BRONZE_BUCKET_NM, f"{f_bronze}.nc") and not overwrite:
            print(f"File {f_bronze} already exists in {BRONZE_BUCKET_NM}")
            bronze_ds = du.s3_to_ds(BRONZE_BUCKET_NM, f"{f_bronze}.nc")
        else:
            bronze_ds = raw_to_bronze_multi(geos, years)
            du.dat_to_s3(bronze_ds, BRONZE_BUCKET_NM, f_bronze)
    

def process_all(huc_lev, huc_id, overwrite_s = False, overwrite_b = False): 
    # track processing time
    time_start = time.time()

    # get shape file
    geos = du.get_basin_geos(huc_lev, huc_id)

    # bronze processing
    process_bronze_all (geos, huc_id, overwrite = overwrite_b) 
    elapsed_time = time.time() - time_start
    print(f"Elapsed time : {elapsed_time:.2f} seconds")

    #silver & gold procesing 
    for i in range (geos.shape[0]):
        row = geos.iloc[[i]]
        huc_id = row.iloc[0]["huc_id"]

        # silver processing
        f_silver = f"raw_swe_in_{huc_id}"
        if du.isin_s3(SILVER_BUCKET_NM, f"{f_silver}.csv") and not overwrite_s: 
            print(f"File{f_silver} already exists in {SILVER_BUCKET_NM}")
            silver_df = du.s3_to_df(f"{f_silver}.csv", SILVER_BUCKET_NM)      
            
        else: 
            print(f"procesing huc_id {huc_id}")
            silver_df = bronze_to_silver(row) 
            du.dat_to_s3(silver_df, SILVER_BUCKET_NM, f_silver, file_type="csv")

        # gold processing     
        gold_df = silver_df.groupby(['time'])['SWE'].mean().reset_index()
        gold_df["huc_id"] = huc_id
        gold_df = gold_df.rename(columns={"SWE": "mean_swe"})
        f_out = f"mean_swe_in_{huc_id}"
        du.dat_to_s3(gold_df, GOLD_BUCKET_NM, f_out, file_type="csv")

    elapsed_time = time.time() - time_start
    print(f"Elapsed time : {elapsed_time:.2f} seconds")


 


