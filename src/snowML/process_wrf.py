""" Module to download and process University of Idaho Gridmet Data"""
 

import requests
import os
import data_utils as du
import s3fs
import time
import warnings
import xarray as xr
from tqdm import tqdm


BRONZE_BUCKET_NM = "wrf-prebronze" # bronze vs. pre_bronze for testing 
SILVER_BUCKET_NM = "wrf-silver"
GOLD_BUCKET_NM =  "wrf-gold"
VARS = ["pr", "tmmn", "vs"] 
VAR_NAMES = ["precipitation_amount", "air_temperature", "wind_speed"] # three example variables, there are others 
VAR_DICT = dict(zip(VARS, VAR_NAMES))


def prep_bronze(geos, var):
    # load_raw
    zarr_store_url = f's3://{BRONZE_BUCKET_NM}/{var}_all.zarr'  # TO DO FIX HARDCODE 
    ds = xr.open_zarr(store=zarr_store_url, chunks={}, consolidated=True)
    ds_sorted = ds.sortby("lat")
    # Perform first cut crude filter 
    min_lon, min_lat, max_lon, max_lat = geos.unary_union.bounds
    small_ds = du.crude_filter(ds_sorted, min_lon, min_lat, max_lon, max_lat)
    transform = du.calc_transform(small_ds)
    small_ds = small_ds.rio.write_transform(transform, inplace=True)
    small_ds.rio.write_crs("EPSG:4326", inplace=True)
    return small_ds 
        

def process_silver_row (small_ds, row):
    ds_filter = du.filter_by_geo (small_ds, row)          
    silver_df = ds_filter.to_dataframe()
    silver_df = silver_df[silver_df.columns.drop(["spatial_ref", "crs"])]
    huc_id = row.iloc[0, 1] # get the id of the smaller huc uit 
    silver_df["huc_id"] = huc_id 
    return silver_df    

def process_gold (silver_df, var, huc_id):
    var_name = VAR_DICT.get(var)
    gold_df = silver_df.groupby(['day'])[var_name].mean().reset_index() # TO DO: Fix Logic
    gold_df["huc_id"] = huc_id
    f_out = f"mean_{var}_in_{huc_id}"
    du.dat_to_s3(gold_df, GOLD_BUCKET_NM, f_out, file_type="csv")


def process_all(huc_lev, huc_id, var, overwrite = False):
    time_start = time.time()
    # get geos
    geos = du.get_basin_geos(huc_lev, huc_id)  

    # get and prep bronze data
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)  # TO DO: ADDRESS THE FUTURE WARNING
        small_ds = prep_bronze(geos, var)
    
    # silver and gold processing
    for i in range (geos.shape[0]):
        row = geos.iloc[[i]]
        huc_id = row.iloc[0, 1] # get the id of the smaller huc uit 
        
        # silver processing
        f_silver = f"raw_{var}_in_{huc_id}" 
        if du.isin_s3(SILVER_BUCKET_NM, f"{f_silver}.csv") and not overwrite: 
            print(f"File{f_silver} already exists in {SILVER_BUCKET_NM}")
            silver_df = du.s3_to_df(f"{f_silver}.csv", SILVER_BUCKET_NM)
        else: 
            print(f"processing silver for huc: {huc_id} ")
            silver_df = process_silver_row(small_ds, row)
            du.dat_to_s3(silver_df, SILVER_BUCKET_NM, f_silver, file_type="csv")
    
        # gold_processing
        f_gold = f"mean_{var}_in_{huc_id}"
        if du.isin_s3(GOLD_BUCKET_NM, f"{f_gold}.csv") and not overwrite: 
            print(f"File{f_gold} already exists in {GOLD_BUCKET_NM}")
        else: 
            print(f"processing gold for huc: {huc_id} ")
            process_gold(silver_df, var, huc_id)   
        
        elapsed(time_start)

        
     
    