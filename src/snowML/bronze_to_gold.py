""" Module to download and process University of Idaho Gridmet Data"""
 

import requests
import os
import data_utils as du
import s3fs
import time
import warnings
import xarray as xr
from tqdm import tqdm



VARS = ["pr", "tmmn", "vs"] 
VAR_NAMES = ["precipitation_amount", "air_temperature", "wind_speed"] # three example variables, there are others 
VAR_DICT = dict(zip(VARS, VAR_NAMES))


def elapsed(time_start):
    elapsed_time = time.time() - time_start
    print(f"elapsed time is {elapsed_time}")

def prep_bronze(geos, var, bucket_dict):
    # load_raw
    if var == "swe": 
        b_bronze = bucket_dict.get(f"{var}-bronze")
    else: 
        b_bronze = bucket_dict.get("wrf-bronze")
    zarr_store_url = f's3://{b_bronze}/{var}_all.zarr'  
    ds = xr.open_zarr(store=zarr_store_url, chunks={}, consolidated=True)
    ds_sorted = ds.sortby("lat")
    
    # Perform first cut crude filter 
    min_lon, min_lat, max_lon, max_lat = geos.unary_union.bounds
    small_ds = du.crude_filter(ds_sorted, min_lon, min_lat, max_lon, max_lat)
    if var != "swe":
        transform = du.calc_transform(small_ds)
        small_ds = small_ds.rio.write_transform(transform, inplace=True)
    else: 
        small_ds.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace = True)
    small_ds.rio.write_crs("EPSG:4326", inplace=True)
    return small_ds 
        

def process_silver_row (small_ds, row):
    ds_filter = du.filter_by_geo (small_ds, row)          
    silver_df = ds_filter.to_dataframe()
    silver_df = silver_df[silver_df.columns.drop(["spatial_ref", "crs"])]
    huc_id = row.iloc[0, 1] # get the id of the smaller huc uit 
    silver_df["huc_id"] = huc_id 
    return silver_df    

def process_gold (silver_df, var, huc_id, b_gold):
    
    if var == "swe": 
        grouper = "time"
        var_name = "SWE"
    else: 
        grouper = "day"
        var_name = VAR_DICT.get(var)
    gold_df = silver_df.groupby([grouper])[var_name].mean().reset_index() # TO DO: Fix Logic
    gold_df["huc_id"] = huc_id
    f_out = f"mean_{var}_in_{huc_id}"
    du.dat_to_s3(gold_df, b_gold, f_out, file_type="csv")


def process_all(huc_lev, huc_id, var, bucket_dict, overwrite = False):
    time_start = time.time()
    # get geos
    geos = du.get_basin_geos(huc_lev, huc_id)  # TO DO DYNAMIC BUCKET NAME 

    # get and prep bronze data
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)  # TO DO: ADDRESS THE FUTURE WARNING
        small_ds = prep_bronze(geos, var, bucket_dict)
    
    # silver and gold processing
    for i in range (geos.shape[0]):
        row = geos.iloc[[i]]
        huc_id = row.iloc[0, 1] # get the id of the smaller huc uit 
        
        # silver processing
        f_silver = f"raw_{var}_in_{huc_id}" 
        if var == "swe": 
            b_silver = bucket_dict.get(f"{var}-silver")
        else: 
            b_silver = bucket_dict.get("wrf-silver")
        if du.isin_s3(b_silver, f"{f_silver}.csv") and not overwrite: 
            print(f"File{f_silver} already exists in {b_silver}")
            silver_df = du.s3_to_df(f"{f_silver}.csv", b_silver)
        else: 
            print(f"processing silver for huc: {huc_id} ")
            silver_df = process_silver_row(small_ds, row)
            du.dat_to_s3(silver_df, b_silver, f_silver, file_type="csv")
    
        # gold_processing
        f_gold = f"mean_{var}_in_{huc_id}"
        if var == "swe": 
            b_gold = bucket_dict.get(f"{var}-gold")
        else: 
            b_gold = bucket_dict.get("wrf-gold")
        if du.isin_s3(b_gold, f"{f_gold}.csv") and not overwrite: 
            print(f"File{f_gold} already exists in {f_gold}")
        else: 
            print(f"processing gold for huc: {huc_id} ")
            process_gold(silver_df, var, huc_id, b_gold)   

        elapsed(time_start)
        
     
    