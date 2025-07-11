
# pylint: disable=C0103


import warnings
import time
import io
import os
import sys
import shutil
import json
import s3fs
import requests
import xarray as xr
import rioxarray as rxr
import pandas as pd
import earthaccess
import tempfile
from snowML.datapipe.utils import data_utils as du
from snowML.datapipe.utils import get_geos as gg 
from snowML.datapipe.utils import set_data_constants as sdc

from snowML.datapipe import get_bronze as gb

# define constants
VAR_DICT = sdc.create_var_dict()
MY_EARTHACCESS_USER = "suetboyd"
MY_EARTHACCESS_LOGIN = "LTsuey78****"

# LOG IN TO EARTHEACCESS 
# Set once per session (or omit entirely if using .netrc)
os.environ.setdefault("EARTHDATA_USERNAME", "MY_EARTHACCESS_USER")
os.environ.setdefault("EARTHDATA_PASSWORD", "MY_EARTHACCESS_LOGIN")
# Login once when module is imported
earthaccess.login(strategy="password")



def format_nsidc_url(north, west, yr):
    """
    Formats a URL for accessing WUS_UCLA_SR data from NSIDC.

    Parameters:
    - north (int or str): Latitude value (e.g., 37)
    - west (int or str): Longitude value (e.g., 120)
    - yr (int or str): Start year (e.g., 1984)

    Returns:
    - str: Formatted URL
    """
    
    yr_end = str(yr + 1)[-2:]
    url_template = du.get_url_pattern("swe_ucla")
    return url_template.format(north=north, west=west, Yr=yr, Yr_end=yr_end)


def url_to_ds_earthaccess(url, timeout=60):
    try:
        temp_dir = tempfile.mkdtemp()
        file_path = earthaccess.download(url, local_path=temp_dir)[0]
        ds = xr.open_dataset(file_path, engine="netcdf4", chunks={"day": -1, "lat": None, "lon": None})
        
        #  Load the data into memory, then remove the file and its containing temporary directory
        ds.load()  # Fully load the dataset into memory
        temp_dir = os.path.dirname(file_path)
        shutil.rmtree(temp_dir)
        
        return ds

    except Exception as e:
        print(f"Failed to download or open dataset: {e}")
        return None



def get_one_file(north, west, yr):
    url = format_nsidc_url(north, west, yr)
    print(yr, url)
    ds = url_to_ds_earthaccess(url)
    return ds


def get_one_year(yr, begin_north, end_north, begin_west, end_west):
    datasets = []

    for north in range(begin_north, end_north + 1):
        for west in range(begin_west, end_west + 1):
            ds = get_one_file(north, west, yr)
            # Select the first stat and remove the Stats dimension
            ds = ds.isel(Stats=0)
            # remove the SCA Variable
            ds = ds[['SWE_Post']]

            datasets.append(ds)

    # Merge spatially (no lat/lon overlap)
    results_ds = xr.merge(datasets)


    # Sort by latitude and longitude
    if not results_ds['Latitude'].values[0] < results_ds['Latitude'].values[-1]:
        results_ds = results_ds.sortby('Latitude')
    if not results_ds['Longitude'].values[0] < results_ds['Longitude'].values[-1]:
        results_ds = results_ds.sortby('Longitude')

    return results_ds

def get_bounds(geos):
    combined = geos.unary_union
    bounds = combined.bounds  # returns a tuple: (min_lon, min_lat, max_lon, max_lat)
    bounds_truncated_sorted = sorted(abs(int(i)) for i in bounds)
    return bounds_truncated_sorted

def clip_and_mean(ds_year, geos):
    ds_year.rio.write_crs(geos.crs, inplace=True)
    try:
        ds_year.rio.set_spatial_dims(x_dim="Longitude", y_dim="Latitude", inplace=True)
        clipped_data = ds_year.rio.clip(geos.geometry, drop=True)
        mean_per_day = clipped_data.mean(dim=["Latitude", "Longitude"])
    except:
        ds_year.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
        clipped_data = ds_year.rio.clip(geos.geometry, drop=True)
        mean_per_day = clipped_data.mean(dim=["lat", "lon"])
    
    return mean_per_day



def assign_water_year_dates(df, start_year):
    """
    Assigns datetime values to a DataFrame indexed by 'Day', where Day 0 corresponds to
    October 1st of start_year.The function accounts for leap years and formats dates as 
    'YYYY-MM-DD'.

    Parameters:
    df (pd.DataFrame): DataFrame with an integer 'Day' index (e.g., 0 to 364 or 365).
    start_year (int): The starting year for the water year (e.g., 1984).

    Returns:
    pd.DataFrame: DataFrame with a new 'Date' column containing formatted datetime strings.
    """
    # Determine the number of days in the DataFrame
    num_days = df.shape[0]

    # Create a date range starting from October 1st of the specified year
    date_range = pd.date_range(start=f"{start_year}-10-01", periods=num_days, freq='D')

    # Assign the date range to a new 'Date' column
    df['Date'] = date_range.strftime('%Y-%m-%d')

    df = df.set_index('Date')
    df.index.name = 'day'

    return df


def get_mean(year, coords, geos):
    ds_year = get_one_year(year, coords[0], coords[1], coords[2]+1, coords[3]+1)
    mean_per_day = clip_and_mean(ds_year, geos)
    mean_df =  mean_per_day.to_dataframe()
    mean_df = mean_df[["SWE_Post"]]
    mean_df = assign_water_year_dates(mean_df, year)
    return mean_df


def get_gold_df(huc, year_start, year_end, overwrite = False):
    error_years = []
    time_start = time.time()
    geos = gg.get_geos(huc, str(len(str(huc))).zfill(2))
    coords = get_bounds(geos)
    results_df = pd.DataFrame()

    # check if file exists
    f_gold = f"mean_swe_ucla_2_in_{huc}"
    b_gold = "snowml-gold"  # TO DO - Make dynamic
    if du.isin_s3(b_gold, f"{f_gold}.csv") and not overwrite:
        print(f"File{f_gold} already exists in {b_gold}, skipping")
        return results_df, []
    
    else: 
        for yr in range(year_start, year_end):
            try:
                mean_df = get_mean(yr, coords, geos)
                results_df = pd.concat([results_df, mean_df], axis=0)
            except:
                print(f"Error processing year_{yr}, skipping")
                error_years.append(f"{huc}_{yr}")
        du.dat_to_s3(results_df, b_gold, f_gold, file_type="csv")
        du.elapsed(time_start)
        
    return results_df, error_years

def get_gold_multi(huc_list, year_start=1984, year_end=2021, overwrite = False):
    error_years = []
    count = 0
    for huc in huc_list:
        count += 1
        print(f"processing huc {count}")
        results_df, new_error_years = get_gold_df(huc, year_start, year_end, overwrite = overwrite)
        error_years = error_years + new_error_years   
    return error_years

def save_gold_df(huc, gold_df): 
    f_gold = f"mean_swe_ucla_2_in_{huc}"
    b_gold = "snowml-gold"  # TO DO - Make dynamic
    du.dat_to_s3(gold_df, b_gold, f_gold, file_type="csv")

def get_gold_df_from_gdf(geos, geos_name, year_start, year_end, overwrite = False):
    error_years = []
    time_start = time.time()
    coords = get_bounds(geos)
    results_df = pd.DataFrame()

    # check if file exists
    f_gold = f"mean_swe_ucla_2_in_{geos_name}"
    b_gold = "snowml-gold"  # TO DO - Make dynamic
    if du.isin_s3(b_gold, f"{f_gold}.csv") and not overwrite:
        print(f"File{f_gold} already exists in {b_gold}, skipping")
        return results_df, []
    
    else: 
        for yr in range(year_start, year_end):
            #try:
            mean_df = get_mean(yr, coords, geos)
            results_df = pd.concat([results_df, mean_df], axis=0)
            #except:
                #print(f"Error processing year_{yr}, skipping")
                #error_years.append(f"{geos_name}_{yr}")
        du.dat_to_s3(results_df, b_gold, f_gold, file_type="csv")
        du.elapsed(time_start)
        save_gold_df(geos_name, results_df)

        
    return results_df, error_years

