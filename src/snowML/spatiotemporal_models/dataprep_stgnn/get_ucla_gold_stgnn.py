"""
UCLA / NSIDC WUS_UCLA_SR SWE gold layer for ST-GNN (and related pipelines).

Downloads Western U.S. Snow Reanalysis (UCLA) NetCDF tiles from NSIDC via ``earthaccess``,
mosaics tiles for a bounding box derived from HUC or custom geometry, clips to the basin,
and writes daily time series to S3 (``snowml-gold``) as ``ucla_swe_in_{huc}.csv``.

Water years: day 0 = October 1 of *year*; see ``assign_water_year_dates``.

Requires NASA Earthdata credentials (env vars, ``~/.netrc``, or ``MY_EARTHACCESS_*`` below).
"""
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

# Earthdata / NSIDC login (https://urs.earthdata.nasa.gov). Set your credentials below, or use
# env vars EARTHDATA_USERNAME / EARTHDATA_PASSWORD, or ~/.netrc (machine urs.earthdata.nasa.gov).
# If login fails: accept EULAs and authorize "NASA GESDISC" app at https://urs.earthdata.nasa.gov
MY_EARTHACCESS_USER = ""  # your Earthdata username
MY_EARTHACCESS_LOGIN = ""  # your Earthdata password

# NSIDC WUS_UCLA_SR Stats dimension order (User Guide): mean, std, median, 25th %, 75th %
STATS_NAMES = ["mean", "std", "median", "p25", "p75"]

# Set env vars from constants so earthaccess can use strategy="environment"
if MY_EARTHACCESS_USER and MY_EARTHACCESS_LOGIN:
    os.environ["EARTHDATA_USERNAME"] = MY_EARTHACCESS_USER
    os.environ["EARTHDATA_PASSWORD"] = MY_EARTHACCESS_LOGIN

# earthaccess supports: "environment" (env vars), "netrc" (~/.netrc), "interactive", "all" (try all)
# Use "environment" when credentials are in env/MY_*; otherwise "all" tries netrc then interactive
_login_strategy = "environment" if (os.environ.get("EARTHDATA_USERNAME") and os.environ.get("EARTHDATA_PASSWORD")) else "all"
earthaccess.login(strategy=_login_strategy)



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
    """Download one NetCDF URL with earthaccess, open with xarray, load into memory, then delete temp files."""
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
    """Fetch a single NSIDC tile identified by integer north latitude, west longitude, and water-year start *yr*."""
    url = format_nsidc_url(north, west, yr)
    print(yr, url)
    ds = url_to_ds_earthaccess(url)
    return ds


def get_one_year(yr, begin_north, end_north, begin_west, end_west, keep_all_stats=True):
    """
    Mosaic all WUS_UCLA_SR tiles in the latitude/longitude integer grid for one water year.

    Parameters
    ----------
    yr : int
        Water year start (e.g. 2012 → Oct 2012–Sep 2013).
    begin_north, end_north, begin_west, end_west : int
        Inclusive tile indices (NSIDC filenames use integer °N and °W).
    keep_all_stats : bool
        If True, keep full ``Stats`` dimension (mean, std, median, p25, p75).
        If False, keep only index 0 (ensemble mean).
    """
    datasets = []

    for north in range(begin_north, end_north + 1):
        for west in range(begin_west, end_west + 1):
            ds = get_one_file(north, west, yr)
            if not keep_all_stats:
                ds = ds.isel(Stats=0)
            ds = ds[["SWE_Post"]]
            datasets.append(ds)

    results_ds = xr.merge(datasets)
    # rioxarray expects monotonic lat/lon for predictable clipping
    if not results_ds["Latitude"].values[0] < results_ds["Latitude"].values[-1]:
        results_ds = results_ds.sortby("Latitude")
    if not results_ds["Longitude"].values[0] < results_ds["Longitude"].values[-1]:
        results_ds = results_ds.sortby("Longitude")
    return results_ds


def get_bounds(geos):
    """
    Tile index bounds for ``get_one_year``: sorted list of |truncated int| for each bound.

    NSIDC tiles are keyed by integer degrees; this derives the min/max north and west
    indices from the union of ``geos`` geometries.
    """
    combined = geos.unary_union
    bounds = combined.bounds  # (min_lon, min_lat, max_lon, max_lat)
    bounds_truncated_sorted = sorted(abs(int(i)) for i in bounds)
    return bounds_truncated_sorted


def clip_and_mean(ds_year, geos):
    """Attach CRS, clip ``ds_year`` to ``geos``, return spatial mean per day (handles lat/lon vs lon/lat dim names)."""
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
    October 1st of start_year. The function uses a daily frequency and formats dates as
    'YYYY-MM-DD'.

    Parameters:
    df (pd.DataFrame): DataFrame with an integer 'Day' index (e.g., 0 to 364 or 365).
    start_year (int): The starting year for the water year (e.g., 1984).

    Returns:
    pd.DataFrame: DataFrame indexed by calendar day string ('YYYY-MM-DD'); index name ``day``.
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


def get_mean(year, coords, geos, keep_all_stats=True):
    """One water year of basin-mean UCLA SWE (``SWE_Post``), as a DataFrame indexed by day."""
    ds_year = get_one_year(
        year, coords[0], coords[1], coords[2] + 1, coords[3] + 1,
        keep_all_stats=keep_all_stats,
    )
    mean_per_day = clip_and_mean(ds_year, geos)
    mean_df = mean_per_day.to_dataframe()

    if keep_all_stats:
        # Unstack Stats → columns SWE_Post_mean, SWE_Post_std, SWE_Post_median, SWE_Post_p25, SWE_Post_p75
        mean_df = mean_df["SWE_Post"].unstack("Stats")
        mean_df.columns = [f"SWE_Post_{name}" for name in STATS_NAMES]
    else:
        mean_df = mean_df[["SWE_Post"]]

    mean_df = assign_water_year_dates(mean_df, year)
    return mean_df


def get_gold_df(huc, year_start, year_end, overwrite=False, keep_all_stats=True):
    """
    Build gold CSV for one HUC: day + 5 UCLA SWE stats (mean, std, median, p25, p75).

    CSV columns: day, SWE_Post_mean, SWE_Post_std, SWE_Post_median, SWE_Post_p25, SWE_Post_p75.
    Output: ucla_swe_in_{huc}.csv in bucket snowml-gold.

    Inputs:
      huc: HUC ID (e.g. 12-digit HUC12 like '171100200501')
      year_start, year_end: water year range (e.g. 1984, 2022) → Oct 1 year_start through Sep 30 year_end
      overwrite: if True, overwrite existing file in S3
      keep_all_stats: if True (default), save all 5 stats; if False, save only mean (single column SWE_Post)
    """
    error_years = []
    time_start = time.time()
    geos = gg.get_geos(huc, str(len(str(huc))).zfill(2))
    coords = get_bounds(geos)
    results_df = pd.DataFrame()

    f_gold = f"ucla_swe_in_{huc}"
    b_gold = "snowml-gold"
    if du.isin_s3(b_gold, f"{f_gold}.csv") and not overwrite:
        print(f"File {f_gold} already exists in {b_gold}, skipping")
        return results_df, []

    for yr in range(year_start, year_end):
        try:
            mean_df = get_mean(yr, coords, geos, keep_all_stats=keep_all_stats)
            results_df = pd.concat([results_df, mean_df], axis=0)
        except Exception:
            print(f"Error processing year_{yr}, skipping")
            error_years.append(f"{huc}_{yr}")
    du.dat_to_s3(results_df, b_gold, f_gold, file_type="csv")
    du.elapsed(time_start)
    return results_df, error_years

def get_gold_multi(huc_list, year_start=1984, year_end=2021, overwrite=False, keep_all_stats=True):
    """
    Build gold CSVs for multiple HUCs. Each file: day + 5 stats (or single mean if keep_all_stats=False).

    Inputs:
      huc_list: list of HUC IDs (e.g. ['171100200501', '171100200502'])
      year_start, year_end: water year range (default 1984–2021)
      overwrite: if True, overwrite existing files in S3
      keep_all_stats: if True (default), each CSV has day + SWE_Post_mean, _std, _median, _p25, _p75
    """
    error_years = []
    for count, huc in enumerate(huc_list, start=1):
        print(f"processing huc {count}/{len(huc_list)}")
        _, new_error_years = get_gold_df(
            huc, year_start, year_end, overwrite=overwrite, keep_all_stats=keep_all_stats
        )
        error_years.extend(new_error_years)
    return error_years

def save_gold_df(huc, gold_df):
    """Upload an in-memory gold DataFrame to S3 (same key pattern as ``get_gold_df``)."""
    f_gold = f"ucla_swe_in_{huc}"
    b_gold = "snowml-gold"  # TO DO - Make dynamic
    du.dat_to_s3(gold_df, b_gold, f_gold, file_type="csv")

def get_gold_df_from_gdf(geos, geos_name, year_start, year_end, overwrite=False, keep_all_stats=True):
    """Like ``get_gold_df`` but for a custom GeoDataFrame; output key uses ``geos_name``."""
    error_years = []
    time_start = time.time()
    coords = get_bounds(geos)
    results_df = pd.DataFrame()

    f_gold = f"ucla_swe_in_{geos_name}"
    b_gold = "snowml-gold"
    if du.isin_s3(b_gold, f"{f_gold}.csv") and not overwrite:
        print(f"File {f_gold} already exists in {b_gold}, skipping")
        return results_df, []

    for yr in range(year_start, year_end):
        mean_df = get_mean(yr, coords, geos, keep_all_stats=keep_all_stats)
        results_df = pd.concat([results_df, mean_df], axis=0)
    du.dat_to_s3(results_df, b_gold, f_gold, file_type="csv")
    du.elapsed(time_start)
    save_gold_df(geos_name, results_df)
    return results_df, error_years

