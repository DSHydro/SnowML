# pylint: disable=C0103,C0116

import os
import time
import pandas as pd
import xarray as xr
import data_utils as du


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

def bronze_to_silver(geos, ds):
    ds.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace = True)
    ds = ds.rio.write_crs("EPSG:4326", inplace=True)
    results = pd.DataFrame()
    for i in range (geos.shape[0]):
        row = geos.iloc[[i]]
        huc = row.iloc[0]["huc_id"]
        if not QUIET:
            print(f"Processing silver data huc: {huc}")
        ds_filter = du.filter_by_geo (ds, row)
        df= ds_filter.to_dataframe()
        df["huc_id"] = huc
        results = pd.concat([results, df])
    return results


def process_all (huc_id, huc_lev, overwrite = False):
    # track processing time
    time_start = time.time()

    # get shape file
    geos = du.get_basin_geos(huc_lev, huc_id)

    # get & save bronze swe data
    for years in YEAR_LIST:
        f_bronze = f"raw_swe_unmasked_in_{huc_id}_{min(years)}_to_{max(years)}"
        if du.isin_s3(BRONZE_BUCKET_NM, f"{f_bronze}.nc"):
            print(f"File{f_bronze} already exists in {BRONZE_BUCKET_NM}")
            if not overwrite:
                bronze_ds = du.s3_to_ds(BRONZE_BUCKET_NM, f"{f_bronze}.nc")
            else:
                bronze_ds = raw_to_bronze_multi(geos, years)
                du.dat_to_s3(bronze_ds, BRONZE_BUCKET_NM, f_bronze)
        else:
            bronze_ds = raw_to_bronze_multi(geos, years)
            du.dat_to_s3(bronze_ds, BRONZE_BUCKET_NM, f_bronze)

        elapsed_time = time.time() - time_start
        print(f"Elapsed time : {elapsed_time:.2f} seconds")

    # get & save silver data
    for years in YEAR_LIST:
        f_silver = f"raw_swe_in_{huc_id}_{min(years)}_to_{max(years)}"
        if du.isin_s3(SILVER_BUCKET_NM, f"{f_silver}.csv"):
            print(f"File{f_silver} already exists in {SILVER_BUCKET_NM}")
            if not overwrite:
                silver_df = du.s3_to_df(f"{f_silver}.csv", SILVER_BUCKET_NM)
            else:
                silver_df = bronze_to_silver(geos, bronze_ds)
                du.dat_to_s3(silver_df, SILVER_BUCKET_NM, f_silver, file_type="csv")
        else:
            silver_df = bronze_to_silver(geos, bronze_ds)
            du.dat_to_s3(silver_df, SILVER_BUCKET_NM, f_silver, file_type="csv")

        elapsed_time = time.time() - time_start
        print(f"Elapsed time : {elapsed_time:.2f} seconds")

    # get & save gold data
    for years in YEAR_LIST:
        gold_df = silver_df.groupby(['time', 'huc_id'])['SWE'].mean().reset_index()
        f_out = f"mean_swe_{huc_lev}_in_{huc_id}_{min(years)}_to{max(years)}"
        du.dat_to_s3(gold_df, GOLD_BUCKET_NM, f_out, file_type="csv")

    return None


def process_all_years(huc_id, huc_lev):
    

    for years in year_list: 
        process_all_medals(huc_id, huc_lev, years)






