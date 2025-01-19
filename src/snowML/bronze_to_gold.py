""" Module to download and process University of Idaho Gridmet Data"""

import importlib
import time
import warnings
import xarray as xr
import data_utils as du
import set_data_constants as sdc

modules_to_reload = [sdc]
for m in modules_to_reload:
    importlib.reload(m)

# define constants
BUCKET_DICT = sdc.create_bucket_dict("prod")
VAR_DICT = sdc.create_var_dict()


def prep_bronze(geos, var, bucket_dict = BUCKET_DICT):
    # load_raw
    if var == "swe":
        b_bronze = bucket_dict.get(f"{var}-bronze")
    else:
        b_bronze = bucket_dict.get("wrf-bronze")
    zarr_store_url = f's3://{b_bronze}/{var}_all.zarr'
    #ds = xr.open_zarr(store=zarr_store_url, chunks={}, consolidated=True)
    ds = xr.open_zarr(store=zarr_store_url, consolidated=True)
    # print(print(ds.chunks))
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
    huc_id = row.iloc[0, 1] # get the id of the smaller huc unit
    silver_df["huc_id"] = huc_id
    return silver_df

def process_gold (silver_df, var, huc_id):

    if var == "swe":
        grouper = "time"
        var_name = "SWE"
    else:
        grouper = "day"
        var_name = VAR_DICT.get(var)
    gold_df = silver_df.groupby([grouper])[var_name].mean().reset_index() # TO DO: Fix Logic
    gold_df["huc_id"] = huc_id
    gold_df = gold_df.rename(columns={var_name: f"mean_{var_name}"})
    return gold_df


def process_all(geos, var, bucket_dict=BUCKET_DICT, save_sil = False, overwrite = False):
    time_start = time.time()
    num_hucs = geos.shape[0]

    # get and prep bronze data
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)  # TO DO: ADDRESS THE FUTURE WARNING
        small_ds = prep_bronze(geos, var, bucket_dict)

    # silver and gold processing
    gold_df_list = []
    for i in range (geos.shape[0]):
        row = geos.iloc[[i]]
        huc_id = row.iloc[0, 1] # get the id of the smaller huc uit
        print(f"Processing {var} for huc_id {huc_id}, {i+1} of {num_hucs}")
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
            silver_df = process_silver_row(small_ds, row)
            if save_sil:
                du.dat_to_s3(silver_df, b_silver, f_silver, file_type="csv")


        # gold_processing
        f_gold = f"mean_{var}_in_{huc_id}"
        if var == "swe":
            b_gold = bucket_dict.get(f"{var}-gold")
        else:
            b_gold = bucket_dict.get("wrf-gold")
        if du.isin_s3(b_gold, f"{f_gold}.csv") and not overwrite:
            print(f"File{f_gold} already exists in {b_gold}")
        else:
            gold_df = process_gold(silver_df, var, huc_id, b_gold)
            gold_df_list.append(gold_df)
            du.dat_to_s3(gold_df, b_gold, f_gold, file_type="csv")
        du.elapsed(time_start)

