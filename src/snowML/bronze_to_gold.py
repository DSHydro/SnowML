"""Module to calculate daily mean of climate data and save as a csv file"""


# pylint: disable=C0103

import importlib
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import xarray as xr
import data_utils as du
import set_data_constants as sdc


# define constants
VAR_DICT = sdc.create_var_dict()


def prep_bronze(var, bucket_dict = None):
    if bucket_dict is None:
        bucket_dict = sdc.create_bucket_dict("prod")

    b_bronze = bucket_dict.get("bronze")
    zarr_store_url = f's3://{b_bronze}/{var}_all.zarr'
    ds = xr.open_zarr(store=zarr_store_url, consolidated=True)
    if var != "swe":
        transform = du.calc_transform(ds)
        ds = ds.rio.write_transform(transform, inplace=True)
    else:
        ds.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace = True)
    ds.rio.write_crs("EPSG:4326", inplace=True)

    return ds


def create_mask(ds, row, crs):
    # Create a mask for the geometry
    mask = ds.rio.clip([row.geometry], crs, drop=True, invert=False)
    return mask

def ds_to_gold(ds, var):
    var_name = VAR_DICT.get(var)
    if not ds['day'].to_index().is_monotonic_increasing:
        ds = ds.sortby("day")
    daily_mean = ds[var_name].resample(day='D').mean(dim=['lat', 'lon'])
    daily_mean = daily_mean.assign_coords(day=ds['day'].resample(day='D').first())
    daily_mean = daily_mean.to_dataset(name=f"mean_{var}")
    daily_mean['day'] = ds['day']
    return daily_mean

def bronze_to_gold (geos, var, bucket_dict = None, overwrite = False):
    time_start = time.time()
    crs = geos.crs
    var_name = VAR_DICT.get(var)
    if bucket_dict is None:
        bucket_dict = sdc.create_bucket_dict("prod")

    # Loop through each geometry in the GeoDataFrame
    for idx, row in geos.iterrows():
        huc_id = row['huc_id']  # Extract the huc_id for naming
        print(f"Processing huc {idx+1} of {geos.shape[0]}, huc_id: {huc_id}")
        # process to silver
        small_ds = prep_bronze(var, bucket_dict=bucket_dict)
        df_silver = create_mask(small_ds, row, crs)
        # TO DO - ADD A SAVE GATE FOR SILVER

        # process to gold
        f_gold = f"mean_{var}_in_{huc_id}"
        b_gold = bucket_dict.get("gold")

        if du.isin_s3(b_gold, f"{f_gold}.csv") and not overwrite:
            print(f"File{f_gold} already exists in {b_gold}")
        else:
            ds_gold = ds_to_gold(df_silver, var)
            gold_df = ds_gold.to_dataframe()
            gold_df["huc_id"] = huc_id
            gold_df = gold_df[gold_df.columns.drop(["crs", "spatial_ref"])]
            if var in ["tmmn", "rmin"]:
                gold_df = gold_df.rename(columns={var_name: f"mean_{var_name}_min"})
            elif var in ["tmmx", "rmax"]:
                gold_df = gold_df.rename(columns={var_name: f"mean_{var_name}_max"})
            else:
                gold_df = gold_df.rename(columns={var_name: f"mean_{var_name}"})

            du.dat_to_s3(gold_df, b_gold, f_gold, file_type="csv")
            du.elapsed(time_start)
       