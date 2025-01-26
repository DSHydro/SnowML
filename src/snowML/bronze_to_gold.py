"""Module to calculate daily mean of climate data and save as a csv file"""


# pylint: disable=C0103

import importlib
import time
import xarray as xr
import data_utils as du
import set_data_constants as sdc

modules_to_reload = []
for m in modules_to_reload:
    importlib.reload(m)

# define constants
VAR_DICT = sdc.create_var_dict()


def prep_bronze(geo, var, bucket_dict = None):
    if bucket_dict is None:
        bucket_dict = sdc.create_bucket_dict("prod")

    # load_raw
    if var == "swe":
        b_bronze = bucket_dict.get(f"{var}-bronze")
    else:
        b_bronze = bucket_dict.get("wrf-bronze")
    zarr_store_url = f's3://{b_bronze}/{var}_all.zarr'
    ds = xr.open_zarr(store=zarr_store_url, consolidated=True)

    # Perform first cut crude filter
    min_lon, min_lat, max_lon, max_lat = geo.bounds
    small_ds = du.crude_filter(ds, min_lon, min_lat, max_lon, max_lat)

    # Sort if necessary
    if not small_ds['lat'].to_index().is_monotonic_increasing:
        small_ds = small_ds.sortby("lat")

    if var != "swe":
        transform = du.calc_transform(small_ds)
        small_ds = small_ds.rio.write_transform(transform, inplace=True)
    else:
        small_ds.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace = True)
    small_ds.rio.write_crs("EPSG:4326", inplace=True)
    return small_ds


def create_mask(ds, row, crs):
    # Create a mask for the geometry
    mask = ds.rio.clip([row.geometry], crs, drop=False, invert=False)
    return mask

# def view_slice(ds):
#     time_slice = slice('1996-12-01', '1996-12-02')
#     ds_subset = ds.sel(time=time_slice)
#     print(ds_subset)
#     df = ds_subset.to_dataframe()
#     df.dropna(subset=['SWE'], inplace=True)
#     print(df)


def ds_to_gold(ds, var):
    if var == "swe":
        var_name = "SWE"
        daily_mean = ds[var_name].resample(time='D').mean(dim=['lat', 'lon'])
        daily_mean = daily_mean.assign_coords(time=ds['time'].resample(time='D').first())
        daily_mean = daily_mean.to_dataset(name=f"mean_{var_name}")
        daily_mean['time'] = ds['time']
    else:
        var_name = VAR_DICT.get(var)
        if not ds['day'].to_index().is_monotonic_increasing:
            ds = ds.sortby("day")
        daily_mean = ds[var_name].resample(day='D').mean(dim=['lat', 'lon'])
        daily_mean = daily_mean.assign_coords(day=ds['day'].resample(day='D').first())
        daily_mean = daily_mean.to_dataset(name=f"mean_{var_name}")
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
        geo = row.geometry  # Extract the geometry
        small_ds = prep_bronze(geo, var, bucket_dict=bucket_dict)
        df_silver = create_mask(small_ds, row, crs)
        # TO DO - ADD A SAVE GATE FOR SILVER

        # process to gold
        f_gold = f"mean_{var}_in_{huc_id}"
        if var == "swe":
            b_gold = bucket_dict.get(f"{var}-gold")
        else:
            b_gold = bucket_dict.get("wrf-gold")

        if du.isin_s3(b_gold, f"{f_gold}.csv") and not overwrite:
            print(f"File{f_gold} already exists in {b_gold}")
        else:
            ds_gold = ds_to_gold(df_silver, var)
            du.elapsed(time_start)
            #ds_gold = ds_gold.chunk({'time': -1})
            #s3_path = f"mean_{var}_in{huc_id}.zarr"
            #ds_gold.to_zarr(f"s3://{b_gold}/{s3_path}", mode="w", consolidated=True)
            gold_df = ds_gold.to_dataframe()
            gold_df["huc_id"] = huc_id
            gold_df = gold_df[gold_df.columns.drop("crs")]
            if var in ["tmmn", "rmin"]:
                gold_df = gold_df.rename(columns={var_name: f"mean_{var_name}_min"})
            elif var in ["tmmx", "rmax"]:
                gold_df = gold_df.rename(columns={var_name: f"mean_{var_name}_max"})
            else: 
                gold_df = gold_df.rename(columns={var_name: f"mean_{var_name}"})
            du.dat_to_s3(gold_df, b_gold, f_gold, file_type="csv")
            du.elapsed(time_start)
