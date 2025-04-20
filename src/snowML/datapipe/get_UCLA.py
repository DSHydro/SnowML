# pylint: disable=C0103
""" Module to get UCLA data into a zarr file stored on S3"""

import json
import warnings
import time
import os
import xarray as xr
import pandas as pd
import s3fs
from snowML.datapipe import data_utils as du
from concurrent.futures import ProcessPoolExecutor, as_completed
from snowML.datapipe import set_data_constants as sdc

# define constants
VAR_DICT = sdc.create_var_dict()


def get_year(year):
    # Check if the year is between 1984 and 2020
    if not 1984 <= year <= 2020:
        raise ValueError("Year must be between 1984 and 2020")
    yr_start = year
    yr_end = str(year + 1)[-2:]
    f = f"SWE_UCLA/WUS_UCLA_SR_v01_N48_0W122_0_agg_16_WY{yr_start}_{yr_end}_SWE_SCA_POST.nc"
    ds = xr.open_dataset(f, engine="h5netcdf", chunks={"Day": -1, "Latitude": None, "Longitude": None})

    # Cleanup
    ds = ds[["SWE_Post"]]  # Drop the SCA variable

    # Rename coordinates
    ds = ds.rename({"Latitude": "lat", "Longitude": "lon", "Day": "day"})
    ds = ds.rename_vars({"SWE_Post": 'SWE'})  # Rename variable to 'SWE'

    # Promote non-dimensional coordinates to dimensions
    timestamps = pd.date_range(start=f'{year}-10-01', end=f'{int(year) + 1}-09-30')
    ds = ds.assign_coords(dict(
       day=timestamps,
       Stats=['mean', 'std', 'median', '25%', '75%']  # Order known from dataset documentation
    ))
    ds = ds.sel(Stats="mean").drop_vars("Stats")

    # Sort coordinates if not already sorted
    if not ds['lat'].to_index().is_monotonic_increasing:
        ds = ds.sortby("lat")
    if not ds['lon'].to_index().is_monotonic_increasing:
        ds = ds.sortby("lon")

    return ds


def download_multiple_years(start_year, end_year, var, s3_bucket, append_to=False):
    """
    Downloads and processes data for multiple years, saving the results to a
    Zarr file on S3.

    Parameters:
       start_year (int): The starting year of the range to download.
       end_year (int): The ending year of the range to download.
       var (str): The variable name to download and process.
       s3_bucket (str): The name of the S3 bucket to store Zarr file.
       append_to (bool, optional): If True, append to an existing Zarr file.
          If False, create a new Zarr file. Default is False.

    Returns:
       str: The S3 path to the saved Zarr file.

    Raises:
       ValueError: If the S3 Zarr path exists and append_to is False.

    Notes:
    - The function tracks completed years using a local progress file.
    - Data is processed and saved year by year, either creating a new Zarr
      file or appending to an existing one.
    """
    time_start = time.time()
    s3_path = f"{var}_all_UCLA.zarr"

    # Initialize the S3 filesystem
    fs = s3fs.S3FileSystem()

    # Check if the S3 Zarr path already exists
    if fs.exists(f"s3://{s3_bucket}/{s3_path}") and not append_to:
        print(f"Warning: The path s3://{s3_bucket}/{s3_path} already exists.")
        print("Skipping file. Set append_to = True if you intended to append.")
        return "no path created"

    # Define some data-specific attributes
    dim_to_concat = "day"

    # Load progress from a local file to keep track of completed years
    progress_file = f"{var}_progress.json"
    completed_years = set()
    if os.path.exists(progress_file):
        with open(progress_file, "r", encoding="utf-8") as f:
            completed_years = set(json.load(f))
    print(f"Resuming with completed years: {sorted(completed_years)}")

    # Process years sequentially
    for year in range(start_year, end_year + 1):
        if year in completed_years:
            print(f"Skipping year {year} (already processed)")
            continue
        print(f"Processing year: {year}")
        ds = get_year(year)
        ds = ds.chunk({dim_to_concat: -1, "lat": None, "lon": None})

        # Append to the existing Zarr file on S3
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            if not fs.exists(f"s3://{s3_bucket}/{s3_path}"):
                # Create a new Zarr file for the first year
                ds.to_zarr(f"s3://{s3_bucket}/{s3_path}", mode="w", consolidated=True)
                print(f"Created new Zarr file at s3://{s3_bucket}/{s3_path}")
                completed_years.add(year)
            else:
                # Append data to the existing Zarr file
                ds.to_zarr(f"s3://{s3_bucket}/{s3_path}", mode="a",
                     append_dim=dim_to_concat, consolidated=True)
                print(f"Appended year {year} to s3://{s3_bucket}/{s3_path}")
                completed_years.add(year)
                with open(progress_file, "w", encoding="utf-8") as f:
                    json.dump(sorted(completed_years), f)
                du.elapsed(time_start)

    print(f"Final dataset saved to s3://{s3_bucket}/{s3_path}")
    return s3_path


def get_bronze_UCLA(
   bronze_bucket_nm="snowml-bronze",
   var="swe",
   year_start=1984,
   year_end=2020,
   append_to=False
):
    """
    Downloads raw data for the specified variable and saves it to an S3 bucket.

    Parameters:
    var (str): The variable for which data is to be downloaded.
    bronze_bucket_nm (str): The name of the S3 bucket where the data will
      be saved.
    year_start (int, optional): The starting year for the data download.
      Must be >= 1983. Default is 1995.
    year_end (int, optional): The ending year for the data download.
      Must be <= 2025. Default is 2023.
    append_to (bool, optional): If True, appends data to the
      existing data in the S3 bucket. Default is False.

    Returns:
    str: The S3 path where the data has been saved.

    Raises:
    ValueError: If year_start is less than 1983 or year_end is greater than 2025.
    """
    # Validate year input
    if year_start < 1984 or year_end > 2021:
        raise ValueError("Year start must be >= 1984; year end must be < 2021")

    # Download raw and save to S3 directly
    s3_path = download_multiple_years(year_start, year_end, var, bronze_bucket_nm, append_to=append_to)

    return s3_path

def prep_bronze(var, bucket_dict = None, append_start = None):
    """
    Prepares a dataset from a Zarr file stored in an S3 bucket.

    This function opens a Zarr file from an S3 bucket, processes the 
    dataset,and writes spatial information to it. The dataset is then closed 
    and returned.

    Parameters:
        var (str): The variable name to be processed.
        bucket_dict (dict, optional): A dictionary containing bucket names. 
            If None, a default bucket dictionary for the "prod" environment
            is created.

        Returns:
            xarray.Dataset: The processed dataset with spatial information.

        Raises:
            KeyError: If the "bronze" key is not found in the bucket_dict.
        """
    if bucket_dict is None:
        bucket_dict = sdc.create_bucket_dict("prod")
    b_bronze = bucket_dict["bronze"]
    zarr_store_url = f's3://{b_bronze}/{var}_all_UCLA.zarr'

    # Open the Zarr file directly with storage options
    ds = xr.open_zarr(zarr_store_url, consolidated=True, storage_options={'anon': False})
    #print("Opened Zarr file successfully.")

    # Process the dataset as needed
    if var != "swe":
        transform = du.calc_transform(ds)
        ds = ds.rio.write_transform(transform, inplace=True)
    else:
        ds.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)

    ds.rio.write_crs("EPSG:4326", inplace=True)
    if append_start is not None:
        ds = ds.sel(day=slice(append_start, None))
    ds.close()  # Close the dataset after processing

    return ds

def create_mask(ds, row, crs):
    """
    Create a mask for the given geometry.

    Parameters:
    ds (xarray.Dataset): The dataset to be masked.
    row (geopandas.GeoSeries): The row containing the geometry to be used for masking.
    crs (str or dict): The coordinate reference system of the geometry.

    Returns:
    xarray.Dataset: The masked dataset.
    """
    mask = ds.rio.clip([row.geometry], crs, drop=True, invert=False)
    return mask

def ds_to_gold(ds, var):
    """
    Converts a dataset to dawgs-gold standard format by resampling and 
    averaging the specified variable.

    Parameters:
    ds (xarray.Dataset): The input dataset containing the data to be processed.
    var (str): The variable name to be processed, should be a key in VAR_DICT.

    Returns:
    xarray.Dataset: A new dataset containing the daily mean of the specified variable, 
                    with coordinates adjusted to match the original dataset.
    """
    var_name = VAR_DICT.get(var)
    if not ds['day'].to_index().is_monotonic_increasing:
        ds = ds.sortby("day")
    daily_mean = ds.mean(dim=['lat', 'lon'])
    return daily_mean


def process_row(row, var, idx, bucket_dict, crs, var_name, overwrite, append_start):
    """
    Processes a single row(geometry) of data from bronze to gold.

    Args:
        row (dict): A series containing the data for a single geometry, 
            including 'huc_id'.
        var (str): The variable name to process (e.g., 'tmmn', 'rmax').
        idx (int): The index of the current row being processed.
        bucket_dict (dict): A dictionary containing S3 bucket information.
        crs (str): Coordinate reference system information.
        var_name (str): The name of the variable to be used in the 
            final gold dataset.
        overwrite (bool): Flag indicating whether to 
            overwrite existing files in the gold bucket.
        append(bool): Flag indicating whether this is (new) additional gold data

    Returns:
        None
    """
    huc_id = row['huc_id']
    print(f"Processing huc {idx+1}, huc_id: {huc_id}")

    # process to silver
    small_ds = prep_bronze(var, bucket_dict=bucket_dict, append_start = append_start)
    df_silver = create_mask(small_ds, row, crs)
    print(f"Processing huc {idx+1}, huc_id: {huc_id} to silver completed")

    # process to gold
    if append_start is not None:
        f_gold = f"mean_{var}_in_{huc_id}_UCLA_append"
    else:
        f_gold = f"mean_{var}_in_{huc_id}_UCLA"
    b_gold = bucket_dict.get("gold")
    


    if du.isin_s3(b_gold, f"{f_gold}.csv") and not overwrite:
        print(f"File{f_gold} already exists in {b_gold}")
    else:
        ds_gold = ds_to_gold(df_silver, var)
        print("ds_gold_created")
        gold_df = ds_gold.to_dataframe()
        print(f"Processing huc {idx+1}, huc_id: {huc_id} to gold completed")
        gold_df["huc_id"] = huc_id
        gold_df = gold_df[gold_df.columns.drop(["spatial_ref"])]
        du.dat_to_s3(gold_df, b_gold, f_gold, file_type="csv")
        #du.elapsed(time_start)

def process_geos(
    geos,
    var,
    bucket_dict= None,
    overwrite=False,
    max_wk = 8,
    append_start = None):
    """
    Processes geographical data in parallel using a ProcessPoolExecutor.

    Args:
        geos (GeoDataFrame): A GeoDataFrame containing the data to be processed.
            var (str): The variable to be processed, used to retrieve the 
                variable name from VAR_DICT.
            bucket_dict (dict, optional): A dictionary containing bucket info. 
                If None, a default bucket dictionary is created. 
            overwrite (bool, optional): A flag indicating whether to 
                overwrite existing data. Defaults to False.
            max_wk (int, optional): The maximum number of worker processes to 
                use for parallel processing. Defaults to 8.

        Returns:
            None
        """
    crs = geos.crs
    var_name = VAR_DICT.get(var)
    if bucket_dict is None:
        bucket_dict = sdc.create_bucket_dict("prod")

    # Use ProcessPoolExecutor to parallelize the tasks
    with ProcessPoolExecutor(max_workers=max_wk) as executor:
        futures = [
            executor.submit(process_row, row, var, idx, bucket_dict, crs,
                            var_name, overwrite, append_start = append_start)
            for idx, row in geos.iterrows()
        ]

        for future in as_completed(futures):
            future.result()  # Wait for the task to complete
