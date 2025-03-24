"""
Module to process climate data from bronze to gold standard.

This module includes functions to prepare datasets from Zarr files stored in S3 
buckets,create masks for specific geometries, and convert datasets to a 
standardized format by resampling and averaging specified variables. 
The processed data is then saved as CSV files.
"""


# pylint: disable=C0103

from concurrent.futures import ProcessPoolExecutor, as_completed
import xarray as xr
from snowML.datapipe import data_utils as du
from snowML.datapipe import set_data_constants as sdc

# define constants
VAR_DICT = sdc.create_var_dict()

def prep_bronze(var, bucket_dict = None):
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
    zarr_store_url = f's3://{b_bronze}/{var}_all.zarr'

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
    daily_mean = ds[var_name].resample(day='D').mean(dim=['lat', 'lon'])
    daily_mean = daily_mean.assign_coords(day=ds['day'].resample(day='D').first())
    daily_mean = daily_mean.to_dataset(name=f"mean_{var}")
    daily_mean['day'] = ds['day']
    return daily_mean


def process_row(row, var, idx, bucket_dict, crs, var_name, overwrite):
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

    Returns:
        None
    """
    huc_id = row['huc_id']
    print(f"Processing huc {idx+1}, huc_id: {huc_id}")

    # process to silver
    small_ds = prep_bronze(var, bucket_dict=bucket_dict)
    df_silver = create_mask(small_ds, row, crs)

    # process to gold
    f_gold = f"mean_{var}_in_{huc_id}"
    b_gold = bucket_dict.get("gold")


    if du.isin_s3(b_gold, f"{f_gold}.csv") and not overwrite:
        print(f"File{f_gold} already exists in {b_gold}")
    else:
        ds_gold = ds_to_gold(df_silver, var)
        gold_df = ds_gold.to_dataframe()
        #print(f"Processing huc {idx+1}, huc_id: {huc_id} to gold completed")
        gold_df["huc_id"] = huc_id
        gold_df = gold_df[gold_df.columns.drop(["crs", "spatial_ref"])]
        if var in ["tmmn", "rmin"]:
            gold_df = gold_df.rename(columns={var_name: f"mean_{var_name}_min"})
        elif var in ["tmmx", "rmax"]:
            gold_df = gold_df.rename(columns={var_name: f"mean_{var_name}_max"})
        else:
            gold_df = gold_df.rename(columns={var_name: f"mean_{var_name}"})

        du.dat_to_s3(gold_df, b_gold, f_gold, file_type="csv")
        #du.elapsed(time_start)

def process_geos(geos, var, bucket_dict= None, overwrite=False, max_wk = 8):
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
            executor.submit(process_row, row, var, idx, bucket_dict, crs, var_name, overwrite)
            for idx, row in geos.iterrows()
        ]

        for future in as_completed(futures):
            future.result()  # Wait for the task to complete
