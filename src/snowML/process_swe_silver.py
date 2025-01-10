# pylint: disable=C0103

""" 
Module with function to download and process SWE data for use in model training.
Requires that you have set earth access credentials as environmental variables.
"""

import os
import time
import s3fs
import xarray as xr
import data_utils as du


# CONSTANTS
ROOT = "https://daacdata.apps.nsidc.org/pub/DATASETS/nsidc0719_SWE_Snow_Depth_v1/"
USERNAME = os.environ["EARTHDATA_USER"]
PASSWORD = os.environ["EARTHDATA_PASS"]
VARS_TO_KP = ["SWE"]
FILE_EXTENTION_MAP = {
        "csv": ".csv",
        "parquet": ".parquet",
        "netcdf": ".nc"
    }


def process_swe_means(year, geo, vars_to_kp = VARS_TO_KP,
                bronze_bucket_nm = "swebronze",
                silver_bucket_nm = "swe-silver",
                output_type = "netcdf"):

    """
    For the given year of swe data, mask by geo, and take the 
    mean of the specified variables by date. Save the file in the specified 
    format into the specified S3 bucket. 

    Args:
        year(int): The water year of the data to process.
        geo(gpd): the geometry for masking.  Must be a gpd file with a geometries 
            column and the second column containing huc id.  
        vars_to_kp: the variables for which to take and save the mean 
        bronze_bucket_name: the name of S3 bucket to save/retrieve raw data
        silver_bucket_name: the name of the S3 bucket to save the procesed output 
        output_type (str): The format to save the file ('csv', 'parquet', or 'netcdf').

    Return:
       None

       """
    # check to see if the SWE file already exists
    huc_id = geo.iloc[0, 1]
    ext = FILE_EXTENTION_MAP[output_type]
    f_out = f"mean_swe_{huc_id}_{year}"
    f_out_long = f"mean_swe_{huc_id}_{year}{ext}"
    if du.isin_s3(silver_bucket_nm, f_out_long):
        print(f"file {f_out} already processed, skipping processing")
        return None

    # download SWE data if not yet avail; upload to xArray(ds)
    file_name = f"4km_SWE_Depth_WY{year}_v01.nc"
    du.url_to_s3(ROOT, file_name, bronze_bucket_nm, requires_auth=True, \
                 username=USERNAME, password=PASSWORD)


    # load the dataset, pre-process, clip by geo, take-mean, close ds
    s3_path = f"s3://{bronze_bucket_nm}/{file_name}"
    fs = s3fs.S3FileSystem(anon=False)
    with fs.open(s3_path) as f:
        ds = xr.open_dataset(f)
        ds = ds[vars_to_kp]
        ds = ds.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace = True)
        ds = ds.rio.write_crs("EPSG:4326", inplace=True)
        ds = du.filter_by_geo (ds, geo)
        ds = du.ds_mean(ds)
        ds.close()

    # save result to s3
    du.dat_to_s3(ds, silver_bucket_nm, f_out, file_type=output_type)
    return None


def process_swe_means_multi(years, geos, bronze_bucket_nm = "swebronze",
                silver_bucket_nm = "swe-silver", output_type = "netcdf"):
    time_start = time.time()
    for yr in years:
        for i in range (geos.shape[0]):
            row = geos.iloc[[i]]
            process_swe_means(yr, row, bronze_bucket_nm=bronze_bucket_nm,
                              silver_bucket_nm=silver_bucket_nm,
                              output_type=output_type)

    elapsed_time = time.time() - time_start
    print(f"Elapsed time: {elapsed_time:.2f} seconds")



 # TO DOS

 # (1) May be faster to go from url -> ds rather than S3-> ds?
 # (2) Fix the buffer thingy in data utils
 # (4) Documentation and error messages
 