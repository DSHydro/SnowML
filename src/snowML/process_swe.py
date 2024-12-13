""" 
Module with function to download and process SWE data for use in model training.
Requires that you have set earth access credentials as environmental variables.
"""

import os, time,  s3fs
import xarray as xr
import data_utils as du


# CONSTANTs 
ROOT = "https://daacdata.apps.nsidc.org/pub/DATASETS/nsidc0719_SWE_Snow_Depth_v1/"
USERNAME = os.environ["EARTHDATA_USER"]
PASSWORD = os.environ["EARTHDATA_PASS"]
VARS_TO_KP = ["SWE"]


def process_swe_means(year, geo, vars_to_kp = VARS_TO_KP, 
                bronze_bucket_nm = "swebronze",
                silver_bucket_nm = "swe-silver"):
    
    """
    For the given year of swe data, mask by geo, and take the 
    mean of the specified variables by date. Return a pandas df saved to
    specified S3 bucket. 

    Args:
        year(int): The water year of the data to process.
        geo(gpd): the geometry for masking.  Must be a gpd file with a geometries 
            column and the second column containing huc id.  
        vars_to_kp: the variables for which to take and save the mean 
        bronze_bucket_name: the name of S3 bucket to save/retrieve raw data
        silver_bucket_name: the name of the S3 bucket to save the procesed output 

    Return:
       swe_mean_df (dataFrame): Means of variables of interest by date 

       """
    # check to see if the SWE file already exists 
    huc_id = geo.iloc[0, 1]
    f_out = f"mean_swe_{huc_id}_{year}.nc"
    if du.isin_s3(silver_bucket_nm, f_out):
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
    du.ds_to_s3(ds, silver_bucket_nm, f_out)


def process_swe_means_multi(years, geos, bronze_bucket_nm = "swebronze",
                silver_bucket_nm = "swe-silver"):
    time_start = time.time()
    for yr in years: 
        for i in range (geos.shape[0]):
            row = geos.iloc[[i]]
            process_swe_means(yr, row, bronze_bucket_nm=bronze_bucket_nm, 
                              silver_bucket_nm=silver_bucket_nm)

    elapsed_time = time.time() - time_start
    print(f"Elapsed time: {elapsed_time:.2f} seconds")   



def get_years():
    """
    Prompts the user for the first and last year of data to get.
    
    Ensures the first year is at least 1982 and the last year is no later than 2022.
    
    Returns:
        tuple: A list of years spanning from first year to last year as integers.
    """
    while True:
        try:
            # Prompt user for the first year
            first_year = int(input("Enter first year of data to get (1982 or later): "))
            # Prompt user for the last year
            last_year = int(input("Enter last year of data to get (no later than 2022): "))
            
            # Validate the year range
            if first_year < 1982:
                print("The first year must be 1982 or later. Please try again.")
                continue
            if last_year > 2022:
                print("The last year must be no later than 2022. Please try again.")
                continue
            if first_year > last_year:
                print("The first year must be less than or equal to the last year. Please try again.")
                continue
            
            return range(first_year, last_year)
        
        except ValueError:
            print("Invalid input. Please enter numeric values for the years.")


def get_sf_path(): 
    bucket_name = input("Enter the bucket name with the shape file for relevanat geos (e.g. 'shape-bronze')")
    file_path =  input ("Enter the file_path (e.g. 'examples/skagit_huc10.geojson')")    
    return(bucket_name, file_path)   

def main():
    years = get_years()
    bucket_name, file_path = get_sf_path()
    print(f"bucket name is: {bucket_name}, file path is {file_path}")
    geos = du.s3_to_gdf(bucket_name, file_path)
    print("geos")
    print("Processing . .. ")
    process_swe_means_multi(years, geos, bronze_bucket_nm = "swebronze",
                silver_bucket_nm = "swe-silver")


 # TO DOS
 
 # (1) May be faster to go from url -> ds rather than S3-> ds?
 # (2) Fix the buffer thingy in data utils 
 # (3) Figure out how to dynamically get shape files 
 # (4) Documentation and error messages 
 


    



