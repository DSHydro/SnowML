# Module to process SWE GOLD

from glob import glob
import s3fs
import pandas as pd
import xarray as xr
import data_utils as du



# CONSTANTS
TEMP = "s3://{silver_bucket_nm}/mean_swe_{huc}_{year}.nc"
QUIET = False

def get_silver_file_paths(huc, bucket_name, pattern_template):
    """
    Generate a list of file paths for files in the specified S3 bucket
    based on a given pattern template.

    Parameters:
        hucs (list): A list of HUC values (e.g., ["101", "102"]).
        bucket_name (str): The name of the S3 bucket.
        pattern_template (str): A template for the file path pattern, 
                        using placeholdersfor HUC and year 
                        (e.g., "s3://{bucket_name}/mean_swe{huc}_{year}.nc").

    Returns:
        list: A list of strings representing the full paths to the matching files on S3,
              formatted for use with xr.open_mfdataset.
    """
    file_paths = []
    fs = s3fs.S3FileSystem()  # Initialize the S3 file system

    # Generate the pattern with wildcard for year
    pattern = pattern_template.replace("{year}", "*").format(silver_bucket_nm=bucket_name, huc=huc)

   # Get all matching files
    matching_files = fs.glob(pattern)
    file_paths.extend([f"s3://{file}" for file in matching_files])

    return file_paths

def silver_data_to_df(hucs, bucket_name, pattern_template):
    results = pd.DataFrame()
    fs = s3fs.S3FileSystem(anon=False)

    for huc in hucs:
        if not QUIET:
            print(f"processing huc {huc}")
        file_paths = get_silver_file_paths(huc, bucket_name, pattern_template)
        for f in file_paths:
            with fs.open(f) as file:
                ds = xr.open_dataset(file)
                df = ds.to_dataframe()
                ds.close()
            df["huc_id"] = huc
            results = pd.concat([results, df])
    return results


def get_swe_gold(huc_no, huc_lev, pattern_template=TEMP, \
                   shape_bucket_nm = "shape-bronze", \
                   silver_bucket_nm = "swe-silver", \
                   gold_bucket_nm = "swe-gold"):

    # Check if gold file already exists
    f_out = f"{huc_lev}_in_{huc_no}.csv"  # TO DO - DON'T HARDCODE EXTENTION
    if du.isin_s3(gold_bucket_nm, f_out):
        print(f"File{f_out} already exists in {gold_bucket_nm}")
        overwrite = input("Do you want to overwrite it? Type 'Y' for Yes or 'N' for No: ").strip().upper()
        if overwrite != 'Y':
            print("Operation cancelled by the user.")
            return

    # Upload shape file if it exists
    shape_file_nm = f"{huc_lev}_in_{huc_no}.geojson"
    if du.isin_s3(shape_bucket_nm, shape_file_nm):
        basin_gdf = du.s3_to_gdf (shape_bucket_nm, shape_file_nm)
        if not QUIET:
            print(f"Shapefile {shape_file_nm} uploaded from shape-bronze")
    else:
        print(f"No shape file found for {shape_file_nm} in {shape_bucket_nm}")
        print("Please upload shape file and try again")
        return

    hucs = basin_gdf["huc_id"].tolist()[0:2] # TO DO - Remove slice
    print(f"HUCs to process: {hucs}")
    swe_gold = silver_data_to_df(hucs, silver_bucket_nm, pattern_template)
    print(type(swe_gold))
    f_out = f"{huc_lev}_in_{huc_no}"
    du.dat_to_s3(swe_gold, gold_bucket_nm, f_out, file_type="csv")
    return swe_gold


#huc_lev =  "Huc10"
#huc_no = 18040009
#huc_no = 17020009
#swe_gold = get_swe_gold(huc_no, huc_lev)
