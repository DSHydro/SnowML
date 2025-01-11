"""Module that retrieves huc geometries for a given huc id and
optionally saves the file into an S3 bucket defined by the user."""


import easysnowdata
import pandas as pd
import geopandas as gpd 
import ee
import numpy as np
import boto3
import time, os
from shapely.geometry import box
import data_utils as du

# function that creates an outer boundary box for a given geometry
def create_bbox (sp): 
    minx, miny, maxx, maxy = sp.bounds
    bbox = box(minx, miny, maxx, maxy)
    return bbox

# function that adds a crude bounding box to each geometry in the gdf
def add_bbox (gdf): 
    result = gdf.copy()
    result["bbox"] = gdf["geometry"].apply(create_bbox)
    return result


def clean_gdf(gdf, huc_lev_id):
    suff = huc_lev_id.lstrip("0")
    col_nm = f"huc{suff}"
    gdf = gdf.sort_values(by=col_nm)
    gdf["huc_level"] = huc_lev_id 
    gdf = gdf.rename(columns={col_nm: "huc_id"}).reset_index(drop=True)
    return gdf

def get_huc(bbox, huc_nm, huc_id, huc_lev_nxt):
    #ee.Authenticate()
    #ee.Initialize(project = PROJ)
    gdf = easysnowdata.hydroclimatology.get_huc_geometries(bbox_input=bbox, huc_level=huc_lev_nxt)
    
    #discard overinclusive entries that don't match starting string
    huc_str=gdf.iloc[:, 1]
    idx = huc_str.str.startswith(huc_id)
    gdf = gdf.loc[idx]
    
    print(f"There are {gdf.shape[0]} HUC{huc_lev_nxt} regions within {huc_nm}, huc region: {huc_id}")
    return gdf

def calc_huc_next(huc_id):
    if isinstance(huc_id, pd.Series):
        huc_id = huc_id.iloc[0]
    next_huc_id = int(huc_id) + 2
    # Convert to string and add a leading zero if it's a single digit
    return f"{next_huc_id:02}"


def get_next_hucs(huc_input, huc_lev_nxt):
    huc_input = add_bbox(huc_input)
    row = huc_input.iloc[0]
    bbox = row["bbox"] 
    huc_id = row["huc_id"]
    huc_nm = row["name"]
    huc_out = get_huc(bbox, huc_nm, huc_id, huc_lev_nxt)
    huc_out = clean_gdf(huc_out, huc_lev_nxt)
    return huc_out 


def get_huc(bbox, huc_nm, huc_id, huc_lev_nxt, quiet = True):
    gdf = easysnowdata.hydroclimatology.get_huc_geometries(bbox_input=bbox, huc_level=huc_lev_nxt)
    
    #discard overinclusive entries that don't match starting string
    huc_str=gdf.iloc[:, 1]
    idx = huc_str.str.startswith(huc_id)
    gdf = gdf.loc[idx]
    if not quiet: 
        print(f"There are {gdf.shape[0]} HUC{huc_lev_nxt} regions within {huc_nm}, huc region: {huc_id}")
    return gdf


def parse_huc(huc_string):
    """
    Parses the input string into a list of substrings, progressively increasing by two characters.

    Args:
        huc_string (str or int): The input string or integer to parse.

    Returns:
        list: A list of substrings containing the first 2, 4, 6, ..., n characters of the string.
    """
    if isinstance(huc_string, int):
        huc_string = str(huc_string)

    if not isinstance(huc_string, str):
        raise ValueError("Input must be a string or an integer")

    if not (2 <= len(huc_string) <= 12) or not huc_string.isdigit():
        raise ValueError("Input must be a string of digits with length between 2 and 12")

    if len(huc_string) % 2 != 0:
        raise ValueError("Input must have an even number of digits")

    valid_first_digits = {f"{i:02}" for i in range(1, 23)}
    if huc_string[:2] not in valid_first_digits:
        raise ValueError("The first two digits must be in the range [01, 02, ..., 22]")

    # Create the list using a comprehension that slices up to every even index.
    return [huc_string[:i] for i in range(2, len(huc_string) + 1, 2)]




def get_huc_geo(input, huc_02_gdf, quiet = False): 
    try: 
        ee.Authenticate()
    except: 
        "Error authenticating into earth engine. Please check your credentials"
   
    print("This function works iteratively, by getting geometries for progressively smaller huc regions")
   
    # initialize with the huc 02_gdf  
    huc_parsed = parse_huc(input)
    huc_gdf = clean_gdf(huc_02_gdf, "02")
    huc_id = huc_parsed.pop(0)
    huc_gdf  = huc_gdf[huc_gdf["huc_id"] == huc_id]
    

    # iteratively get smaller huc geometries 
    
    while huc_parsed: 
        huc_id = huc_parsed.pop(0)
        huc_lev = huc_gdf["huc_level"]
        huc_lev_next = calc_huc_next(huc_lev)

        if not quiet: 
            print(f"Processing geometries for HUC {huc_id}. This step may take a while")
        huc_gdf_next = get_next_hucs(huc_gdf, huc_lev_next)
        huc_gdf = huc_gdf_next[huc_gdf_next["huc_id"] == huc_id]
        if huc_gdf.empty: 
            print (f"Error - please check if {huc_id} is an accurate huc code")
            return None
            
    return huc_gdf


# Returns a tuple of geopandas dataframes 
def get_geos(input, final_huc_levs, save = True, bucket_name = "shape-bronze"):
    results = ()
    huc_02_gdf = du.s3_to_gdf("shape-bronze", "all_huc02.geojson")
    outer_huc_gdf = get_huc_geo(input, huc_02_gdf)
    while final_huc_levs: 
        final_huc_lev = final_huc_levs.pop(0)
        if isinstance(final_huc_lev, int):
            final_huc_lev = str(final_huc_lev)
        lower_huc_gdf = get_next_hucs(outer_huc_gdf, final_huc_lev)
        if save: 
            f_out = f"Huc{final_huc_lev}_in_{input}.geojson"
            lower_huc_gdf.to_file(f_out, driver="GeoJSON")
            s3_client = boto3.client('s3')
            s3_client.upload_file(f_out, bucket_name, f_out)
            os.remove(f_out)
    
        print(f"File {f_out} successfully uploaded to {bucket_name}")
        results = results + (lower_huc_gdf,)
    
    return results




# TO DO
# 1 Error checking, testing and documentation

