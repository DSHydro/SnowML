"""Module that retrieves huc geometries for a given huc id and
savess to a local file, or optionally to an S3 bucket
defined by the user."""

import os
import json
import boto3
import ee
import geopandas as gpd
import logging


def get_geos(huc_id, final_huc_lev, s3_save = False, bucket_nm = "shape-bronze"):
    """
    Retrieves and processes geographic data from the USGS Watershed Boundary Dataset (WBD) 
    for a specified Hydrologic Unit Code (HUC) level and ID, and optionally saves the 
    resulting GeoJSON to an S3 bucket.

    Parameters:
    huc_id (str or int): The starting HUC ID to filter the dataset.
    final_huc_lev (str): The final HUC level to retrieve. Must be one of 
        ['02', '04', '06', '08', '10', '12'].
    s3_save (bool, optional): Whether to save the resulting GeoJSON to an 
        S3 bucket. Defaults to True.
    bucket_nm (str, optional): The name of the S3 bucket to save the file to. 
        Defaults to "shape-bronze".

    Returns:
    gpd.GeoDataFrame: A GeoDataFrame containing the filtered geographic data.

    Raises:
    ValueError: If there is an issue with Earth Engine credentials, or if the
    input HUC levels are invalid.
    """

    # make sure earth engine credentials are working
    try:
        ee.Authenticate()
        ee.Initialize(project="ee-frostydawgs")
    except Exception as exc:
        raise ValueError("Problem with earth link credentials") from exc

    # validate inputs
    huc_levs = ['02', '04', '06', '08', '10', '12']
    if not final_huc_lev in huc_levs:
        raise ValueError(f"Final Huc Levels must one of {huc_levs}")
    if isinstance(huc_id, int):
        huc_id = str(huc_id)
    huc_lev_start = str(len(huc_id)).zfill(2)
    if not huc_lev_start in huc_levs:
        raise ValueError("Huc id must be an even number between 2 and 12")

    # Define the feature collection
    collection = ee.FeatureCollection(f'USGS/WBD/2017/HUC{final_huc_lev}')

    # Filter the collection to only include features within top level huc
    final_huc_lev = final_huc_lev.lstrip('0')
    filtered_collection = collection.filter(ee.Filter.stringStartsWith(f"huc{final_huc_lev}", huc_id))


    # Extract HUC IDs and geometries
    output = filtered_collection.map(lambda feature: feature.select([f"huc{final_huc_lev}"]))

    # Convert to a GeoJSON dictionary
    geojson_dict = output.getInfo()

    # Modify the GeoJSON dictionary
    for feature in geojson_dict["features"]:
        feature.pop("id")
        feature["properties"]["huc_id"] = feature["properties"].pop(f"huc{final_huc_lev}")

     # Sort features by 'huc_id'
    geojson_dict["features"].sort(key=lambda x: x["properties"]["huc_id"])

    # Save the GeoJSON dictionary to a file
    f_out = f"Huc{final_huc_lev}_in_{huc_id}.geojson"
    with open(f_out, "w") as file:
        json.dump(geojson_dict, file)

    gdf = gpd.read_file(f_out)

    if s3_save:
        # Save the file to S3
        s3_client = boto3.client('s3')
        s3_client.upload_file(f_out, bucket_nm, f_out)
        print(f"File {f_out} successfully uploaded to {bucket_nm}")
    os.remove(f_out)
    return gdf


def get_geos_with_name(huc_id, final_huc_lev, s3_save=False, bucket_nm="shape-bronze"):
    """
    Retrieves and processes geographic data from the USGS Watershed Boundary Dataset (WBD)
    for a specified Hydrologic Unit Code (HUC) level and ID, extracts the name of the HUC,
    and optionally saves the resulting GeoJSON to an S3 bucket.

    Parameters:
    huc_id (str or int): The starting HUC ID to filter the dataset.
    final_huc_lev (str): The final HUC level to retrieve. Must be one of 
        ['02', '04', '06', '08', '10', '12'].
    s3_save (bool, optional): Whether to save the resulting GeoJSON to an 
        S3 bucket. Defaults to False.
    bucket_nm (str, optional): The name of the S3 bucket to save the file to. 
        Defaults to "shape-bronze".

    Returns:
    gpd.GeoDataFrame: A GeoDataFrame containing the filtered geographic data with the HUC name.

    Raises:
    ValueError: If there is an issue with Earth Engine credentials, or if the
    input HUC levels are invalid.
    """
    try:
        ee.Authenticate()
        ee.Initialize(project="ee-frostydawgs")
    except Exception as exc:
        raise ValueError("Problem with Earth Engine credentials") from exc

    huc_levs = ['02', '04', '06', '08', '10', '12']
    if final_huc_lev not in huc_levs:
        raise ValueError(f"Final Huc Levels must be one of {huc_levs}")
    if isinstance(huc_id, int):
        huc_id = str(huc_id)
    huc_lev_start = str(len(huc_id)).zfill(2)
    if huc_lev_start not in huc_levs:
        raise ValueError("Huc id must be an even number between 2 and 12")

    collection = ee.FeatureCollection(f'USGS/WBD/2017/HUC{final_huc_lev}')
    final_huc_lev = final_huc_lev.lstrip('0')
    filtered_collection = collection.filter(ee.Filter.stringStartsWith(f"huc{final_huc_lev}", huc_id))

    output = filtered_collection.map(lambda feature: feature.select([f"huc{final_huc_lev}", "name"]))
    geojson_dict = output.getInfo()

    for feature in geojson_dict["features"]:
        feature.pop("id")
        feature["properties"]["huc_id"] = feature["properties"].pop(f"huc{final_huc_lev}")
        feature["properties"]["huc_name"] = feature["properties"].get("name", "Unknown")
    
    geojson_dict["features"].sort(key=lambda x: x["properties"]["huc_id"])
    
    f_out = f"Huc{final_huc_lev}_in_{huc_id}.geojson"
    with open(f_out, "w") as file:
        json.dump(geojson_dict, file)
    
    gdf = gpd.read_file(f_out)
    
    if s3_save:
        s3_client = boto3.client('s3')
        s3_client.upload_file(f_out, bucket_nm, f_out)
        print(f"File {f_out} successfully uploaded to {bucket_nm}")
    os.remove(f_out)
    
    return gdf