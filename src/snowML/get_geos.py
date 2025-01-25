"""Module that retrieves huc geometries for a given huc id and
savess to a local file, or optionally to an S3 bucket
defined by the user."""

import os
import boto3
import ee
import geemap
import json
import geopandas as gpd
import easysnowdata
from shapely.geometry import box


def get_geos(huc_id, final_huc_lev, s3_save = True, bucket_nm = "shape-bronze"):
    # make sure earth engine credentials are working
    try:
        ee.Authenticate()
        ee.Initialize(project = "ee-frostydawgs")
    except:
        raise ValueError("Problem with earth link credentials")

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
    asset_id = f'USGS/WBD/2017/HUC{final_huc_lev}'
    collection = ee.FeatureCollection(asset_id)
    print(f"collection retreived: {asset_id}")
    

    # Filter the collection to only include features within top level huc 
    filtered_collection = collection.filter(ee.Filter.stringStartsWith(f"huc{final_huc_lev}", huc_id))

    # Extract HUC IDs and geometries
    output = filtered_collection.map(lambda feature: feature.select([f"huc{final_huc_lev}"]))

    # Convert to a GeoJSON dictionary
    geojson_dict = output.getInfo()

    # Modify the GeoJSON dictionary
    for feature in geojson_dict["features"]:
        # Remove the 'id' field if it exists
        if "id" in feature:
            feature.pop("id")

    # Save the GeoJSON dictionary to a file
    f_out = f"Huc{final_huc_lev}_in_{huc_id}.geojson"
    with open(f_out, "w") as file:
        json.dump(geojson_dict, file)

    if s3_save:
        gdf.to_file(f_out, driver="GeoJSON")
        s3_client = boto3.client('s3')
        s3_client.upload_file(f_out, bucket_nm, f_out)
        os.remove(f_out)
        print(f"File {f_out} successfully uploaded to {bucket_nm}")
