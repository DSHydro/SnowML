"""Module that retrieves huc geometries for a given huc id and
optionally saves the file into an S3 bucket defined by the user."""

import os
import boto3
import ee
import geemap
import geopandas as gpd
import easysnowdata
from shapely.geometry import box



def ee_to_geojson(asset_id, field_name, value, output_file):
    """
    Filters an Earth Engine FeatureCollection by a specific field and value,
    and exports it to a GeoJSON file locally.

    Args:
        asset_id (str): The Earth Engine asset ID.
        field_name (str): The field name to filter on.
        value (str): The value to filter by.
        output_file (str): The name of the output GeoJSON file.

    Returns:
        None
    """
    dataset = ee.FeatureCollection(asset_id)
    filtered = dataset.filter(ee.Filter.eq(field_name, value))
    geemap.ee_export_vector(filtered, filename=output_file)

# function that creates an outer boundary box for a given geometry
def create_bbox (sp):
    minx, miny, maxx, maxy = sp.bounds
    bbox = box(minx, miny, maxx, maxy)
    return bbox

def get_geos(huc_id, final_huc_lev, save = True, bucket_nm = "shape-bronze"):
    # make sure earth engine credentials are working
    try:
        ee.Authenticate()
        ee.Initialize(project='ee-frostydawgs')
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

    # Get the geometry for the top level huc
    asset_id = f'USGS/WBD/2017/HUC{huc_lev_start}'
    f_out = "temp_ee.geojson"
    ee_to_geojson(asset_id, f"huc{len(huc_id)}", huc_id, f_out)
    filtered_gdf = gpd.read_file(f_out)
    os.remove(f_out)
    print(f"Temporary file {f_out} removed")
    outer_geo = filtered_gdf.iloc[0]["geometry"]

    # create a df of all the subunit within the bounding box
    bbox = create_bbox(outer_geo)
    gdf = easysnowdata.hydroclimatology.get_huc_geometries(bbox_input=bbox, huc_level=final_huc_lev)

     # save results
    if save:
        f_out = f"Huc{final_huc_lev}_in_{huc_id}.geojson"
        gdf.to_file(f_out, driver="GeoJSON")
        s3_client = boto3.client('s3')
        s3_client.upload_file(f_out, bucket_nm, f_out)
        os.remove(f_out)
        print(f"File {f_out} successfully uploaded to {bucket_nm}")

