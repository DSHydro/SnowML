"Module to create basin and watershed visualizations"

import boto3, io, json, rioxarray, s3fs, time
import geopandas as gpd
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import sys
import os

# Import get_geos fro a diff directory 
snowml_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, snowml_path)
from snowML import get_geos as gg


def basic_map(initial_huc, final_huc_lev):
    # Load the shapefile
    geos = gg.get_geos(initial_huc, final_huc_lev)
    map_object = geos.explore()
    output_dir = os.path.join("../docs/Visualizations", "basic_maps")
    file_name = f"Huc{final_huc_lev}_in_{initial_huc}.html"
    file_path = os.path.join(output_dir, file_name)
    map_object.save(file_path)
    print(f"Map saved to {file_path}")