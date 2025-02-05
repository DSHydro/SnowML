"Module to create basin and watershed visualizations"

import boto3, io, json, rioxarray, s3fs, time
import geopandas as gpd
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import sys
import os
import data_utils as du
import snow_types as st
import get_geos as gg



def basic_map(geos, final_huc_lev, initial_huc):
    map_object = geos.explore()
    output_dir = os.path.join("../../docs/Visualizations", "basic_maps")
    file_name = f"Huc{final_huc_lev}_in_{initial_huc}.html"
    file_path = os.path.join(output_dir, file_name)
    map_object.save(file_path)
    print(f"Map saved to {file_path}")

def create_snow_type_legend():
    snow_class_colors = {
        1: "#FFC0CB",  # Tundra (pink)
        2: "#008000",  # Boreal Forest (green)
        3: "#FFFF00",  # Maritime (yellow)
        4: "#FFFFFF",  # Ephemeral (white)
        5: "#E31A1C",  # Prairie (red)
        6: "#FDBF6F",  # Montane Forest (orange)
        7: "#000000",  # Ice (black)
        8: "#0000FF",  # Ocean (blue)
    }
    return snow_class_colors

def map_snow_types(ds, huc):
    # Create a colormap and normalization based on the dictionary
    snow_class_colors = create_snow_type_legend()
    cmap = mcolors.ListedColormap([snow_class_colors[i] for i in sorted(snow_class_colors.keys())])
    bounds = list(snow_class_colors.keys()) + [max(snow_class_colors.keys()) + 1]  # Class boundaries
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    # Plot the `snow_class` variable from ds_conus
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.pcolormesh(
        ds.lon, ds.lat, ds.SnowClass, cmap=cmap, norm=norm
    )
    cb = plt.colorbar(im, ax=ax, orientation="vertical", ticks=sorted(snow_class_colors.keys()))
    cb.set_label("Snow Classes")

    # Set axis labels and title
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"Snow Classes in {huc}")

    # Save the plot
    output_dir = os.path.join("../../docs/Visualizations", "basic_maps")
    file_name = f"Snow_classes_in_{huc}.png"
    file_path = os.path.join(output_dir, file_name)
    plt.savefig(file_path)  
    plt.close(fig)  # Close the figure to free memory
    print(f"Map saved to {file_path}")


def create_vis_all(initial_huc, final_huc_lev): 
    geos_detail = gg.get_geos(initial_huc, final_huc_lev)
    #basic_map(geos_detail, final_huc_lev, initial_huc)
    initial_huc_lev = str(len(str(initial_huc))).zfill(2)
    geos = gg.get_geos(initial_huc, initial_huc_lev)
    ds = st.get_snow_class_data(geos)
    map_snow_types(ds, initial_huc)
    return ds