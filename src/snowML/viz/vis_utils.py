# pylint: disable=C0103
"Module to create basin and watershed visualizations"

import os
import fsspec
import pandas as pd
import xarray as xr
import rioxarray
import zarr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
from snowML.datapipe import snow_types as st
from snowML.datapipe import get_geos as gg
from snowML.datapipe import data_utils as du
from snowML.datapipe import set_data_constants as sdc
from snowML.datapipe import get_dem as gd

def plot_var(df, var, huc, initial_huc):
    plt.figure(figsize=(12,  6))
    plt.plot(df.index, df[var], c='b', label= f"Actual {var}")  
    
    # Set y-axis limits for "mean_swe"
    if var == "mean_swe":
        plt.ylim(0, 2)
    
    plt.legend()
    plt.xlabel('Date')
    plt.ylabel(var)
    ttl = f'Actual {var} for huc{huc}'
    plt.title(ttl)
    # save file

    # Define the output directory and ensure it exists
    output_dir = os.path.join("docs", "var_plots_actuals", str(initial_huc))
    os.makedirs(output_dir, exist_ok=True)
    file_name = f"{ttl}.png"
    file_path = os.path.join(output_dir, file_name)
    plt.savefig(file_path)
    plt.close()  # Close the figure to free memory
    #print(f"Map saved to {file_path}")


def get_model_ready (huc, bucket_dict = None):
    if bucket_dict is None:
        bucket_dict = sdc.create_bucket_dict("prod")
    bucket_name = bucket_dict.get("model-ready")
    file_name = f"model_ready_huc{huc}.csv"
    df = du.s3_to_df(file_name, bucket_name)
    df['day'] = pd.to_datetime(df['day'])
    df.set_index('day', inplace=True)  # Set 'day' as the index
    return df 

def plot_actual(huc, var, initial_huc, bucket_dict = None):
    df = get_model_ready(huc, bucket_dict= bucket_dict)
    plot_var(df, var, huc, initial_huc)

def summarize_swe(df):
    """
    Summarizes the mean_swe variable for each water year.
    A water year starts on October 1 and ends on September 30.
    
    Parameters:
        df (pd.DataFrame): DataFrame with index 'day' and column 'mean_swe'.
        
    Returns:
        pd.DataFrame: DataFrame with water year as index and columns 'annual_max_swe' and 'annual_mean_swe'.
    """
    # Ensure the index is a datetime index
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    
    # Resample by water year (Oct 1 - Sep 30)
    summary = df.resample('YE-SEP').agg(
        annual_peak_swe=('mean_swe', 'max'),
        annual_mean_swe=('mean_swe', 'mean')
    )
    
    # Adjust index to represent the water year
    summary.index = summary.index.year
    
    # Calculate and print median values
    median_peak_swe = summary['annual_peak_swe'].median()
    median_ann_mean_swe = summary['annual_mean_swe'].median()
    #print(f"Median of annual max SWE: {median_peak_swe}")
    #print(f"Median of annual mean SWE: {median_ann_mean_swe}")

    return median_peak_swe, median_ann_mean_swe, summary

def basin_swe_summary(huc_id, final_huc_lev): 
    geos = gg.get_geos(huc_id, final_huc_lev)
    hucs = geos["huc_id"]
    medians = []
    for huc in hucs: 
        df = get_model_ready(huc)
        median_peak_swe, _, _, = summarize_swe(df)
        medians.append(median_peak_swe)
    results = pd.DataFrame({'huc_id': hucs, 'Median Peak Swe': medians})
    results.set_index('huc_id', inplace=True)
    f_out = f"docs/tables/Peak_annual_swe_huc{huc}.csv"
    results.to_csv(f_out)
    return results

    
def basic_map(geos, final_huc_lev, initial_huc):
    map_object = geos.explore()
    output_dir = os.path.join("docs", "basic_maps")
    file_name = f"Huc{final_huc_lev}_in_{initial_huc}.html"
    file_path = os.path.join(output_dir, file_name)
    map_object.save(file_path)
    print(f"Map saved to {file_path}")

def snow_colors():
    snow_class_colors_small = {
    3: "#FFFF00",  # Maritime (yellow)
    4: "#FFFFFF",  # Ephemeral (white)
    5: "#E31A1C",  # Prairie (red)
    6: "#FDBF6F",  # Montane Forest (orange)
    7: "#000000",  # Ice (black)
    }
    return snow_class_colors_small

def calc_bounds(geos):
    merged_geom = geos.geometry.union_all()
    outer_bound = merged_geom.convex_hull
    return outer_bound


def map_snow_types(ds, geos, huc, class_colors = None, output_dir = None):

    # Set up the Cartopy projection
    fig, ax = plt.subplots(
    figsize=(10, 6),
    subplot_kw={"projection": ccrs.PlateCarree()}
    )

    # Add a baselayer
    
    if class_colors is None:
        class_colors = snow_colors()
    # Create a colormap and normalization based on the dictionary
    cmap = mcolors.ListedColormap([class_colors[i] for i in sorted(class_colors.keys())])
    bounds = list(class_colors.keys()) + [max(class_colors.keys()) + 1]  # Class boundaries
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    #Set extent
    outer_bound = calc_bounds(geos)
    minx, miny, maxx, maxy = outer_bound.bounds
    ax.set_extent([minx, maxx, miny, maxy], crs=ccrs.PlateCarree())

    # Plot the `snow_class` variable
    im = ax.pcolormesh(
        ds.lon,
        ds.lat,
        ds.SnowClass,
        cmap=cmap,
        norm=norm,
        transform=ccrs.PlateCarree()
    )

    # Plot the geometry outlines from the GeoDataFrame
    geos.plot(
        ax=ax,
        edgecolor="black",  # Color for the outlines
        facecolor="none",   # No fill
        linewidth=1.0,      # Thickness of the outlines
        transform=ccrs.PlateCarree()  # Ensure proper projection
    )

    # Set title and gridlines
    ax.set_title(f"Snow Classes In Huc {huc}", fontsize=14)
    ax.gridlines(draw_labels=True, linewidth=0.5, color="gray", alpha=0.5, linestyle="--")

    # Add a legend with a border around each color
    legend_patches = [
        mpatches.Patch(facecolor=color, edgecolor="black", label=label, linewidth=1)
        for label, color in zip(
            ["Maritime", "Ephemeral", "Prairie", "Montane Forest", "Ice"],
            [class_colors[i] for i in sorted(class_colors.keys())]
        )
    ]

    # Position the legend to the right, aligned with the top
    ax.legend(
        handles=legend_patches,
        title="Snow Classification",
        loc="upper left",
        bbox_to_anchor=(1.15, 1),
        borderaxespad=0
    )

    plt.tight_layout()

    # Save the plot
    
    file_name = f"Snow_classes_in_{huc}.png"
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
    else: 
        output_dir = os.path.join("docs", "basic_maps")
    file_path = os.path.join(output_dir, file_name)
    plt.savefig(file_path)
    plt.close(fig)  # Close the figure to free memory
    print(f"Map saved to {file_path}")

def plot_dem(dem_ds, geos, huc_id):
    f, ax = plt.subplots(figsize=(10, 6))
    dem_ds.plot(ax=ax, cmap='terrain')
    # Plot geometries in black outline
    geos.plot(ax=ax, edgecolor='black', facecolor='none', linewidth=2, zorder=5)  # Higher zorder to ensure geos are above DEM
    ax.set_title(f"Digital Elevation Model (DEM) for huc {huc_id}")
    f_out = f"docs/basic_maps/dem_huc{huc_id}" # TO DO: Fix path to be dynamic
    plt.savefig(f_out, dpi=300, bbox_inches="tight")
    print(f"Map saved to {f_out}")
    plt.close()

def create_vis_all(initial_huc, final_huc_lev):
    geos = gg.get_geos(initial_huc, final_huc_lev)
    basic_map(geos, final_huc_lev, initial_huc) # create and save basic map
    ds_snow = st.snow_class_data_from_s3(geos) 
    map_snow_types(ds_snow, geos, initial_huc) # create and save snow class map
    dem_ds = gd.get_dem(geos)
    plot_dem(dem_ds, geos, initial_huc) # create and save map of elevation
    # for huc in geos["huc_id"].tolist(): # create and save map of actuals
        # plot_actual(huc, "mean_swe", initial_huc, bucket_dict = None)
    # swe_summary = basin_swe_summary(initial_huc, final_huc_lev) # create and save csv of median peak
    
