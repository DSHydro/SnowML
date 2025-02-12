# pylint: disable=C0103
"Module to create basin and watershed visualizations"

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
import snow_types as st
import get_geos as gg
import data_utils as du
import set_data_constants as sdc


def plot_actual(huc, var, bucket_dict = None):
    if bucket_dict is None:
        bucket_dict = sdc.create_bucket_dict("prod")
    bucket_name = bucket_dict.get("model-ready")
    print(bucket_name)
    file_name = f"model_ready_huc{huc}.csv"
    df = du.s3_to_df(file_name, bucket_name)
    df['day'] = pd.to_datetime(df['day'])
    df.set_index('day', inplace=True)  # Set 'day' as the index
    plt.figure(figsize=(12,  6))
    plt.plot(df.index, df[var], c='b', label= f"Actual {var}")  
    plt.legend()
    plt.xlabel('Date')
    plt.ylabel(var)
    ttl = f'Actual {var} for huc{huc}'
    plt.title(ttl)
    # save file
    output_dir = os.path.join("../../docs", "var_plots_actuals")
    file_name = f"{ttl}.png"
    file_path = os.path.join(output_dir, file_name)
    plt.savefig(file_path)
    plt.close(fig)  # Close the figure to free memory
    print(f"Map saved to {file_path}")
    


def basic_map(geos, final_huc_lev, initial_huc):
    map_object = geos.explore()
    output_dir = os.path.join("../../docs", "basic_maps")
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


def map_snow_types(ds, geos, huc):

    # Set up the Cartopy projection
    fig, ax = plt.subplots(
    figsize=(10, 6),
    subplot_kw={"projection": ccrs.PlateCarree()}
    )

    # Add a baselayer
    

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
    output_dir = os.path.join("../../docs", "basic_maps")
    file_name = f"Snow_classes_in_{huc}.png"
    file_path = os.path.join(output_dir, file_name)
    plt.savefig(file_path)
    plt.close(fig)  # Close the figure to free memory
    print(f"Map saved to {file_path}")

def create_vis_all(initial_huc, final_huc_lev):
    geos = gg.get_geos(initial_huc, final_huc_lev)
    basic_map(geos, final_huc_lev, initial_huc)
    ds = st.get_snow_class_data(geos)
    map_snow_types(ds, geos, initial_huc)
    return ds
