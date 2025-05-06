"""
Module to retrieve DEM (Digital Elevation Model) data and calculate mean 
elevation.

Functions:
    get_dem(geos):
        Retrieves DEM data for given geometries and clips to the outer boundary.
        
    plot_dem(dem_ds, geos, huc_id):
        Plots the DEM data for the given geometries and saves as an image file.
        
    process_dem_all(huc_id, huc_lev, plot=False):
        Retrieves geometries for the given HUC (Hydrologic Unit Code), 
        processes the DEM data, optionally plots the DEM, and calculates 
        the mean elevation for that HUC. 
"""
# pylint: disable=C0103

import matplotlib.pyplot as plt
import easysnowdata as easy
from snowML.datapipe.utils import get_geos as gg

def get_dem(geos):
    """
    Retrieve and clip the Copernicus Digital Elevation Model (DEM) 
    for a given geographic area.

    Parameters:
    geos (geopandas.GeoDataFrame): A GeoDataFrame containing the geographic 
        area(s) of interest.

    Returns:
    xarray.Dataset: A dataset containing the clipped DEM data.

    Raises:
    ValueError: If geometry of the outer boundary is not Polygon or MultiPoly.
    """
    ds = easy.topography.get_copernicus_dem(geos, resolution = 90)
    ds = ds.rio.write_crs("EPSG:4326", inplace=True)
    outer_boundary = geos.unary_union
    if outer_boundary.geom_type == "Polygon":
        clip_geom = [outer_boundary]
    elif outer_boundary.geom_type == "MultiPolygon":
        # Convert MultiPolygon to a list of Polygons
        clip_geom = list(outer_boundary.geoms)
    else:
        raise ValueError("Unexpected geometry type for clipping.")
    dem_ds = ds.rio.clip(clip_geom, geos.crs, drop=True, invert=False)
    return dem_ds

def plot_dem(dem_ds, geos, huc_id):
    """
    Plots a Digital Elevation Model (DEM) with overlaid geometries.

    Parameters:
        dem_ds (xarray.DataArray): The DEM dataset to be plotted.
        geos (geopandas.GeoDataFrame): Geometries to overlay on the DEM plot.
        huc_id (str): Hydrologic Unit Code (HUC) identifier for the plot title 
            and output file name.

    Returns:
        None: The function saves the plot as an image file.
    """
    _, ax = plt.subplots(figsize=(10, 6))
    dem_ds.plot(ax=ax, cmap='terrain')
    # Plot geometries in black outline
    geos.plot(ax=ax, edgecolor='black', facecolor='none', linewidth=2, zorder=5)
    ax.set_title(f"Digital Elevation Model (DEM) for huc {huc_id}")
    f_out = f"docs/basic_maps/dem_huc{huc_id}" # TO DO: Fix path to be dynamic
    plt.savefig(f_out, dpi=300, bbox_inches="tight")
    plt.close()

def calc_mean_dem(dem_ds):
    """
    Calculate the mean elevation from a digital elevation model (DEM) dataset.

    Parameters:
    dem_ds (xarray.DataArray): The DEM dataset from which to calculate the mean.

    Returns:
    float: The mean elevation in meters.
    """
    mean_elevation = dem_ds.mean().compute().item()
    #print(f"Mean Elevation: {mean_elevation:.2f} meters")
    return mean_elevation

def process_dem_all(huc_id, huc_lev, plot = False):
    """
    Processes the Digital Elevation Model (DEM) for a given Hydrologic Unit 
    Code (HUC) and level.

    Parameters:
        huc_id (str): The Hydrologic Unit Code (HUC) identifier.
        huc_lev (int): The level of the HUC subunits for which you want to plot
            the mean.  
        plot (bool, optional): If True, plots the DEM. Default is False.

    Returns:
    float: The mean elevation of the DEM.
    """
    geos = gg.get_geos(huc_id, huc_lev)
    dem_ds = get_dem(geos)
    if plot:
        plot_dem(dem_ds, geos, huc_id)
    mean_elev = calc_mean_dem(dem_ds)
    return mean_elev
