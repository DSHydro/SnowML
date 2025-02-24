# Module to retreive dem data and calc mean elevation
# pylint: disable=C0103

import matplotlib.pyplot as plt
import easysnowdata as easy
from snowML.datapipe import get_geos as gg

def get_dem(geos):
    ds = easy.topography.get_copernicus_dem(geos, resolution = 90)
    ds = ds.rio.write_crs("EPSG:4326", inplace=True)
    outer_boundary = geos.unary_union
    if outer_boundary.geom_type == "Polygon":
        clip_geom = [outer_boundary]
    elif outer_boundary.geom_type == "MultiPolygon":
        clip_geom = list(outer_boundary.geoms)  # Convert MultiPolygon to a list of Polygons
    else:
        raise ValueError("Unexpected geometry type for clipping.")
    dem_ds = ds.rio.clip(clip_geom, geos.crs, drop=True, invert=False)
    return dem_ds

def plot_dem(dem_ds, geos, huc_id):
    f, ax = plt.subplots(figsize=(10, 6))
    dem_ds.plot(ax=ax, cmap='terrain')
    # Plot geometries in black outline
    geos.plot(ax=ax, edgecolor='black', facecolor='none', linewidth=2, zorder=5)  # Higher zorder to ensure geos are above DEM
    ax.set_title(f"Digital Elevation Model (DEM) for huc {huc_id}")
    f_out = f"docs/basic_maps/dem_huc{huc_id}" # TO DO: Fix path to be dynamic
    plt.savefig(f_out, dpi=300, bbox_inches="tight")
    plt.close()

def calc_mean_dem(dem_ds):
    mean_elevation = dem_ds.mean().compute().item()
    #print(f"Mean Elevation: {mean_elevation:.2f} meters")
    return mean_elevation

# returns mean elevation for the given huc in meters
def process_dem_all(huc_id, huc_lev, plot = False):
    geos = gg.get_geos(huc_id, huc_lev)
    dem_ds = get_dem(geos)
    if plot:
        plot_dem(dem_ds, geos, huc_id)
    mean_elev = calc_mean_dem(dem_ds)
    return mean_elev
