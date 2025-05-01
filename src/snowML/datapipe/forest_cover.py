""" Module to Calculate Forest Cover For Hucs"""

# pylint: disable=C0103

import rasterio
import xarray as xr
import pandas as pd
import numpy as np
import rioxarray
from pyproj import Transformer
import matplotlib.pyplot as plt
from snowML.datapipe import get_geos as gg
from snowML.datapipe import data_utils as du


def load():
    tif_path = "notebooks/Land_Cover/nlcd_tcc_conus_2021_v2021-4.tif"
    ds = rioxarray.open_rasterio(tif_path)
    land_cover_ds = ds.to_dataset(name="tree_canopy_cover")
    return land_cover_ds

def get_geos_series(huc_id):
    huc_lev = str(len(str(huc_id))).zfill(2)
    geos = gg.get_geos(huc_id, huc_lev)
    geo = geos.geometry
    return geos, geo

def clip_and_reproject(ds_land_cover, geo_series):
    # Get bounds in EPSG:4326
    minx, miny, maxx, maxy = geo_series.total_bounds
    # Transform bounds to EPSG:5070
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:5070", always_xy=True)
    minx_5070, miny_5070 = transformer.transform(minx, miny)
    maxx_5070, maxy_5070 = transformer.transform(maxx, maxy)

    # Clip the dataset in native CRS
    ds_subset = ds_land_cover.sel(
        x=slice(minx_5070, maxx_5070),
        y=slice(maxy_5070, miny_5070)  # reverse y order
    )

    # Reproject to EPSG:4326
    ds_subset = ds_subset.rio.write_crs("EPSG:5070")
    ds_reprojected = ds_subset.rio.reproject("EPSG:4326")

    return ds_reprojected

def subset_by_row(ds, row):
    ds.rio.write_crs(row.crs, inplace=True)
    clipped_data = ds.rio.clip(row.geometry, drop=True)
    return clipped_data

def remove_invals(ds, var_name="tree_canopy_cover", inval_value=255, quiet = True):
    """
    Replace invalid values in a variable with NaN and return the modified dataset.
    Prints the count and percentage of values replaced.

    Parameters:
        ds (xarray.Dataset): The dataset containing the variable.
        var_name (str): The name of the variable to clean.
        inval_value (int or float): The value considered invalid.

    Returns:
        xarray.Dataset: A copy of the dataset with invalid values replaced by NaN.
    """
    da = ds[var_name]
    total = da.size
    invalid_count = (da == inval_value).sum().item()
    invalid_percent = (invalid_count / total) * 100

    if not quiet:
        print(f"Invalid value count ({inval_value}): {invalid_count}")
        print(f"Invalid value percentage: {invalid_percent:.2f}%")

    ds_clean = ds.copy()
    ds_clean[var_name] = da.where(da != inval_value, np.nan)
    ds_clean = ds_clean.sortby(['x', 'y'])

    return ds_clean


def calc_mean (ds):
    # Take mean over 'y' and 'x', skipping NaNs
    mean_value = ds['tree_canopy_cover'].mean(dim=['x', 'y'], skipna=True).item()
    return mean_value

def plot_tree_canopy(ds_clean, var_name="tree_canopy_cover"):
    """
    Create a colorplot for the given variable in ds_clean with NaN values shown as black.
    It retains the x and y dimensions while removing the band dimension and sets axis labels to the values.
    
    Parameters:
        ds_clean (xarray.Dataset): The dataset with the variable.
        var_name (str): The name of the variable to plot.
    """
    # Extract the variable values and remove the 'band' dimension (keeping x and y)
    data = ds_clean[var_name].values.squeeze()  # Remove singleton dimensions

    # Mask NaN values
    masked_data = np.ma.masked_invalid(data)

    # Get x and y coordinates (assuming x and y are coordinate variables in the dataset)
    x_coords = ds_clean.coords['x'].values
    y_coords = ds_clean.coords['y'].values

    # Create the plot
    plt.figure(figsize=(10, 8))

    # Set NaN values to black in the colormap
    cmap = plt.cm.viridis
    cmap.set_bad(color='black')

    # Plot the data
    im = plt.imshow(masked_data, cmap=cmap)

    # Set the ticks and labels for x and y axes
    plt.xticks(np.linspace(0, len(x_coords)-1, 5), np.round(np.linspace(x_coords.min(), x_coords.max(), 5), 2))
    plt.yticks(np.linspace(0, len(y_coords)-1, 5), np.round(np.linspace(y_coords.min(), y_coords.max(), 5), 2))

    # Add a colorbar and title
    plt.colorbar(label="Tree Canopy Cover")
    plt.title(f"{var_name} (NaN = Black)")

    # Show the plot
    plt.show()

def load_current_cover_data(save_ttl):
    b = "snowml-gold"  # TO DO - make dynamic
    f = save_ttl + ".csv"
    df = du.s3_to_df(f, b)
    return df 
    

def forest_cover_all (huc_list, save_ttl = "Forest_Cover_Percent"):
    land_cover_ds = load()
    mean_cover_list = []
    new_huc_list = []
    existing_df = load_current_cover_data(save_ttl)
    processed_hucs = list(existing_df["huc_id"])
    
    tot = len(huc_list)
    count = 0
    for huc_id in huc_list:
        count += 1
        print(f"processing huc {count} of {tot}") 
        if int(huc_id) in processed_hucs: 
            print("already exists")
        else: 
            geos, geo = get_geos_series(huc_id)
            ds_small = clip_and_reproject(land_cover_ds, geo).squeeze()
            mask = subset_by_row(ds_small, geos).squeeze()
            ds_clean = remove_invals(mask)
            mean_cover = calc_mean(ds_clean)
            mean_cover_list.append(mean_cover)
            new_huc_list.append(int(huc_id))
    new_df = pd.DataFrame({"huc_id": new_huc_list, "Mean Forest Cover": mean_cover_list})
    results = pd.concat([existing_df, new_df], ignore_index=True)
    results.set_index("huc_id", inplace=True)
    results.sort_index(inplace=True)
    if save_ttl is not None: 
        b = "snowml-gold"  # TO DO - make dynamic
        du.dat_to_s3(results, b, save_ttl, file_type="csv")
    return results

