import zarr
import xarray as xr
from snowML.datapipe import geos as gg
from snowML.datapipe import set_data_constants as sdc

def prep_bronze(var, bucket_dict = None):
    if bucket_dict is None:
        bucket_dict = sdc.create_bucket_dict("prod")
    b_bronze = bucket_dict["bronze"]
    zarr_store_url = f's3://{b_bronze}/{var}_all.zarr'

    # Open the Zarr file directly with storage options
    ds = xr.open_zarr(zarr_store_url, consolidated=True, storage_options={'anon': False})

    # Process the dataset as needed
    if var != "swe":
        transform = du.calc_transform(ds)
        ds = ds.rio.write_transform(transform, inplace=True)
    else:
        ds.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)

    ds.rio.write_crs("EPSG:4326", inplace=True)
    ds.close()  # Close the dataset after processing

    return ds


def create_mask(ds, row, crs):
    """
    Create a mask for the given geometry.

    Parameters:
    ds (xarray.Dataset): The dataset to be masked.
    row (geopandas.GeoSeries): The row containing the geometry to be used for masking.
    crs (str or dict): The coordinate reference system of the geometry.

    Returns:
    xarray.Dataset: The masked dataset.
    """
    mask = ds.rio.clip([row.geometry], crs, drop=True, invert=False)
    return mask

def save_silver(huc_id, huc_lev, var_list):
    geos = gg.get_geos(huc_id, huc_lev)
    for var in var_list: 
        ds = prep_bronze(var)
        for i in range(geos.shape[0]):
            row = geos.iloc[i]
            huc_id = row["huc_id"]
            ds_small = create_mask(ds, row, crs)
            df_small = ds_small.to_dataframe()
            f_out = f"raw_{var}_in_{huc_id}" 
            du.dat_to_s3(df_small, bucket_name, f_out, file_type="csv", region_name="us-west-2")

