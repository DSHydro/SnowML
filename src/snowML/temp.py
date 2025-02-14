import requests
import xarray as xr
import os
import io


aws s3 sync s3://dawgs-model-ready/ s3://snowml-modle-ready/ --profile testAccount --region us-east-1





def get_snow_class_data(geos=None):
    url = "https://daacdata.apps.nsidc.org/pub/DATASETS/nsidc0768_global_seasonal_snow_classification_v01/SnowClass_NA_05km_2.50arcmin_2021_v01.0.nc"
    
    # Load the Earthdata token from the environment variable
    earthdata_token = os.getenv("EARTHDATA_TOKEN")
    
    if not earthdata_token:
        raise ValueError("Earthdata token is not set in the environment variables.")

    # Set the headers with the Bearer token
    headers = {'Authorization': f'Bearer {earthdata_token}'}

    # Make the request with the token in the headers
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        # Check the content type and open the dataset
        ds = xr.open_dataset(io.BytesIO(response.content), engine="h5netcdf")
        if geos is None:  # return the data for CONUS
            lat_min, lat_max = 24.396308, 49.384358
            lon_min, lon_max = -125.0, -66.93457
            ds_conus = ds.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
            ds_conus = ds_conus.rio.write_crs("EPSG:4326")
            return ds_conus
        else:
            ds = ds.rio.write_crs("EPSG:4326")
            geos = geos.to_crs(ds.rio.crs)
            ds_final = ds.rio.clip(geos.geometry, geos.crs, drop=True)
            return ds_final
    else:
        raise ValueError(f"Failed to download a valid NetCDF file. Status Code: {response.status_code}")
