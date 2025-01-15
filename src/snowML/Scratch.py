
import importlib
importlib.reload(your_module_name)

sudo ntpdate time.nist.gov

huc_lev =  "Huc10"
huc_no = 18040009 (Toulumne)
huc_no = 17020009 (Chelan)
huc_no = 17110005 (Skagit)

#For WRF TESTING 
ROOT = "http://www.northwestknowledge.net/metdata/data/"
USERNAME = os.environ["EARTHDATA_USER"]
PASSWORD = os.environ["EARTHDATA_PASS"]
FILE = "pr_1996.nc"



# Use Skagit Data for testing 
# def set_up_data(): 
#     GEOS = du.s3_to_gdf("shape-bronze", "examples/skagit_huc10.geojson")
#     HUC_IDs = ["1711000504", "1711000505", "1711000506", "1711000507", "1711000508", 
#          "1711000509", "1711000510", "1711000511"] 
#     VARS = ["SWE", "DEPTH"]
#     year = 1996
#     geo = GEOS.iloc[[0]]
#     HUC_IDs = GEOS.iloc[0]["huc10"]
#     return geo, VARS, year

bucket_name = "shape-bronze"
geojson_path = f"s3://{bucket_name}/huc02.geojson"
all_huc02.geojson

# Save the GeoDataFrame to S3 as GeoJSON
huc02_gdf.to_file(geojson_path, driver="GeoJSON", storage_options={"anon": False})

years_01 = range(1982, 1990)
years_02 = range(1990, 2000)
years_03 = range(2000, 2010)
years_04 = range(2010, 2020)
years_05 = range(2020, 2024)

from osgeo import gdal

# Input NetCDF file and variable
input_file = "input_1km.nc"
variable_name = "variable_name"  # Replace with your variable name
output_file = "output_4km.nc"

# Open the input dataset
input_dataset = gdal.Open(f'NETCDF:"{input_file}":{variable_name}')

# Set the output format and target resolution
output_format = "netCDF"
target_resolution_x = 4000  # 4km resolution in the x-direction
target_resolution_y = 4000  # 4km resolution in the y-direction

# Create the gdal_translate options
translate_options = gdal.TranslateOptions(
    format=output_format,
    xRes=target_resolution_x,
    yRes=target_resolution_y
)

# Perform the translation
gdal.Translate(output_file, input_dataset, options=translate_options)

print(f"Resampled NetCDF file saved to {output_file}")


import xarray as xr
import s3fs

# Dataset to save
ds = xr.Dataset(
    {
        "temperature": (("time", "lat", "lon"), [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]),
    },
    coords={
        "time": [1, 2],
        "lat": [10, 20],
        "lon": [30, 40],
    },
)

# S3 bucket and Zarr store path
s3_bucket = "my-s3-bucket"
zarr_store_path = "zarr-datasets/my-dataset.zarr"

# Create an S3 file system
s3 = s3fs.S3FileSystem(anon=False)

# Save the Zarr file to S3
store = s3fs.S3Map(root=f"{s3_bucket}/{zarr_store_path}", s3=s3, check=False)
ds.to_zarr(store, mode="w")

print(f"Zarr file saved to s3://{s3_bucket}/{zarr_store_path}")
