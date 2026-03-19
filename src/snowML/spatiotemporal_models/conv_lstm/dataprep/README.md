# ConvLSTM dataprep

This folder contains the notebooks used to create **model-ready ConvLSTM inputs**:

- `get_huc_shapes.ipynb`: fetch watershed boundaries (HUC10 and HUC12) and upload them to S3.
- `dataprep.ipynb`: use those shapes to clip/sample gridded hydro-meteorological data onto a fixed grid and write one **zarr per HUC10** to S3.

## Expected run order

1. Run `get_huc_shapes.ipynb`
2. Run `dataprep.ipynb`

`dataprep.ipynb` expects the HUC10 shape outputs + metadata CSV written by `get_huc_shapes.ipynb`.

## `get_huc_shapes.ipynb` (shapes + metadata)

**What it does**

- Chooses a set of HUC8 basins, fetches all contained **HUC10** polygons (with names), and uploads **one GeoJSON per HUC10**.
- For each HUC10, also fetches all **HUC12 polygons inside it** and uploads a companion GeoJSON.
- Builds metadata CSVs for downstream processing.

**Outputs on S3** (bucket: `convlstm-model-ready`)

- `huc10_shapes/Huc10_in_<HUC10_ID>.geojson`
- `huc12_shapes_in_huc10/Huc12_in_<HUC10_ID>.geojson`
- `huc10_shapes/metadata/huc10_list.csv` (HUC8_ID, HUC10_ID, HUC10_Name)
- `huc12_shapes_in_huc10/metadata/huc12_in_huc10_list.csv` (HUC10_ID, HUC12_ID, HUC12_Name)

**Requirements**

- Google Earth Engine authentication (the notebook calls `get_geos.ee_creds()`).
- AWS credentials configured for S3 upload.

## `dataprep.ipynb` (HUC10 → model-ready zarr)

**What it does**

For each HUC10:

- Loads the HUC10 boundary GeoJSON from S3 and reprojects to EPSG:5070.
- Rasterizes the polygon to a grid (default resolution: 4 km), builds a binary mask, and computes pixel-center coordinates (EPSG:5070 + lat/lon).
- Samples source zarr datasets (SWE, precipitation, min/max temperature, snow class) onto that grid.
- Pads to a fixed spatial size (default: `24×24`), applies the watershed mask (outside = NaN), rasterizes HUC12 IDs onto the same grid, and writes a zarr to S3.

**Inputs**

- Shapes + metadata produced by `get_huc_shapes.ipynb` in `s3://convlstm-model-ready/…`
- Source (“bronze”) zarrs, configured in the notebook (defaults point at `s3://snowml-bronze/...`)

**Outputs on S3**

- One zarr per HUC10, under the configured prefix (see notebook config):
  - `s3://convlstm-model-ready/model_reday_data/huc10_<HUC10_ID>_<HMAX>x<WMAX>.zarr`

## Common config knobs (in `dataprep.ipynb`)

- **Which HUC10s to process**: set `HUC10_IDS_OVERRIDE = ["1711000507", ...]` for a small test; otherwise it reads `huc10_list.csv`.
- **Grid/padding**: `PIXEL_KM`, `HMAX`, `WMAX`.
- **S3 locations**: `BUCKET`, `S3_PREFIX_SHAPES`, `S3_PREFIX_HUC12_IN_HUC10`, `S3_PREFIX_DATA`.

