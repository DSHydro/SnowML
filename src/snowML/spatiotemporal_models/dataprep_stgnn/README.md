# Data Preparation

This folder contains notebooks and scripts for preparing snow water equivalent (SWE) and related geospatial data for the SnowML project. The workflow processes data from multiple sources (DHSVM, ERA5, SNODAS, UCLA/NSIDC Western U.S. Snow Reanalysis) and integrates them into model-ready HUC12 watershed datasets.

## Files Overview

### **dataprep_stgnn.py**
Python script for preparing ST-GNN model-ready data for additional HUC12s not included in the main training set.

- **Purpose**: Creates gold and model-ready data for extra HUC12s
- **Workflow**:
  - Reads `dhsvm_lidar_hucs.csv` and identifies HUC12s marked with `not_in_training == 1`
  - Builds geometries for those HUC12s
  - Runs the data pipeline: bronze → gold → snowml-model-ready
- **Key Functions**:
  - `get_extra_test_hucs()`: Extracts HUC12s flagged for extra processing
  - `run_stgnn_dataprep()`: Main pipeline orchestration

### **dhsvm_lidar_hucs.csv**
Configuration file that specifies which HUC12s are included in the DHSVM/LIDAR dataset and whether they are excluded from the training set.

**Columns**:
- `basin`: Basin name (e.g., "lidar", "cedar", "green", "snohomish", "stilly")
- `huc_id`: 12-digit HUC identifier
- `not_in_training`: Binary flag (1 = not in training set, 0 = in training set)

**Use Case**: Identifies 91 HUC12s from DHSVM basins that were not part of the original training dataset but should be prepared for testing/validation.

### **lidar_swe_huc12_aggregated_olympic.csv**
Aggregated Airborne Snow Observatory (ASO) snow water equivalent data for HUC12 watersheds in the Olympic Mountains region.

**Columns**:
- `huc_id`: 12-digit HUC identifier
- `date`: Date of ASO acquisition (YYYY-MM-DD)
- `mean_lidar_swe`: Mean snow water equivalent in meters (aggregated from 50m ASO raster)

**Coverage**: 39 HUC12s within the Olympic Mountains extent; multiple acquisition dates (e.g., Feb–Mar 2016)

**Use Case**: Provides high-resolution reference SWE data for validating ST-GNN model predictions against actual field observations in the Olympic region. ASO data offers fine spatial detail at 50m resolution compared to coarser satellite or model products.

### **get_DHSVM_data.ipynb**
Jupyter notebook for processing DHSVM model output and S3 integration.

**Workflow**:
1. **Load DHSVM Output**: Fetches daily SWE data from 4 Washington State basins (Cedar, Green, Snohomish, Stilliguamish) from remote URLs
2. **Process Data**: Aggregates to daily mean SWE values
3. **Upload to S3**: Writes merged basin data to `s3://snowml-bronze/dhsvm/dhsvm_swe_all_basins.csv`
4. **Validation**: Checks HUC12 overlap with training set
   - Identifies which basin HUC12s are missing from the training dataset
   - Results used to populate `dhsvm_lidar_hucs.csv`

**Key Output**: 54,789 rows of daily SWE values spanning 1980–2020

### **get_era5.ipynb**
Jupyter notebook for retrieving Snow Cover Fraction (SCF), SWE, and terrain data.

**Workflow**:
1. **HUC12 Boundaries**: Downloads and merges GeoJSON files for all HUC12s; creates shapefiles for Google Earth Engine
2. **Missing Boundaries**: Fetches missing HUC12 boundaries from USGS NHD API
3. **ERA5 Data Export**: Uses Google Earth Engine to export yearwise (1983–2022) ERA5 snow cover and SWE data for all HUC12s
4. **Slope Data**: Calculates mean slope for each HUC12 using USGS 3DEP 10m DEM
5. **Data Integration**: Merges ERA5 and slope data into existing model-ready HUC12 CSVs and uploads to S3

**Outputs**:
- Yearly ERA5-derived snow cover and SWE features
- Mean slope statistic per HUC12
- Enriched model-ready CSVs in `s3://snowml-model-ready-stgnn/`

### **get_olympic_lidar.ipynb**
Jupyter notebook for Airborne Snow Observatory (ASO) LiDAR SWE data processing.

**Workflow**:
1. **Load ASO Rasters**: Fetches ASO 50m resolution SWE GeoTIFF files from NSIDC (e.g., ASO_50M_SWE_USWAOL_20160208.tif)
2. **Define Region**: Establishes bounding box for Olympic Mountains extent in Pacific Northwest
3. **HUC12 Intersection**: Queries and filters HUC12 boundaries that fall within the ASO raster extent
4. **Spatial Aggregation**:
   - Rasterizes HUC12 polygons to match ASO grid
   - Masks each HUC12 polygon to extract valid SWE pixels
   - Computes mean SWE per HUC12 per date
5. **Generate Output**: Creates CSV with HUC12-level aggregated SWE values

**Output**: `lidar_swe_huc12_aggregated_olympic.csv` — HUC12-aggregated ASO SWE data for model evaluation

### **get_snowdas.ipynb**
Jupyter notebook for SNODAS SWE data processing and integration.

**Workflow**:
1. **SNODAS Download**: Downloads daily SNODAS SWE grids from NSIDC (2003–2026)
2. **Spatial Aggregation**:
   - Rasterizes HUC12 boundaries aligned with SNODAS grid
   - Computes area- and latitude-weighted mean SWE for each HUC12
3. **S3 Storage**:
   - Writes monthly shards as parquet files
   - Combines monthly files into a single comprehensive parquet
4. **Data Integration**: Merges SNODAS SWE values into model-ready HUC12 CSVs

**Output**: Complete SNODAS SWE timeseries (2003–2026) aggregated to HUC12 level

### **get_ucla_gold_stgnn.py**
Python script that builds **gold-layer** UCLA / NSIDC Western U.S. Snow Reanalysis (WUS_UCLA_SR) SWE time series for ST-GNN and related pipelines.

**Workflow**:
1. **Authenticate**: Logs in to NASA Earthdata via `earthaccess` (environment variables `EARTHDATA_USERNAME` / `EARTHDATA_PASSWORD`, `~/.netrc`, or optional constants at the top of the script).
2. **Download tiles**: Fetches NetCDF tiles from NSIDC using URL patterns from `snowML.datapipe.utils.data_utils` (`swe_ucla`).
3. **Mosaic and clip**: For each water year, mosaics tiles covering a bounding box derived from the HUC geometry, clips to the basin, and takes the spatial mean per day.
4. **Stats**: Optionally keeps the full NSIDC `Stats` dimension: ensemble **mean**, **std**, **median**, **25th** and **75th** percentiles (`SWE_Post_*` columns); or mean only.
5. **Calendar dates**: Maps day index to calendar dates with water year starting October 1 (`assign_water_year_dates`).
6. **Upload**: Writes `ucla_swe_in_{huc}.csv` to S3 bucket `snowml-gold`.

**Key functions**:
- `get_gold_df(huc, year_start, year_end, ...)`: One HUC, water-year range; skips if the gold file already exists unless `overwrite=True`.
- `get_gold_multi(huc_list, ...)`: Batch over many HUCs.
- `get_gold_df_from_gdf(geos, geos_name, ...)`: Same pipeline for a custom GeoDataFrame and a chosen output name.

**Prerequisites**: Valid Earthdata account, NSIDC dataset access, and (if required) authorization for the NASA GES DISC application at [Earthdata](https://urs.earthdata.nasa.gov).

**Before running**: Configure NASA Earthdata credentials *first*—the script calls `earthaccess.login` at import time. Use one of: set `EARTHDATA_USERNAME` and `EARTHDATA_PASSWORD` in your environment; add a `~/.netrc` entry for `urs.earthdata.nasa.gov`; or fill in `MY_EARTHACCESS_USER` and `MY_EARTHACCESS_LOGIN` at the top of `get_ucla_gold_stgnn.py`. Do not commit real passwords into the repo.

### **get_ucla_quartile_data.ipynb**
Jupyter notebook that **batch-runs** the UCLA gold pipeline for every distinct `huc_id` in an ST-GNN evaluation export CSV.

**Workflow**:
1. Load a model-results CSV (e.g. `st_gnn_time_split_test_results_all_runs.csv`) with a `huc_id` column.
2. Call `get_gold_multi` from `get_ucla_gold_stgnn` with configurable water-year range, `overwrite`, and `keep_all_stats`.
3. With `keep_all_stats=True`, each output file includes **quartiles and spread** (`SWE_Post_p25`, `SWE_Post_p75`, etc.) alongside the mean—useful for uncertainty-aware analysis without a separate quartile-only step.

**Before running**: Same as the script—set Earthdata credentials (env, `~/.netrc`, or the constants in `get_ucla_gold_stgnn.py`) before executing cells that import `get_ucla_gold_stgnn`, or login will fail.

**Note**: Large HUC lists imply many NetCDF downloads per HUC per year; plan runtime and respect NSIDC load.

## Data Pipeline Summary

```
Inputs (HUC12 list)
    ↓
`dhsvm_lidar_hucs.csv` (rows where `not_in_training == 1`)
    ↓
`dataprep_stgnn.py`
  - `compile_geos(extra_hucs)` → fetch/build HUC polygons (via `snowML.datapipe.utils.get_geos`)
  - `process_multi_huc(...)` → end-to-end datapipe:
      bronze → gold (`mean_{var}_in_{huc_id}.csv` in `snowml-gold`)
      gold + silver static (`Static_No_Geo_Region_17.csv` in `snowml-silver`) → model-ready
    ↓
Model-ready HUC CSVs: `model_ready_huc{huc_id}.csv`
  - Bucket: `snowml-model-ready` (from `create_bucket_dict("prod")`)

`get_era5.ipynb`
  - Google Earth Engine exports (ERA5-Land SCF + SWE + mean slope) → uploads CSVs to `snowml-model-ready`
  - Then merges ERA5/slope into per-HUC model-ready CSVs and uploads results to `snowml-model-ready-stgnn`

Upstream bronze/enrichment notebooks (run separately as needed)
    ↓
`get_DHSVM_data.ipynb`
  - Remote DHSVM basin URLs → `s3://snowml-bronze/dhsvm/dhsvm_swe_all_basins.csv`
  - Also used to identify DHSVM-basin HUC12s missing from the training/model-ready set
    (results populate `dhsvm_lidar_hucs.csv`)

`get_snowdas.ipynb`
  - SNODAS (NSIDC G02158) daily grids → HUC12-aggregated parquet shards + combined parquet in
    `s3://snowml-bronze/snodas/`
  - Then merges SNODAS SWE into per-HUC model-ready CSVs (targets `snowml-model-ready-stgnn` in notebook config)

`get_ucla_gold_stgnn.py` / `get_ucla_quartile_data.ipynb`
  - NSIDC WUS_UCLA_SR NetCDF tiles (via `earthaccess`) → clip to HUC → daily basin-mean SWE (+ optional ensemble stats)
  - Gold CSVs: `ucla_swe_in_{huc_id}.csv` in `snowml-gold`
```


## Notes

- **Credentials before you run**: Most notebooks and scripts assume authentication is already set up (Earth Engine for ERA5, Earthdata for UCLA/SNODAS/NSIDC where applicable, and AWS or your usual SnowML method for S3). Configure those credentials in your environment or project config *before* starting a run; notebooks that upload to S3 also need valid cloud credentials.
- Google Earth Engine authentication is needed for ERA5 and related Earth Engine notebooks
- NASA Earthdata credentials are required for UCLA/NSIDC downloads (`get_ucla_gold_stgnn.py`); see **Before running** under that section
- The `dhsvm_lidar_hucs.csv` file serves as a configuration for identifying test/validation HUC12s
