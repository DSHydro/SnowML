# Data Preparation

This folder contains notebooks and scripts for preparing snow water equivalent (SWE) and related geospatial data for the SnowML project. The workflow processes data from multiple sources (DHSVM, ERA5, SNODAS) and integrates them into model-ready HUC12 watershed datasets.

## Files Overview

### **dataprep_stgnn.py**
Python script for preparing ST-GNN model-ready data for additional HUC12s not included in the main training set.

- **Purpose**: Creates gold and ST-GNN-specific model-ready data for extra HUC12s
- **Workflow**:
  - Reads `dhsvm_lidar_hucs.csv` and identifies HUC12s marked with `not_in_training == 1`
  - Builds geometries for those HUC12s
  - Runs the data pipeline: bronze → gold → model-ready (ST-GNN bucket configuration)
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

## Data Pipeline Summary

```
DHSVM Data (remote URLs)
    ↓
get_DHSVM_data.ipynb → dhsvm_swe_all_basins.csv (S3 bronze)
    ↓
Identify extra HUC12s → dhsvm_lidar_hucs.csv

ERA5 Data (Google Earth Engine)
    ↓
get_scf.ipynb → ERA5 snow cover, SWE, slope → model-ready CSVs (S3)

SNODAS Data (NSIDC)
    ↓
snowdas.ipynb → HUC12 aggregated SWE → model-ready CSVs (S3)

DHSVM/LIDAR HUC12s
    ↓
dataprep_stgnn.py → ST-GNN model-ready data (S3 stgnn bucket)
```


## Notes

- Google Earth Engine authentication is needed
- The `dhsvm_lidar_hucs.csv` file serves as a configuration for identifying test/validation HUC12s
