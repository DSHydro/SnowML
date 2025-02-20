This Document Describes the Data used in Training the Frosty Dawgs SnowML Model. <br>
Sections include  
-   [Raw Data](#raw-data)
-   [Data Pipeline Repo Steps](#data-pipeline---repo-steps)
-   [Model Ready Data](#model-ready-data)
-   [Regions Available for Analysis](#Regions-Available-for-Analysis)

  If you are most interested in understanding the final data used in the model, jump straight to [Model Ready Data](#model-ready-data)!

# Raw Data <br>
- Data related to Snow Water Equivilent ("SWE") was obtained from the University of Arizona https://climate.arizona.edu/data/UA_SWE/
- Data for meteorological variables was obtaind from the University of Idaho https://www.climatologylab.org/gridmet.html.
- Data related to HUC geometries, i.e. the geojson files for each of the Huc units we modeled were obtained from the [USGS Water Data Boundary Data Set](https://www.usgs.gov/national-hydrography/watershed-boundary-dataset) 2017, accessed via Google Earth Engine.  See [here](https://developers.google.com/earth-engine/datasets/catalog/USGS_WBD_2017_HUC12)
- Snow type classification data was obtained from Liston, G. E., and M. Sturm, 2021: Global Seasonal-Snow Classification, Version 1. National Snow and Ice Data Center, https://doi.org/10.5067/99FTCYYYLAQ0.
- Elevation data was obtained from the Copernicus DEM, a Digital Surface Model (DSM) derived from the WorldDEM, with additional
    editing applied to water bodies, coastlines, and other special features. European Space Agency (2024).  <i>Copernicus Global Digital Elevation Model</i>.  Distributed by OpenTopography.  The data was accessed using the easysnowdata open source python module.  


# Data Pipeline - Repo Steps <br>

# Model Ready Data <br>
Data ready to be deployed into the Frosty Dawgs SnowML model can be downloaded from the S3 bucket named "dawgs-model-ready". <br>
Each file contains data for one discrete HUC unit (watershed/subwatershed).

**Naming convention**
- Each file is named "model_ready_{huc_id}.csv"
- For example, "model_ready_huc170200090302.csv" contains data for huc_id 170200090302, a HUC 12 (sub-watershed) huc unit that is part of the Chelan basin.

**Time Period** <br>
Each file contains data for the water years 1984 through 2022. <br>
Each variable is measured daily for the period "1983-10-01" through "2022-09-30"

**Variables** <br>

| Variable Short Name | Description                                                    | Units |
|---------------------|----------------------------------------------------------------|-------|
| mean_pr            | The mean precipitation on a daily scale, aggregated over the entire HUC unit. | mm    |
| mean_tair          | The mean air temperature on a daily scale, aggregated over the entire HUC unit. | °C    |
| mean_vs            | The mean wind speed on a daily scale, aggregated over the entire HUC unit. | m/s   |
| mean_srad          | The mean solar radiation on a daily scale, aggregated over the entire HUC unit. | W/m²  |
| mean_rmax          | The mean maximum relative humidity on a daily scale, aggregated over the entire HUC unit. | %     |
| mean_rmin          | The mean minimum relative humidity on a daily scale, aggregated over the entire HUC unit. | %     |
| mean_swe           | The mean snow water equivalent on a daily scale, aggregated over the entire HUC unit. | meters |
| Tundra            | Percent of the region with snow type Tundra.                    | %     |
| Boreal_Forest     | Percent of the region with snow type Boreal Forest.             | %     |
| Maritime          | Percent of the region with snow type Maritime.                  | %     |
| Ephemeral         | Percent of the region with snow type Ephemeral.                 | %     |
| Prairie          | Percent of the region with snow type Prairie.                    | %     |
| Montane Forest    | Percent of the region with snow type Montane Forest.            | %     |
| Ice              | Percent of the region with snow type Ice.                        | %     |
| Ocean            | Percent of the region with snow type Ocean.                      | %     |



# Regions Available for Analysis <br>
The following regions have been process and are currently available in the "dawgs-model-ready" bucket: <br>
  - All the HUC10 or Huc 12 subunits within the [Skagit Basin](https://github.com/DSHydro/SnowML/blob/main/docs/basin_fact_sheets/Skagit.md) (17110005) 
  - All the HUC12 subunits within the [Chelan Basin](https://github.com/DSHydro/SnowML/blob/main/docs/basin_fact_sheets/Chelan.md) (17020009)
  - All the HUC12 subunits within the [Upper Yakima Basin](https://github.com/DSHydro/SnowML/blob/main/docs/basin_fact_sheets/UpperYakima.md) (17030002)
  - All the HUC12 subunits within the [Tuolumne Basin](https://github.com/DSHydro/SnowML/blob/main/docs/basin_fact_sheets/Tuolumne.md) (18040009)

