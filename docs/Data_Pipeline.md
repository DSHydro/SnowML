This Document Describes the Data used in Training the Frosty Dawgs SnowML Model. <br>
Sections include  
-   [Raw Data](#raw-data)
-   [Data Pipeline Repo Steps](#data-pipeline---repo-steps)
-   [Model Ready Data](#model-ready-data)
-   [Regions Available for Analysis](#Regions-Available-for-Analysis)

  If you are most interested in understanding the final data used in the model, jump straight to [Model Ready Data](#model-ready-data)!

# Raw Data <br>
- Data related to Snow Water Equivilent ("SWE") was obtained from teh University of Arizona https://climate.arizona.edu/data/UA_SWE/
- Data for meteorological variables was obtaind from the University of Idaho https://www.climatologylab.org/gridmet.html.
- Data related to HUC geometries, i.e. the geojson files for each of the Huc units we modeled were obtained from [FILL IN]
- Snow type classification data was obtained from [FILL IN]  


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
| tundra            | Percent of the region with snow type Tundra.                    | %     |
| boreal_forest     | Percent of the region with snow type Boreal Forest.             | %     |
| maritime          | Percent of the region with snow type Maritime.                  | %     |
| ephemeral         | Percent of the region with snow type Ephemeral.                 | %     |
| prairie          | Percent of the region with snow type Prairie.                    | %     |
| montane_forest    | Percent of the region with snow type Montane Forest.            | %     |
| ice              | Percent of the region with snow type Ice.                        | %     |
| ocean            | Percent of the region with snow type Ocean.                      | %     |



# Regions Available for Analysis <br>
The following regions have been process and are currently available in the "dawgs-model-ready" bucket: <br>
  - All the HUC10 or Huc 12 subunits within the [Skagit Basin](https://github.com/DSHydro/SnowML/blob/main/docs/basin_fact_sheets/Skagitt.md) (17110005) 
  - All the HUC12 subunits within the [Chelan Basin](https://github.com/DSHydro/SnowML/blob/main/docs/basin_fact_sheets/Chelan.md) (17020009)
  - All the HUC12 subunits within the [Toulumne Basin] (https://github.com/DSHydro/SnowML/blob/main/docs/basin_fact_sheets/Toulumne) (18040009)
