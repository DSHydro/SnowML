This document describes the data used in training the Frosty Dawgs SnowML Model. Sections include:
-   [Raw Data](https://github.com/DSHydro/SnowML/blob/main/docs/Data_Pipeline.md#raw-data-)
-   [Data Pipeline - A Modular, Scalabel Approach](https://github.com/DSHydro/SnowML/blob/main/docs/Data_Pipeline.md#data-pipleline---a-modular-scalable-approach)
-   [Data Pipeline Repo Steps](https://github.com/DSHydro/SnowML/blob/main/docs/Data_Pipeline.md#data-pipeline---repo-steps-)
-   [Model Ready Data](https://github.com/DSHydro/SnowML/blob/main/docs/Data_Pipeline.md#model-ready-data-)
-   [Regions Available for Analysis](https://github.com/DSHydro/SnowML/blob/main/docs/Data_Pipeline.md#regions-available-for-analysis-)

  If you are most interested in understanding the final data used in the model, jump straight to [Model Ready Data](https://github.com/DSHydro/SnowML/blob/main/docs/Data_Pipeline.md#model-ready-data-)!

# Raw Data <br>
**SWE (Target Data)**
- Data related to Snow Water Equivilent ("SWE") was obtained from the University of Arizona https://climate.arizona.edu/data/UA_SWE/
- Data is available on a daily time scale, for water years 1983 through 2023 (10/1/1982 - 9/30/2024) and partial data for WY 2024. However, water years 2023 and 2024 (partial) were not yet available at the time of our data acquisition and expirimentation, and are not yet included in our datapipeline.
- This data set combines data from in-situ measurements at thousands of ground sites (including both SNOWTEL and community based monitoring sites) and extrapoloates using a physcis based model and PRISM precipitation air temperature data.
- Data is available for the continental United States (CONUS). <br>

**Meterological Data (GRIDMET)**
- Data for meteorological variables was obtaind from the University of Idaho https://www.climatologylab.org/gridmet.html.
- The GRIDMET data is available for a variety of variables and time scales.  We used 4km gridded data, and data related to the seven variables discussed below in the Bronze Data section. <br>

**HUC Geometries**
- In order to select data for a given watershed(Huc10) or sub-watershed(Huc12), we used the HUC geometries available from the [USGS Water Data Boundary Data Set](https://www.usgs.gov/national-hydrography/watershed-boundary-dataset) 2017, accessed via Google Earth Engine.  See [here](https://developers.google.com/earth-engine/datasets/catalog/USGS_WBD_2017_HUC12) <br>

**Snow Classification**
- Snow type classification data was obtained from Liston, G. E., and M. Sturm, 2021: Global Seasonal-Snow Classification, Version 1. National Snow and Ice Data Center, https://doi.org/10.5067/99FTCYYYLAQ0. <br>

**Watershed Elevation**
- Elevation data was obtained from the Copernicus DEM, a Digital Surface Model (DSM) derived from the WorldDEM, with additional editing applied to water bodies, coastlines, and other special features. European Space Agency (2024).  <i>Copernicus Global Digital Elevation Model</i>.  Distributed by OpenTopography.
- The data was accessed using the [easysnowdata](https://egagli.github.io/easysnowdata/examples/hydroclimatology_examples/) open source python module.  

# Data Pipleline - A Modular, Scalable Approach
The Frosty Dawgs datapipeline uses a medallion inspired datalake architecture with the tiers described below. The modular architecture is designed to provide future researchers with flexibiliy to update the data pipeline and approach at any stage of the pipeline, as desired.  Data is stored in S3 buckets corresponding to the Bronze, Silver, Gold, and Model Ready Tiers described below.  

The Pipeline is also scaleable. The Frosty Dawgs team used the pipeline to preprocess data for over 500 Huc12 sub-watershed in the Pacfic Northwest, spanning 15 different regional sub-Basins (Huc08 sub-Basins) listed below in the [Regions Available for Analysis](https://github.com/DSHydro/SnowML/blob/main/docs/Data_Pipeline.md#regions-available-for-analysis-) section. The code provided in this repo can be easily used to process data from any hydrological unit in the United States, at any level of granularity (e.g. Huc02, Huc04, . . . Huc12). Please consult the [DataPipe Notebook](https://github.com/DSHydro/SnowML/blob/main/notebooks/DataPipe.ipynb) for instructions on how to do so. 


## Bronze Data - Raw Data in Zarr Files 
Bronze data includes the raw SWE and metorological data acquired directly from the raw data sources. Data acquisition is challenging and time consuming given the amount of data and the fact that several sources make data available via separate files organized by year. Since our data acquistion pattern is most typically by region accross all years, the first step was to download the raw data and reconfigure it into zarr files with storage chuncks more suited to our access patterns. 

These zarr files are then saved in the S3 bucket "snowml-bronze."

The naming convention is "{var_short_name}_all.zarr". 

| Variable                     | Shortname        | Temporal Granularity | Temporal Scope    | Geographic Scope | Geographic Granularity |
|---------------------------|----------------|---------------------|----------------|-------------------|--------------------------|
| SWE                            | swe                  | Daily                         | WY83 – WY22      | CONUS                | 4km grid                       |
| Precipitation               | pr                     | Daily                         | 1/1/83-12/31/23 | CONUS+            | 4km grid                       |
| Air Temperature Daily Min | tmmn              | Daily                         | 1/1/83-12/31/23 | CONUS+            | 4km grid                       |
| Air Temperature Daily Max | tmmax            | Daily                         | 1/1/83-12/31/23 | CONUS+            | 4km grid                       |
| Relative Humidity – Daily Min | rmax              | Daily                         | 1/1/83-12/31/23 | CONUS+            | 4km grid                       |
| Relative Humidity Daily Max | rmin              | Daily                         | 1/1/83-12/31/23 | CONUS+            | 4km grid                       |
| Solar Radiation             | srad                 | Daily                         | 1/1/83-12/31/23 | CONUS+            | 4km grid                       |
| Wind Speed                  | vs                     | Daily                         | 1/1/83-12/31/23 | CONUS+            | 4km grid                       |


---

*Note1: WY stands for Water Year, which runs from Oct. 1 – Sept. 30.* <br>
*Note 2: The 4km grids used for the SWE data and the meteorological data are not fully aligned, but this discrepency is mitigated by the regional aggregation steps below.* 

## Silver Data - Static Variables, Calculated for each HUC Unit 
The model uses several static variables which do not vary over time, including snow-type, elevation ("dem"), and forest cover.  For each huc unit of interest, we calculated the mean value of each static variable for that huc unit.  Values are stored in the S3 bucket "snowml-silver." 



## Gold Data - SWE and Meteorological Data Aggregated by Variable, by Region (e.g. Huc12) 
Once the raw data has been retrieved and converted to zarr files otimized for region-based queries, the next step is to extract data for the regions(s) of interest.  For each region of interest -- for example 170300010402, the High Creek-Naneum Creek sub-watershed in the Naches Basin near Yakima -- we created "gold" data" for each of the SWE and Meteorological variables, as follows: <br>
1. Dynamically create a geopandas dataframe containing the geometry for the desired region, using ```snowML.datapipe get_geos``` module. <br>
2. Apply a geographical mask, using rioxarray, to extract from the bronze files filtered only the region of interest as specified by the geopandas df in step 1.  This step required some further processing of hte raw data, including renaming spacial dimensions, updating crs (coordinate systems), and/or calculating missing coordinate transform information for some of the data sets.  Please refer to the ```snowML.datapipe bronze_to_gold``` module for details. <br>
3. Aggregate the relevant variable into the regional mean for the selected region. <br>
4. Save results to a csv file.

Processed gold files were saved in to the S3 bucket "snowml-gold" with the naming convention "mean_{var_short_name}_in_{huc_no}.csv."  For example "mean_swe_in 170300010402." 



## Model Ready Data - All Variables For a Given Region
As the final step, for each region of interest, the huc-level means for each of the staticvariables (silver bucket) and the swe/meteorologic data (gold bucket) were aggregated into a single csv file containing all relevant variables.  Additional calculations were performed to update units to the value shown below in the [Model Ready Data](https://github.com/DSHydro/SnowML/blob/main/docs/Data_Pipeline.md#model-ready-data-) section. In addition, daily max and min airtemperature were averaged into a single daily average temperature variable; likewise daily max and min humidity was aggegated into a single daily relative humidity variable. Finally, data was filtered to the period 1983-10-01 through 2022-09-30 for all variables.  Please consult the module ```snowML.datapipe bronze_to_gold``` module for further details.  

*Note1: The Model Ready data is not yet normalized.  Normalization was performed dynamically with each expirement, normalizing data with reference to the training (and, where applicable, validation) sets relevant to that expirement.*<br>
* Note2: The snow type information was not used as a feature variable in training, but was used to calculate the predominant snow type for each huc for purposes of post-training analytics on the model.*


# Data Pipeline - Repo Steps <br>
For instructions on how to reproduce the data pipeline, or create model-ready data for additional huc units, please refer to our [DataPipe Notebook](https://github.com/DSHydro/SnowML/blob/main/notebooks/DataPipe.ipynb)

# Model Ready Data <br>
Data ready to be deployed into the Frosty Dawgs SnowML model can be downloaded from the S3 bucket named "snowML-model-ready". <br>
Each file contains data for one discrete HUC unit (watershed/subwatershed).

**Naming convention**
- Each file is named "model_ready_{huc_id}.csv"
- For example, "model_ready_huc170300010402.csv" contains data for huc_id 170300010402, a HUC 12 (sub-watershed) that is part of the Naches sub-basin near Yakima.

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
| Mean Elevation   | Mean Elevation for the region                                    | meters |
| Forest Cover     | Mean Forest cover for the region                                 | %      |



# Regions Available for Analysis <br>
The following regions have been processed at the Huc12 graunularity level and are currently available in the "snowML-model-ready" [bucket](arn:aws:s3:::snowml-model-ready): <br>

**Regions where Maritime Snow Predominates** 
  - [Chelan Basin](basin_fact_sheets/Chelan(17020009).md) (17020009)
  - [Sauk](basin_fact_sheets/Sauk(17110006).md) (17110006)
  - [Skagit Basin](basin_fact_sheets/Skagit(17110005).md) (17110005) (*Data for Huc10 granularity also available)
  - [Skykomish](basin_fact_sheets/Skykomish(1711009).md) (17110009)
  - [Wenatche](basin_fact_sheets/Wenatche(17020011).md) (17020011)

**Regions where Montane Forest Snow Predominates** 

  - [Middle Salmon Chamberlain](basin_fact_sheets/Middle_Salmon-Chamberlain(17060207).md)(17060207)
  - [St.Joe](basin_fact_sheets/St._Joe(17010304).md) (17010304)
  - [South Fork Coeur d'ALene](basin_fact_sheets/South_Fork_Coeur_d'Alene(17010302).md) (17010302)
  - [South Fork Salmon River](basin_fact_sheets/South_Fork_Salmon_River(17060208).md)(17060208)
  - [Upper Couer d'Alene](basin_fact_sheets/Upper_Coeur_d'Alene(17010301).md) (17010301)
  
**Mixed Regions** 
  - [Naches](basin_fact_sheets/Naches(17030002).md) (17030002) (mix with Maritime/Montane Forest)
  - [Stillaguamish](basin_fact_sheets/Stillaguamish(17110008).md) (17110008) (Maritime/Ephemeral)
  - [Upper Yakima](basin_fact_sheets/UpperYakima(17030001).md) (17030001) (Ephemeral/Maritime/Montane Forest mix)

**Regions where Ephemeral Snow Predominates:** 
- [Lower Yakima](basin_fact_sheets/Lower_Yakima(17030003).md) (17030003)
- [Lower Skagit](basin_fact_sheets/LowerSkagit(17110007).md) (17110007)
- [Upper_Columbia-Entiat](basin_fact_sheets/Upper_Columbia-Entiat(17020010).md) (17020010)
  
