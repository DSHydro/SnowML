# Welcome to the Frosty Dawgs!

Welcome to our capstone Github. Snowpack is a critical part of the hydrological cycle in the Western United States, where it acts like a natural reservoir, storing water in winter and gradually releasing it in spring and summer. Snow water equivalent (SWE) determines how much water the snowpack contains. Accurate estimation of SWE is essential for a variety of important modeling tasks such as water resource management, avalanche conditions, and river flow forecasting. This metric drives accurate predictions of the volume of snowmelt runoff, subsequently affecting water supply (Liljestrand et al.). Earth science research teams have begun to assess the efficacy of machine learning models using physics-based model predictions with varying degrees of geospatial or temporal granularities (Steele et al.).

Even with improvements to snowpack estimation with ML models, challenges such as retention of temporal information and comprehensive aggregated datasets exist (Thapa et al.). To address these challenges, we propose a workflow that integrates machine learning techniques with physics-based model predictions that leverage a large-scale spatiotemporal dataset to improve the accuracy and resolution of snowpack estimation in the Western US.

This project houses three primary experiments: 

\- 1) [Experiment 1: Use More Data](https://github.com/DSHydro/SnowML/blob/main/docs/Ex1_MoreData.md)

\- 2) [Investigate Model Performance Across Varied Huc-12 Sub-Watersheds](Experiment 2: https://github.com/DSHydro/SnowML/blob/main/docs/Ex2_VariationByHuc.md)

\- 3) [Multi-Watershed Training](https://github.com/DSHydro/SnowML/blob/main/docs/Ex3_MultiHucTraining.md)

**Team Name:** Frosty Dawgs (this is our informal name)

**Team Project Title:** Deep Snow, Deep Learning: Refining Predictions for Water Resource Management

**Team Members:** Sue Boyd, Sarah Kilpatrick, Derek Tropf


1. docs/ Directory

Contains documentation related to the project:​

**Data_Pipeline.md:** Details the data pipeline, including sources of raw data, processing steps, and regions available for analysis. ​
GitHub

**basin_fact_sheets/:** Holds fact sheets for various basins, providing information such as predominant snow types and elevation maps. Examples include:​

[Chelan](https://github.com/DSHydro/SnowML/blob/main/docs/basin_fact_sheets/Chelan%2817020009%29.md)

[Upper Coeur d'Alene](https://github.com/DSHydro/SnowML/blob/main/docs/basin_fact_sheets/Upper_Coeur_d'Alene(17010301).md)

2. notebooks/ Directory

Contains Jupyter notebooks used in the project:​

**DataPipe.ipynb:** Provides instructions on reproducing the data pipeline or creating model-ready data for additional HUC units. ​

3. src/snowML/ Directory

This folder holds the core Python source code.

4. mlflow_data/ Directory

Stores data related to MLflow experiments and tracking.​

5. Root Directory Files

**.gitignore:** Specifies files and directories to be ignored by Git.​

**Frosty Dawgs 2025 Final Poster:** Final project's poster presented in the 2025 capstone event.

**LICENSE:** Contains the project's license information.​

**README.md:** Provides an overview of the project, including installation instructions and development guidelines.​

**Regridding - Spatial Interpolation Guide.pages:** Guide on regridding and spatial interpolation techniques to be used in the future.

**Task Directory:** Offers ideas for project continuation, categorized by completion status and modules/tools. ​

**pyproject.toml:** Configuration file for Python project metadata.​

**requirements.txt:** Lists Python dependencies required for the project. ​
GitHub

**setup.py:** Script for installing the SnowML package. ​
GitHub

## Data 
Read about our [data pipleine](docs/Data_Pipeline.md)

## Development

### Installation Instructions
Clone the repo and then do the following:
`cd SnowML`
`pip install -e .`

### MLflow
Follow [installation instructions](https://mlflow.org/docs/latest/getting-started/tracking-server-overview/index.html#minute-tracking-server-overview) or quickly install and run below:
1. Activiate a virtual environment through conda or pip first
2. `pip install mlflow`
3. Open a terminal, `cd` to a directory where you model will run, and then type the command `mlfow ui` and press entire.
4. Keep the termial alive, open a browser, and navigate to `localhost:5000` and press enter. This should direct to the MLflow enterface where all experiments and things live.
 
## New Teammates

### Where to Begin?
the **data** folder contains our outputs from loss functions. The **docs** folder contains the most information, check out Experiments 1-3 and the data-pipeline file for an in-depth description of our data pipeline. The **notebooks** folder contains the initial model provided by our sponsor and our LSTM model jupyternotebook scripts. 

### Task Directory
Take a look at the task directory for ideas on where to take this project and interesting ideas to include in an academic poster. Sort the directory by completion status, type of project, and document your progress.
