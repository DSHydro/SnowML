# Welcome to the Frosty Dawgs!

Welcome to our capstone Github. Snowpack is a critical part of the hydrological cycle in the Western United States, where it acts like a natural reservoir, storing water in winter and gradually releasing it in spring and summer. Snow water equivalent (SWE) determines how much water the snowpack contains. Accurate estimation of SWE is essential for a variety of important modeling tasks such as water resource management, avalanche conditions, and river flow forecasting. This metric drives accurate predictions of the volume of snowmelt runoff, subsequently affecting water supply (Liljestrand et al.). Earth science research teams have begun to assess the efficacy of machine learning models using physics-based model predictions with varying degrees of geospatial or temporal granularities (Steele et al.).

Even with improvements to snowpack estimation with ML models, challenges such as retention of temporal information and comprehensive aggregated datasets exist (Thapa et al.). To address these challenges, we propose a workflow that integrates machine learning techniques with physics-based model predictions that leverage a large-scale spatiotemporal dataset to improve the accuracy and resolution of snowpack estimation in the Western US.

**Team Name:** Frosty Dawgs (this is our informal name)

**Team Project Title:** Deep Snow, Deep Learning: Refining Predictions for Water Resource Management

**Team Members:** Sue Boyd, Sarah Kilpatrick, Derek Tropf

### Where to Begin?
the **data** folder contains our outputs from loss functions. The **docs** folder contains the most information, check out Experiments 1-3 and the data-pipeline file for an in-depth description of our data pipeline. The **notebooks** folder contains the initial model provided by our sponsor and our LSTM model jupyternotebook scripts. 

This project houses three primary experiments in the **docs** folder: 

# Take a look at our experiments:

## 1) [Experiment 1: Use More Data](https://github.com/DSHydro/SnowML/blob/main/docs/Ex1_MoreData.md)

### Methodology:
This experiment evaluated an LSTM model for predicting Snow Water Equivalent (SWE) using watershed-scale training. A new dataset from the University of Arizona was incorporated, extending the available time series from 2012 all the way up until 2022. The model was trained separately for seven HUC10 watersheds, with a 2/3 train and 1/3 test split.

### Main Results:
- **Extended Data Improved Model Fit:** Using a longer time series reduced overfitting and generally improved KGE scores, though MSE results varied.
- **Performance Varied by Watershed:** Model accuracy differed across HUC10 watersheds, highlighting potential local influences.
- **Dataset Discrepancies Impact Results:** Differences between SWE datasets indicate model predictions depend on input data accuracy.

## 2) [Investigate Model Performance Across Varied Huc-12 Sub-Watersheds](https://github.com/DSHydro/SnowML/blob/main/docs/Ex2_VariationByHuc.md)

### Methodology:
This experiment examined the performance of a simple, locally trained LSTM model across 533 HUC12 sub-watersheds. The model aimed to predict Snow Water Equivalent (SWE) using mean temperature and mean precipitation as feature inputs. By using the smaller HUC12 scale, data availability for training was increased compared to previous experiments.

### Main Results:
- **Variation in Model Effectiveness by Snow Type:** The model exhibited large differences in performance across snow types. Ephemeral snow regions had the lowest Kling-Gupta Efficiency (KGE) values, indicating poorer predictive performance. However, their Mean Squared Error (MSE) was relatively low, likely due to inherently lower SWE values. Montane Forest regions outperformed Maritime regions in both KGE and MSE.Statistical tests (Welch’s t-test) confirmed significant differences in mean Test KGE and Test MSE across snow types (p < 0.001).
- **Elevation Strongly Correlates with Model Performance:** Higher-elevation watersheds demonstrated better predictive performance. The relationship between elevation and model accuracy was particularly strong at low-to-mid elevations, and especially pronounced for Ephemeral and Maritime snow types. This suggests that elevation may be a key factor influencing SWE predictability, potentially overriding differences in snow classification.
- **Reliable Performance in Select Basins:** Despite variations in overall model performance, certain high-elevation and Montane Forest basins showed strong predictive capability. These results suggest that water managers in well-performing basins could use this locally trained model as a simple, computationally efficient forecasting tool. The model can be trained in approximately 10 minutes on a high-end laptop (Intel Core i9-13900H, 20 threads, 6GB RAM), making it practical for real-time applications.
- **Divergence in Goodness of Fit Measures:** KGE and MSE diverged significantly in some HUC12 sub-watersheds, particularly those dominated by Ephemeral snow. While KGE is a preferred metric in hydrology, it is challenging to use directly as a loss function due to its non-differentiability. Initial experiments with KGE or hybrid KGE-MSE loss functions resulted in instability and impractical training times. Further research is warranted to explore optimized loss functions, potentially employing a hybrid approach where MSE is used in early epochs and KGE in later epochs once the model stabilizes.

## 3) [Multi-Watershed Training](https://github.com/DSHydro/SnowML/blob/main/docs/Ex3_MultiHucTraining.md)

### Methodology

Experiment 3 investigated model generalization to ungauged basins and the impact of variable selection and learning rate on stability. Training focused on 270 Huc12 sub-watersheds in Maritime and Montane Forest regions of the Pacific Northwest, excluding areas dominated by ephemeral snow. These were randomly split into training (60%), validation (20%), and Test Set A (20%). An additional Test Set B included Huc12 units from the Upper Yakima and Naches basins, representing fully ungauged regions.

Eight model variations were tested using four feature sets (Base, plus Solar Radiation, Wind Speed, or Humidity) at two learning rates (0.001 and 0.0003). Each model was trained for 30 epochs, with validation performance monitored at each epoch. The best model was selected based on the highest median Test KGE across the validation set. This model was then evaluated on both test sets to assess generalization.

### Main Results:

- **Variable Selection & Learning Rate:** The eight model variations reached similar peak KGE values (0.78–0.82). The Base Model plus Humidity at a 0.0003 learning rate remained stable across epochs, while other models oscillated before overfitting. Unexpectedly, models trained at the lower 0.0003 rate showed greater early-epoch noise.

- **Model Generalization:** The LSTM model generalized well to untrained basins. Median KGE values were 0.85 for Test Set A and 0.82 for Test Set B, slightly lower than in Experiment 2. Results suggest reasonable transferability.

- **Snow Type Performance:** No significant difference in KGE was observed between Maritime and Montane Forest snow classes, but MSE varied significantly. This suggests error magnitude differed by snow type, even though overall model accuracy remained similar across classes.

- **Elevation Effects:** Unlike Experiment 2, no significant correlation was found between mean elevation and model performance, likely due to multi-Huc12 training and the explicit inclusion of elevation as a model feature.

# Task Directory
View the task directory for ideas on where to take this project and interesting ideas to include in an academic poster. Sort the directory by completion status, type of project, and document your progress.

## Future Research Directions:
1) Investigate optimal hyperparameter tuning for different snow types and elevations.
2) Expand feature inputs to include additional meteorological variables such as solar radiation, humidity, and wind speed.
3) Further refine loss function strategies to better optimize for KGE without sacrificing training stability.

# Folder Guide

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
Read about our [Data Pipeline Here](docs/Data_Pipeline.md)

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

