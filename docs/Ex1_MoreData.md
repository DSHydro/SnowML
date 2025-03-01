# Expirement 1: Use more Data! 

** TO DO: Explain that the original prototyped model was run on 8 Huc10 watersheds within Skagit and we did same **

The first adjustment we made to the proptoyped LSTM Model was to use the University of Arizona estimates of Snow Water Equivilent (SWE) [data]( https://climate.arizona.edu/data/UA_SWE/) as our target dataset for training and evaluating the model. This dataset contains a longer time series of available SWE data than used in the protoyped model. 

We reran the protoype LSTM model with the new data, leaving all other hyperparmeters unchanged except one. We reduced the number of epochs to 10 after observing early convergence of the model -- possibly due to the increased training data available. The graphs below compare the prior data run at 200 epochs with the new data run at 10 epochs, but the observations are robust to the number of epochs used. Please refer to the [Viz10COmpare notebook](notebooks/Prototype_Model_Results/VizHuc10Compare.ipynb) for sensitivy analysis related to number of epochs used.  

In this Expirement 1, each Huc10 watershed was trained *only* using the data from that wattershed. Train/test split was accomplished by reserving the final two thirds of the time period as test data. 


# Observations and Results 

**More Data = Better Results** <br>
Not surprisingly, including longer time series of data generally increased model fit. Figure 1 below graphs actual and predicted levels of swe for two example Huc10 units within the [Skagit Basin](docs/basin_fact_sheets/Skagit(17110005).md), 1711000504 and 171100506, using both data sets. From visual inspection, using the longer time series from the UA data set appears to decrease overfitting and aid model performance. 

These visual observations are confimed by two goodness of fit measures: Mean Squared Error ("MSE"), and  [Klinge-Gupta Efficiency](https://github.com/DSHydro/SnowML/blob/main/docs/Ex1_MoreData.md#figure1--ua-data-vs-original-data-prediction-plots-for-two-example-huc12-units---1711000504-and-1711000506) "KGE". As shown in Figure 2, the longer time series data showed test KGE improvements for each of the eight watersheds tested. Mean Squared Error results were more mixed. In the majority of watersheds, MSE improved when running the model with the longer time series of data, but in two watersheds, MSE deteriorated with the increased time series of data.

**Discrepency in Model Performance Accross Different Huc10 Units** <br>
The variation in model performance in different Huc10 watershed units is also notable, given that in this expirement a separate model was trained for each watershed. In Expirement 2, we investigate perforance variation acrros different huc units in more detail. 

**Variability in "Actual" SWE Between UA and SnowTel Datasets** <br>
[** TO BE INSERTED **] 

## Figure1
**UA Data vs. SnowTel Data Prediction Plots for two Example HUC10 Units - 1711000504 and 1711000506**

| ProtoTyped Model - UA Data | ProtoTyped Model - Original (SnowTel) Data |
|----------------------------|---------------------------------|
| ![UA Data](../notebooks/Ex1_MoreData/charts/UAData_SWE_Post_Predictions_for_huc_1711000504.png) | ![Original Data](../notebooks/Ex1_MoreData/charts/SWE_Post_Predictions_for_huc_1711000504.png) |
| ![UA Data](../notebooks/Ex1_MoreData/charts/UAData_SWE_Post_Predictions_for_huc_1711000506.png) | ![Original Data](../notebooks/Ex1_MoreData/charts/SWE_Post_Predictions_for_huc_1711000506.png) |

## Figure2

**KGE - Higher Values (Closer to 1) Represent Better Fit** 
![KGE_Compare](https://github.com/DSHydro/SnowML/blob/7b9d88797ac90603c03b732958c8c35ee3aa0d18/notebooks/Ex1_MoreData/charts/Klinge_Gupta_Efficiency_By_Huc_and_Type_of_Data_Used.png)

**MSE - Lower Values (Closer to 0) Represent Better Fit**
![MSE_Compare](https://github.com/DSHydro/SnowML/blob/399f07908822c42e43942bc28d908fbc3a58ab06/notebooks/Ex1_MoreData/charts/Mean_Square_Error_By_Huc_and_Type_of_Data_Used.png)

# What is KGE? 
[Klinge-Gupta Efficiency](https://en.wikipedia.org/wiki/Kling%E2%80%93Gupta_efficiency), is a metric is commonly used to assess the performance of hydrological models. KGE is a composite measure that considers (i) Correlation (r) between observed and simulated data, (ii) the Bias (β) assesaws S the ratio of the mean of simulated data to the mean of observed data, and (iii) Variability (y), which compares the standard deviations of simulated and observed data to evaluate the model's ability to reproduce the variability in the observed data.  It is calculated as KGE=1− sqrt((r−1)^2+(β−1)^2+(γ−1)^2).  

KGE values range from negative infinity to 1, with a KGE value of 1 indicating perfect agreement between the simulated and observed data. A model which simply predicts the mean will have a KGE of -0.44.  What is considered a "good" KGE score is context specific.  For our expirements, we considered KGE > 0.7 to be acceptable, KGE >0.8 to be good, and KGE > 0.9 to be excellent based on literature reviwe of similar expirements.  


# Limitations and Questions For Further Research
[** TO BE INSERTED **]

# How to Reproduce The Reults 
The results for this expirement were produced using jupyter notebooks: [Ex1: Notebook - Snowtel Data](https://github.com/DSHydro/SnowML/blob/a38b71732907f79f4150c4a3fa5794b1a0aafe2e/notebooks/Prototype_Model_Results/Original_Tutorial_Time_Series_Prediction_of_Snow_Water_Equivalent_(SWE)_Using_LSTM_in_PyTorch.ipynb)  and [Ex1: Notebook - UA Data](https://github.com/DSHydro/SnowML/blob/a38b71732907f79f4150c4a3fa5794b1a0aafe2e/notebooks/Prototype_Model_Results/Original_Tutorial_Time_Series_Prediction_of_Snow_Water_Equivalent_(SWE)_Using_LSTM_in_PyTorch-NewData.ipynb).  Parameters used resulting mdoel weights, and diagnostic metrics were all saved in MLFlow for later retreiveal and analysis. 

Note that during training data is split into batches and shuffled for randomness, so different runs of the same expirement may result in somewhat different outcomes. 

## Model Parameters

| Parameter          | Value (UA Data)         | Value (SnowTel Data)       |
|--------------------|----------------------|----------------------|
| Training From     | Prototype Notebook    | Prototype Notebook  |
| Data Source       | UA SWE Data           | SnowTel Data        |
| Variables         | "mean_pr", "mean_tair" | "mean_pr", "mean_tair" |
| input_size        | 2                      | 2                    |
| hidden_size       | 64                     | 64                   |
| num_class        | 1                      | 1                    |
| num_layers       | 1                      | 1                    |
| dropout          | 0.5                    | 0.5                  |
| learning_rate    | 0.001                  | 0.001                |
| train_size_fraction | 0.67                | 0.67                 |
| lookback_values  | [180]                  | [180]                |
| epochs          | **10**                      | **200**            |
| MLFlow Expirement Name | Prototype-Results | Prototype-Results |
| MLFlow run name       | bittersweet-bug-588       | clumsy-ray-666       |
| MLFLow run id    | f0bde540d8184f7e8ea057456ec56f38 | 41f1a7b318f94b2bb2c4fdfbf892a018 |


