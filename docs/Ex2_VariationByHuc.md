# Expirement 2: Investigate Model Performance Accross Hucs 

[** TO BE INSERTED **] 


# Observations and Results 

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

# How to Reproduce The Results
The results for this expirement were produced using the snowML.LSTM package in this repo.  The hyperparameters were set as shown in the section below. 
The training/validation/huc splits are also recorded below.  The expirement was then run by importing the module `multi-huc-expirement.py` and by calling the function
`run_expirement(train_hucs, val_hucs, test_hucs)` Note that during training data is split into batches and shuffled for randomness, so different runs of the same expirement may result in somewhat different outcomes. 

The results were gathered over three MLflow expirements(using the same parameters), each on a subset of the total hucs used, to make run_times manageable.


The metrics discussed above were downloaded from ML flow using [this notebook](https://github.com/DSHydro/SnowML/blob/d1653c0b190fa6e54b4473dc1d4808fe5c590e81/notebooks/Ex2_VarianceByHuc/DownloadMetrics.ipynb) and analyzed using this notebook **TO INSERT**

## Model Parameters

| Parameter             | Value                        |
|-----------------------|----------------------------|
| hidden_size          | 64                           |
| num_class           | 1                            |
| num_layers         | 1                            |
| dropout            | 0.5                          |
| learning_rate      | 0.001                        |
| n_epochs           | 10                           |
| lookback           | 180                          |
| batch_size         | 64                           |
| n_steps            | 1                            |
| num_workers        | 8                            |
| var_list           | ['mean_pr', 'mean_tair']     |
| expirement_name    | Single All                   |
| loss_type          | mse                          |
| mse_lambda         | 1                            |
| train_size_dimension | time                        |
| train_size_fraction | 0.67                         |
| MLFlow Expirement   | Single ALl                  |
| ML Flow Run IDs|  69d929bbdfdd43b4a2f45b823d945eb7, b0fa8180481a4dc6954f138e9ee934aa, aca5e4d1b42044a2aeabc055a0b14d8d|

 


