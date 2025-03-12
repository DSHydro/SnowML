 # Experiment 2: Investigate Model Performance Across Varied Huc12 Sub-Watersheds

In the second experiment, we examined how the simple, local training LSTM model performed accross a variety of Huc12 (sub-watershed) units.  Again the model aimed to predict swe values using an LSTM model with mean_temperature and mean_precipitation as the feature inputs.  

We examined 534 watersheds with a variety of predominant snow types - Ephemeral, Maritime, and Montane Forest. Figure 1 maps the Huc12 units by snow type. 

Each individual Huc12 unit was trained using data only from that same Huc12 unit, and tested on later years of data using a train/test split of .67.  


# Observations and Results 

## Variation in Model Effectiveness by Snow Type

Figure 2 visualized variation in goodness of fit measures accross different dimensions. Figure  reveals stark contrast in goodness of fit between the relevant snow classes, with Ephemral snow classes generally performing the worst in terms of KGE Efficiency. Ephemeral basins perform relatively well in terms of MSE, likley because the lower levels of snow in these basins inherently create lowere MSE as MSE is a unit dependent measure.  Montane Forest regions outperform Maritime regions with respect to both Test KGE and Tese MSE. The small number of Prairie and Boreal Forrest watersheds are excluded from Figures 2 and 3 due to the low sample size in these categories. 

We ran pairwise t-tests with unequal variance (Welch Test) to test for inequality of mean Test KGE and mean Test MSE between snow types. All values were significaant at p = 0.001 level.  

## Variation in Model Effectiveness by Elevantion of Tested Region

Elevation is also highly correlated with how well a watershed is likely to be predicted by a locally trained model, with higher elevations performing better, as shown in Figure 3.  This relationship is more pronounced at low to mid elevations, and with Ephemeral and Maritime Snow. To a certain extent, differences in performance among snow classes may simply be "passing through" differences in elevation that are also correlated with snow class types.  


## Impressive Results in Select Basins <br>
While the variation in results accross Huc12 sub-watersheds is interesting and suggests avenues for future refinements, it is worth emphasizing that the current, fairly simply model produces relatively reliable resutls in select basins, as shown in Figure 4. **Thus, water managers monitoring swe in high elevation and/or montane forest regions could consider using this model as a simple forecasting tool given predictions of future meteorological variables (precipitation, temperatures).**  This is a practical result because this locally trained model - unlike the multi-huc model discussed in Expirement 3 - can be trained quickly and with relatively limited compute power. Using the code in this repo, a local Huc12 sub-watershed model can be trained in approximately 10 minutes using a high end laptop computer. (When running in laptop mode, we used a 13th Gen Intel Core i9-13900H processor with 20 threads (10 cores, 2 threads per core), on a computer with 6GB of available RAM. 

## Divergence in Goodness of Fit Measures 
Figure 5 plots [Test KGE](https://github.com/DSHydro/SnowML/blob/main/docs/Ex2_VariationByHuc.md#what-is-kge) and Test MSE. The two goodness of fit measures diverge significantly for many of the Huc12 units, especially for regions dominated by ephemeral snow. In hydrology, KGE is typically considered the more relevant goodness of fit meausure, however, it is difficult to use directly as a loss function because it is not easily differentiable.  Nonetheless, figure 3 highlight the imprecision introduced by using MSE as a loss function if the ultimate goal is to produce high values of KGE.  We briefly expiremented with using KGE, or a hybring KGE+MSE loss function during training but observed impractical training times and chaotic results, so did not further pursue this avenue at this time.  Nontheless, investigation into the best loss function strategy is a ripe area for future research. 

# Tables and Figures 

## Figure1 - Map of Huc12 Units Used in this Expirement, by Snow Type. 


**Legend** <br>
 - Blue- Maritime Snow Predominates (155)
 - Green - Montane Forest Snow Predominates (187)
 - Lilac - Ephemeral Snow Predominates (180)
-  Light Green - Prairie Snow Predominates (11)
-  White - Boreal Forest (1)

![Map of Huc12 Units Tested - by Snow Type](https://github.com/DSHydro/SnowML/blob/main/notebooks/Ex2_VarianceByHuc/charts/Expirement2Map.png)


## Figure2 - Goodness of Fit Measures By Snow Type

| Test KGE | Test MSE |
|----------|----------|
| ![Test KGE By Predominant Snow Type - Excludes Ephemeral ](https://github.com/DSHydro/SnowML/blob/main/notebooks/Ex2_VarianceByHuc/charts/Boxplot%20of%20Test%20KGE%20by%20Predominant%20Snow%20Type%20-%20Locally%20Trained%20Hucs%20-%20Excludes%20Ephemeral.png) | ![Test MSE By Predominant Snow Type - Excludes Ephemeral](https://github.com/DSHydro/SnowML/blob/main/notebooks/Ex2_VarianceByHuc/charts/Boxplot%20of%20Test%20MSE%20by%20Predominant%20Snow%20Type%20-%20Locally%20Trained%20Hucs%20_Excludes%20Ephemeral_.png) |
|:---:|:---:|
| ![Test KGE by Predominant Snow Type - Includes Ephemeral](https://github.com/DSHydro/SnowML/blob/main/notebooks/Ex2_VarianceByHuc/charts/Boxplot%20of%20Test%20KGE%20by%20Predominant%20Snow%20Type%20-%20Locally%20Trained%20Hucs.png) | ![Test MSE By Predominant Snow Type - Includes Ephemeral](https://github.com/DSHydro/SnowML/blob/main/notebooks/Ex2_VarianceByHuc/charts/Boxplot%20of%20Test%20MSE%20by%20Predominant%20Snow%20Type%20-%20Locally%20Trained%20Hucs.png) |


Results from pair-wise Welch's t-test of null hypothesis of equality of mean_kge in different snow_type classes
| Group1           | Group2           | P-Value        |
|------------------|------------------|----------------|
| Ephemeral        | Maritime         | 3.964527e-13   |
| Ephemeral        | Montane Forest   | 9.953465e-22   |
| Maritime         | Montane Forest   | 2.322637e-09   |

Results from pair-wise Welch's t-test of null hypothesis of equality of mean_kge in different snow_type classes
| Group1           | Group2           | P-Value        |
|------------------|------------------|----------------|
| Ephemeral        | Maritime         | 6.300075e-23   |
| Ephemeral        | Montane Forest   | 1.279809e-16   |
| Maritime         | Montane Forest   | 1.869237e-19   |



## Figure 3 - Test KGE By Mean Elevation of Huc12 Sub-Watershed

| ![All Snow Types](https://github.com/DSHydro/SnowML/blob/main/notebooks/Ex2_VarianceByHuc/charts/Test%20KGE_vs_mean_elevation_Locally%20Trained%20HUCs.png) | ![Maritime Only](https://github.com/DSHydro/SnowML/blob/main/notebooks/Ex2_VarianceByHuc/charts/Test%20KGE_vs_mean_elevation_Locally%20Trained%20Hucs%2C%20Maritime%20Only.png) |
|:---:|:---:|
| ![Montane Forest](https://github.com/DSHydro/SnowML/blob/main/notebooks/Ex2_VarianceByHuc/charts/Test%20KGE_vs_mean_elevation_Locally%20Trained%20Hucs%2C%20Montane%20Forest%20Only.png) | ![Ephemeral](https://github.com/DSHydro/SnowML/blob/main/notebooks/Ex2_VarianceByHuc/charts/Test%20KGE_vs_mean_elevation_Locally%20Trained%20Hucs%2C%20Ephemeral%20Only.png) |

The R-squared value for each chart represents the proportion of the variance in Test KGE observed between different Hucs that can be attributed to differences in mean basin elvation. 


## Figure 4- Goodness of Fit Measures By Basin
| Test KGE By Huc08 SubBasin | Test MSE By Huc08 SubBasin |
|--------------|--------------|
| ![KGE By Basin](https://github.com/DSHydro/SnowML/blob/main/notebooks/Ex2_VarianceByHuc/charts/Boxplot%20of%20Test%20KGE%20by%20Basins%20-%20Locally%20Trained%20Sub-Watersheds%20_Excludes%20Ephemeral_.png) | ![MSE By Basin](https://github.com/DSHydro/SnowML/blob/main/notebooks/Ex2_VarianceByHuc/charts/Boxplot%20of%20Test%20MSE%20by%20Basins%20-%20Locally%20Trained%20Sub-Watersheds%20_Excludes%20Ephemeral%20Sub-Watersheds_.png)|


## Figure5 
| TEst KGE vs. Test MSE - All Snow Types | Test KGE vs. Test MSE - Maritime and Montane Forest |
|------------------------------|------------------------------------------|
| ![KGE vs. MSE - All Snow Types](https://github.com/DSHydro/SnowML/blob/9b099ff3cf6c3d787694b100911481a9cdb9f3a1/notebooks/Ex2_VarianceByHuc/charts/Scatter_Plot_of_Test_KGE_vs_Test_MSE_(Colored_by_Predominant_Snow_Type).png) | ![KGE vs. MSE - Maritime and Montane Forest](https://github.com/DSHydro/SnowML/blob/9b099ff3cf6c3d787694b100911481a9cdb9f3a1/notebooks/Ex2_VarianceByHuc/charts/Test%20KGE%20vs.%20Test%20MSE%2C%20Excluding%20Hucs%20where%20Ephemeral%20Snow%20Predominates.png) |


[**TO DO** - Insert example plots of mismatched MSE/KGE]





# Limitations and Questions For Further Research
- We did not perform significant hyperparameter tuning. Future researchers may wish to investigate hyperparameter tuning using stratified cross-validation.  Moreover, given the differences in model performance accross snow types and elevation, it would be worth purusing whether different snow types/elevations require different sets of optimized hyperparameters.  
- The model may benefit from additional meteorological variables, such as solar radiation, humidity, or wind speed, an aspect we investigate in Expirement 3.
- As discussed w/r/t Expirement 1, the training data SWE values are themselves a model, so any imprecisions in the training SWE values would make model results diverge from ground truth even assuming the model perfectly predicted the training SWE values.
- As noted above, using MSE as a training loss function may not optimize for the best KGE fit, but in hydrology KGE is viewed as the superior goodness of fit measure.  Using KGE itself as a loss funciton is challenging because the measure is not easily differentiable and therefor highly computationally intensive. Training with KGE as a loss function can also lead to instability/lack of model convergence, as we observed directly with some initial experimentation using KGE or hybrid loss functions. Nonetheless, further research into using KGE as a training loss measure is warranted.  Future researchers could consider hybrid training strategies, such as using MSE as the loss function in early epochs and switching to a KGE or hybird measure only in later epochs when the model has stabilized. 

# What is KGE? 
[Klinge-Gupta Efficiency](https://en.wikipedia.org/wiki/Kling%E2%80%93Gupta_efficiency), is a metric is commonly used to assess the performance of hydrological models. KGE is a composite measure that considers (i) Correlation (r) between observed and simulated data, (ii) the Bias (β) assesaws S the ratio of the mean of simulated data to the mean of observed data, and (iii) Variability (y), which compares the standard deviations of simulated and observed data to evaluate the model's ability to reproduce the variability in the observed data.  It is calculated as KGE=1− sqrt((r−1)^2+(β−1)^2+(γ−1)^2).  

KGE values range from negative infinity to 1, with a KGE value of 1 indicating perfect agreement between the simulated and observed data. A model which simply predicts the mean will have a KGE of -0.44.  What is considered a "good" KGE score is context specific.  For our expirements, we considered KGE > 0.7 to be acceptable, KGE >0.8 to be good, and KGE > 0.9 to be excellent based on literature review of similar expirements.  


# How to Reproduce The Results
The results for this expirement were produced using the `snowML.LSTM` package in this repo.  The hyperparameters were set as shown in the section below. The expirement was then run by importing the module `local-training-expirement.py` and by calling the function
`run_expirement()` Note that during training data is split into batches and shuffled for randomness, so different runs of the same expirement may result in somewhat different outcomes. 


The metrics discussed above were downloaded from ML flow using [this notebook](https://github.com/DSHydro/SnowML/blob/d1653c0b190fa6e54b4473dc1d4808fe5c590e81/notebooks/Ex2_VarianceByHuc/DownloadMetrics.ipynb) and analyzed using [this notebook](https://github.com/DSHydro/SnowML/blob/main/notebooks/Ex2_VarianceByHuc/LSTM_By_Huc.ipynb)). 

# Model Parameters

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
| MLFlow Expirement   | Single All                 |
| ML Flow Run IDs|  69d929bbdfdd43b4a2f45b823d945eb7, b0fa8180481a4dc6954f138e9ee934aa, aca5e4d1b42044a2aeabc055a0b14d8d|

 


