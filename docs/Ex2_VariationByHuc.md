# Expirement 2: Investigate Model Performance Accross Huc12 Sub-Watersheds

In the second expirement, we examined how the simple, local training LSTM model performed accross a variety of Huc12(sub-watershed) units.  Again the model aimed to predict swe values using an LSTM model with mean_temperature and mean_precipittion as the feature inputs.  

We examined 534 watersheds with a variety of predominant snow types - Ephemeral, Maritime, and Montane Forest. Figure 1 maps the Huc12 units by snow type. 

Each individual Huc12 unit was trained using data only fromthat same Huc12 unit, and tested on later years of data using a train/test split of .67.  


# Observations and Results 

Several interesting observations result from this expirement: 

**Variation in Model Effectiveness by Snow Type and Model Elevation** <br>
Figure 2 visualized variation in goodness of fit measures accross different dimensions. Figure 2A reveals stark contrast in goodness of fit between the relevant snow classes, with Ephemral snowclasses generally performing the worst in terms of KGE Efficiency. Ephemeral basins perform relatively well in terms of MSE, likley because the lower levels of snow in these basins inherently create lowere MSE as MSE is a unit dependent measure.  Montane Forest basin outperform Epheremal basins on both Test KGE and Tese MSE. The small number of Prairie and Boreal Forrest watersheds are excluded from Figure 2A-2D due to the low sample size in these categories.  

Figures 2B and 2C explore the differences between Montane Forest sub-watersheds and Maritime sub-watersheds. Because the more important goal in many contexts is the modelling of *non*-ephemeral regions, we focus in these charts only on sub-watersheds with non-ephermeral snow.  In terms of both KGE (higher is better) and MSE (lower is better) measures, Montane Forest sub-watershed are better predicted by the locally trained model than in Maritime sub-waterheds (Figure 2A).  Elevation is also highly correlated with how well a watershed is likely to be predicted by a locally trained model, with higher elevations performing better. (Figure 2B).  To a certain extent, differences in performance among snow classes may simply be "passing through" differences in elevation that are also correlated with snow class types.  However, snow classification appears to remain relevant even controlling for elevation. (Figure 2C).    

Finally, Figure 2D plots model performance by HUC08 region within the Pacific North West.  

**Impressive Results in Select Basins** <br>
While the variation in results accross Huc12 sub-watersheds is interesting and suggests avenues for future refinements, it is worth emphasizing that the current, fairly simply model produces relatively reliable resutls in select basisn. **Thus, watermanagers monitoring swe in high elevation and/or montane forest regions could consider using this model as a simple forecasting tool given predictions of future meteorological variables (precipitation, temperatures).**  This is a practicla result because this locally trained model - unlike the multi-huc model discussed in Exhibit 3 - can be trained quickly and with relatively limited compute power. Using the code in this repo, a local watershed model can be trained in approximately 10 minutes using a high end laptop computer. (When running in laptop mode, we used a 13th Gen Intel Core i9-13900H processor with 20 threads (10 cores, 2 threads per core), on a computer with 6GB of available RAM. 

**Divergence in Goodness of Fit Measures** <br>
Figure **XX** Test KGE values against Test MSE values accross Huc12 sub-watershed units.  The two goodness of fit measures diverge significantly for many of the Huc12 units, especially for regions dominated by ephemeral snow. In hydrology, KGE is typically considered the more relevant goodness of fit meausure, however, it is difficult to use directly as a loss function because it is not easily differentiable.  Nonetheless, figures **X** and *XX** highlight the imprecision introduced by using MSE as a loss function if the ultimate goal is to produce high values of KGE.  We briefly expiremented with using KGE, or a hybring KGE+MSE loss function during training but observed impractical training times and chaotic results, so did not further pursue this avenue at this time.  Nontheless, investigation into the best loss function strategy is a ripe area for future research. 



## Figure1
Map of Huc12 Units Used in this Expirement, by Snow Type. 

**Legend** <br>
 - Yellow - Maritime Snow Predominates (155)
 - Orange - Montane Forest Snow Predominates (187)
 - Blue - Ephemeral Snow Predominates (180)
-  Red - Prairie Snow Predominates (11)
-  White - Boreal Forest (1)

![Map of Huc12 Units Tested - by Snow Type](https://github.com/DSHydro/SnowML/blob/252b8399f385c7bb212a1f9f3c0dd62b57d67174/notebooks/Ex2_VarianceByHuc/charts/TrainingHucMapBySnowType.png)


## Figure2

**Figure 2A - Goodness of Fit Measures By Snow Type**
| Test KGE | Test MSE |
|----------|----------|
| ![Test KGE By Predominant Snow Type - Excludes Ephemeral](https://github.com/DSHydro/SnowML/blob/094ee64af6af3a735df95ab1b9897bbf435b4007/notebooks/Ex2_VarianceByHuc/charts/Boxplot%20of%20Test%20KGE%20by%20Predominant%20Snow%20Type%20-%20Locally%20Trained%20Hucs%20-%20Excludes%20Ephemeral.png) | ![Test MSE By Predominant Snow Type - Excludes Ephemeral](https://github.com/DSHydro/SnowML/blob/6192f2ae1ef7ce6792ff17c6f1c85deb4e456cc0/notebooks/Ex2_VarianceByHuc/charts/Boxplot%20of%20Test%20MSE%20by%20Predominant%20Snow%20Type%20-%20Locally%20Trained%20Hucs.png) |
| ![Test KGE by Predominant Snow Type - Includes Ephemeral](https://github.com/DSHydro/SnowML/blob/094ee64af6af3a735df95ab1b9897bbf435b4007/notebooks/Ex2_VarianceByHuc/charts/Boxplot%20of%20Test%20KGE%20by%20Predominant%20Snow%20Type%20-%20Locally%20Trained%20Hucs.png) | ![Test MSE By Predominant Snow Type - Includes Ephemeral](https://github.com/DSHydro/SnowML/blob/094ee64af6af3a735df95ab1b9897bbf435b4007/notebooks/Ex2_VarianceByHuc/charts/Boxplot%20of%20Test%20KGE%20by%20Predominant%20Snow%20Type%20-%20Locally%20Trained%20Hucs.png) |


**Figure 2B - Goodness of Fit Measures By Mean Basin Elevation**
| KGE by Mean Basin Elevation | MSE by Mean Basin Elevation |
|----------------------------|----------------------------|
| ![Test KGE by Mean Basin Elevation](https://github.com/DSHydro/SnowML/blob/0aedec097ad929da3e7b93882af1fa0540d83206/notebooks/Ex2_VarianceByHuc/charts/Boxplot%20of%20Test%20KGE%20by%20Elevation%20Category%20-%20Locally%20Trained%20Sub-Watersheds%20_Includes%20Ephemeral%20Sub-Watersheds_.png) | ![Test MSE by Mean Basin Elevation](https://github.com/DSHydro/SnowML/blob/0aedec097ad929da3e7b93882af1fa0540d83206/notebooks/Ex2_VarianceByHuc/charts/Boxplot%20of%20Test%20MSE%20by%20Elevation%20Category%20-%20Locally%20Trained%20Sub-Watersheds%20_Includes%20Ephemeral%20Sub-Watersheds_.png) |


**Figure 2C - Goodness of Fit Measures by Snow Type and Mean Basin Elevation** 
| Test KGE by Elevation and SnowType | Test MSE by Elevation and SnowType |
|------------------------------------|------------------------------------|
| ![Test KGE by Elevation and SnowType](https://github.com/DSHydro/SnowML/blob/4319d34278c70bd2498704ad7b87d5e764fa96be/notebooks/Ex2_VarianceByHuc/charts/Boxplot%20of%20Test%20KGE%20by%20Elevation%20Category%20and%20Predominant_Snow%20-%20Locally%20Trained%20Sub-Watersheds.png) | ![Test MSE by Elevation and SnowType](https://github.com/DSHydro/SnowML/blob/4319d34278c70bd2498704ad7b87d5e764fa96be/notebooks/Ex2_VarianceByHuc/charts/Boxplot%20of%20Test%20MSE%20by%20Elevation%20Category%20and%20Predominant_Snow%20-%20Locally%20Trained%20Sub-Watersheds.png)


**Figure 2D - Goodness of FIt Measures By Basin**
| KGE By Basin | MSE By Basin |
|--------------|--------------|
| ![KGE By Basin](https://github.com/DSHydro/SnowML/blob/e167b7d1b6d78f23b2e39b0f428a400358be1bc0/notebooks/Ex2_VarianceByHuc/charts/Boxplot%20of%20Test%20KGE%20by%20Basins%20-%20Locally%20Trained%20Sub-Watersheds%20(Exludes%20Ephemeral%20Sub-Watersheds).png) | ![MSE By Basin](https://github.com/DSHydro/SnowML/blob/094ee64af6af3a735df95ab1b9897bbf435b4007/notebooks/Ex2_VarianceByHuc/charts/Boxplot%20of%20Test%20MSE%20by%20Basins%20-%20Locally%20Trained%20Sub-Watersheds%20(Excludes%20Ephemeral%20Sub-Watersheds).png)|


## Figure3 
| KGE vs. MSE - All Snow Types | KGE vs. MSE - Maritime and Montane Forest |
|------------------------------|------------------------------------------|
| ![KGE vs. MSE - All Snow Types](https://github.com/DSHydro/SnowML/blob/9b099ff3cf6c3d787694b100911481a9cdb9f3a1/notebooks/Ex2_VarianceByHuc/charts/Scatter_Plot_of_Test_KGE_vs_Test_MSE_(Colored_by_Predominant_Snow_Type).png) | ![KGE vs. MSE - Maritime and Montane Forest](https://github.com/DSHydro/SnowML/blob/9b099ff3cf6c3d787694b100911481a9cdb9f3a1/notebooks/Ex2_VarianceByHuc/charts/Test%20KGE%20vs.%20Test%20MSE%2C%20Excluding%20Hucs%20where%20Ephemeral%20Snow%20Predominates.png) |

[**TO DO** - Insert example plots of mismatched MSE/KGE]





# Limitations and Questions For Further Research
- We did not perform significant hyperparameter tuning. Future researchers may wish to investigate hyperparameter tuning using stratified cross-validation.Moreover, given the differences in model performance accross snow types and elevation, it would be worth purusing whether different snow types/elevations require different sets of optimized hyperparameters.  [**Add point about noisyness makes especially relevant to investigate finetuning**] 
- The model may benefit from additional meteorological variables, such as solar radiation, humidity, or wind speed, an aspect we investigate in Expirement 3.
- As discussed w/r/t Expirement 1, the training data SWE values are themselves a model, so any imprecisions in the training SWE values would make model results diverge from ground truth even assuming the model perfectly predicted the training SWE values.
- As noted above, using MSE as a training loss function may not optimize for the best KGE fit, but in hydrology KGE is viewed as the superior goodness of fit measure.  Using KGE itslef as a loss funciton is challenging because the measure is not easily differentiable and therefor highly computationally intensive. Training with KGE as a loss function can also lead to instabiliyt/lack of model convergence, as we observed directly with some initial experimentation using KGE or hybrid loss functions. Nonetheless, further research into using KGE as a training loss measure is warranted.  Future researchers could consider hybrid training strategies, such as using MSE as the loss function in early epochs and switching to a KGE or hybird measure only in later epochs when the model has stabalized. 

# How to Reproduce The Results
The results for this expirement were produced using the snowML.LSTM package in this repo.  The hyperparameters were set as shown in the section below. The expirement was then run by importing the module `local-training-expirement.py` and by calling the function
`run_expirement()` Note that during training data is split into batches and shuffled for randomness, so different runs of the same expirement may result in somewhat different outcomes. 


The metrics discussed above were downloaded from ML flow using [this notebook](https://github.com/DSHydro/SnowML/blob/d1653c0b190fa6e54b4473dc1d4808fe5c590e81/notebooks/Ex2_VarianceByHuc/DownloadMetrics.ipynb) and analyzed using [this notebook](https://github.com/DSHydro/SnowML/blob/24819a388afc00303ca350f9256376979f2f5298/notebooks/Ex2_VarianceByHuc/LSTM_By_Huc.ipynb). 

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

 


