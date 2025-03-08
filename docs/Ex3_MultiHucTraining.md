# Expirement 3: Multi-Huc Training 

In Expirement 3, we considered whether model results could be generalized for use in ungauged basins.  We also expiremented with different variable combinations and training rates. 

For training, we focused on Huc08 sub-Basins where Maritime or Montane Forest snow predominated, including:  <br>
 - Maritme Basins: [Chelan](basin_fact_sheets/Chelan(17020009).md) (17020009), [Sauk](basin_fact_sheets/Sauk(17110006).md) (17110006), [Skagit](basin_fact_sheets/Skagit(17110005).md) (17110005), [Skykomish](basin_fact_sheets/Skykomish(1711009).md) (17110009), and [Wenatche](basin_fact_sheets/Wenatche(17020011).md) (17020011)
 - Montane Forest Basins: [Middle Salmon Chamberlain](basin_fact_sheets/Middle_Salmon-Chamberlain(17060207).md)(17060207), [St.Joe](basin_fact_sheets/St._Joe(17010304).md) (17010304), [South Fork Coeur d'ALene](basin_fact_sheets/South_Fork_Coeur_d'Alene(17010302).md) (17010302), [South Fork Salmon River](basin_fact_sheets/South_Fork_Salmon_River(17060208).md)(17060208) and [Upper Couer d'Alene](basin_fact_sheets/Upper_Coeur_d'Alene(17010301).md) (17010301)

Within these regions, we treated each Huc12 sub-unit as a  separate time series of data.  We excluded Huc12 sub-watersheds dominated by ephemral snow as we are primiarly interested in modelling persistent snow pack, and we learned from expirement 2 that ephemeral snow is less well modelled by our LSTM model.  This selection process reulted in 270 Huc12 units available for training.

We randomly split these 270 Huc12 sub-watersheds into training, validaton, and test groups using a 
60/20/20 split, resulting in 162 Huc12 sub-watersheds used in model training, 54 in validation, and 54 in Test Set A. Its worth noting that train/validation/test splits resulting from random selection resulted in a validation set that was somewhat overweighted in Motane Forest sub-watersheds (65%) compared to the training (52%) and test sets(56%).

We also defined Test set B as the Huc12 units in  [Upper Yakima](basin_fact_sheets/UpperYakima(17030001).md) (17030001) and [Naches](basin_fact_sheets/Naches(17030002).md) (17030002), which contain a mix of Maritime and Montane Forest snowpack, again excluding any Huc12 sub-watershed units predominated by ephemeral snow. The Yakima/Naches region is ecologically important in the Pacific Northwest and for farming in the Yakima valley. These Huc08 sub-Basins play the role of completely ungauged regions in our expirement set up. 

In this expirement, we explored eight model different variations, consisting of four different variabe combinations, each run using a learning rate of .001 and .0003. 
| Name                | Features                                      |
|---------------------|----------------------------------------------|
| Base Model         | Precipitation, Temperature, Mean Elevation   |
| Base plus Srad     | Precipitation, Temperature, Elevation, Solar Radiation |
| Base plus WindSpeed | Precipitation, Temperature, Elevation, Wind Speed |
| Base plus Humidity | Precipitation, Temperature, Elevation, Humidity |


For each of 8 different variable and learning rate combinations, we trained the model using the multiple Huc12 units in the training set, for 30 epochs, assessing performance against the validation set at the end of each epoch, and logging a separate model in mlflow at the end of each epoch.  After all runs were complete, we selected the "best" model by examining the median test_kge accross all the validation set hucs and choosing the model that resulted in the maximum value of median_test_kge. Finally, we used the selected model to predict results on both Test Set A and Test Set B. 


# Observations and Results

## Impact Of Variable Selection and Learning Rate

Figure 2 plots the results of each of the eight models by epoch, in terms of the "Median Test_Kge" observed accross the validation set of 54 Huc12 units for that epoch.  The results are relatively noisy. The main impact of variable selection and learning rate appears to be on the stability of the model over epochs.  All eight models reach a maximum value of median kge within a relatively tight range, from .78 (Solar Radiation, .0001 learning rate, epoch 9) to .82 (Humidity, .0003 learning rate, epoch 27). All models, except one, spend several epochs oscilatting up and down, before dropping precipitously in later epochs, likely due to overfitting in later epochs.  The exception is the Base Model plus Humidity, at the .0003 learning rate, which remains relatively stable throughout all 30 epochs. More generally, the modele using the lower .0003 learning rate appear noisier during early epohcs than the models run at the larger, .001 learning rate, contrary to what might be expected.  

## The LSTM Model Generalizes  
The results of the multi-trained model on ungauged basins is, on average, on par with the results of the locally trained model. Figure 3 plots the Test KGE values for each of the Huc12 subunits in both Test Set A and Test Set B, as well as providing summary statistics for several goodness of fit measures incuding Test KGE, Test MSE, and Test R2.  For Test Set A, the median Test KGE value was .85 with an interquartile range of (0.63 to 0.90). Test Set B had a median Test KGE value of .82 with an interquartile range of (.70, .88).  

These distribution of test_kge values within Test Sets A and B is similar to the distribution of Test KGE values obtained in Expirement 2 for locally trained Huc12 subunits dominated by Maritime and Montane Forest snow.  In Expirement 2 we observed a median KGE and interquartile range of ___ among Maritime HUc12 subunits and a median KGE of __ and interquartile range of ___.  However, the two expirements are not directly comprable.  The second expirement introduced the enw variables of mean_humidity and mean_elevation (although mean elevation would not be relevant for a locally trained huc since it would always be a constant). Expirement 3 also used a smaller batch size as the introduction of new variables required smaller batches to stay within the limits of the computing power available to us.  An interesting area for future research will be to compare the performacne of particular Huc12 sub-watersheds using (a) a locally trained model; (b) a multi-huc model model with local fine tuning, and (c) a multi-huc model with no local fine tuning, using a stable set of variables and hyperparameters.  





## Figure 1 - Map of Training, Validation, and Test Sets 

| Training Data | Validation Data |
|--------------|----------------|
| ![MapOfTrainingData](https://github.com/DSHydro/SnowML/blob/408f037565594e8bddcce673dc55bb12909509ed/notebooks/Ex3_MultiHucTraining/charts/TrainSet.png) | ![MapOfValidationData](https://github.com/DSHydro/SnowML/blob/408f037565594e8bddcce673dc55bb12909509ed/notebooks/Ex3_MultiHucTraining/charts/ValSet.png) |

| Test Set A | Test Set B |
|------------|------------|
| ![MapOfTestSetA](https://github.com/DSHydro/SnowML/blob/408f037565594e8bddcce673dc55bb12909509ed/notebooks/Ex3_MultiHucTraining/charts/TestSetA.png) | ![MapOfTestSetB](https://github.com/DSHydro/SnowML/blob/408f037565594e8bddcce673dc55bb12909509ed/notebooks/Ex3_MultiHucTraining/charts/TestSetB.png) |




## Figure 2 - Comparison of Model Performance by Variable, Learning Rate, and Epoch 
| ![All Variables, .0003 Learning Rate](https://github.com/DSHydro/SnowML/blob/main/notebooks/Ex3_MultiHucTraining/charts/ModelChoice/Median_KGE_Comparison_By_Variable_Combo_Using_Learning_Rate_0.0003.png) | ![All Variables, .001 Learning Rate ](https://github.com/DSHydro/SnowML/blob/main/notebooks/Ex3_MultiHucTraining/charts/ModelChoice/Median_KGE_Comparison_By_Variable_Combo_Using_Learning_Rate_0.001.png) |
|---|---|



# Figure 3- Results on Ungauged Huc Units 
| Test Set A | Naches and Upper Yakima |
|------------|-------------------------|
| ![Test Set A](https://github.com/DSHydro/SnowML/blob/main/notebooks/Ex3_MultiHucTraining/charts/Historgram%20of%20Test%20KGE%20Values%20Among%20Test%20SetA.png) | ![Naches and Upper Yakima](https://github.com/DSHydro/SnowML/blob/main/notebooks/Ex3_MultiHucTraining/charts/Historgram%20of%20Test%20KGE%20For%20HUC12%20Sub-Watersheds%20In%20Test%20Set%20B.png) |

**Summary Statistics Test Set A**
|       | Test MSE  | Test KGE  | Test R2  |
|-------|----------|----------|----------|
| count | 54.000000 | 54.000000 | 54.000000 |
| mean  | 0.007387  | 0.723357  | 0.820370  |
| std   | 0.012596  | 0.311120  | 0.372874  |
| min   | 0.000522  | -0.614623 | -1.173180 |
| 25%   | 0.001691  | 0.635164  | 0.846439  |
| 50%   | 0.004524  | 0.845155  | 0.931609  |
| 75%   | 0.007354  | 0.900350  | 0.957855  |
| max   | 0.087530  | 0.982700  | 0.986104  |

**Summary Statistics Test Set B**
|       | Test MSE  | Test KGE  | Test R2  |
|-------|----------|----------|----------|
| count | 48.000000 | 48.000000 | 48.000000 |
| mean  | 0.007014  | 0.789205  | 0.887679  |
| std   | 0.008653  | 0.145063  | 0.076185  |
| min   | 0.000222  | 0.265396  | 0.550655  |
| 25%   | 0.002085  | 0.706568  | 0.863799  |
| 50%   | 0.004246  | 0.821192  | 0.918686  |
| 75%   | 0.008932  | 0.883383  | 0.934607  |
| max   | 0.050682  | 0.969418  | 0.963966  |





# Figure 4 - Test Results on UnGauged Huc Units - Test Set B 

| ![Plot of Predicted vs. Actual SWE - Example of High Test KGE (90% percentile)](https://github.com/DSHydro/SnowML/blob/cdbdff61a60b5cb9df9935d9a1b0de16823d940e/docs/model_results/SWE_Predictions_for_huc170300010402.png) |
| ![Plot of Predicted vs. Actual SWE - Example Near Median Test KGE Value](https://github.com/DSHydro/SnowML/blob/cdbdff61a60b5cb9df9935d9a1b0de16823d940e/docs/model_results/SWE_Predictions_for_huc17030020106.png) |
| ![Plot of Predicted vs. Actual SWE - Example of Low Test KGE Value (10% percentile)](https://github.com/DSHydro/SnowML/blob/cdbdff61a60b5cb9df9935d9a1b0de16823d940e/docs/model_results/SWE_Predictions_for_huc170300010701.png) |



## Limitations and Questions For Further Research
- The eight model variations explored in this expirement barely scratch the surface in terms of the potential for tuning hyperparameters and investigating variable selection. - Future researchers may wish to pursue further tuning.  However, given lengthy training tiems for each model run, computational intensive methods, such as grid search among hyperparameters may not be practical. 
- An interesting area for future research will be to compare the performacne of particular Huc12 sub-watersheds using (a) a locally trained model; (b) a multi-huc model model with local fine tuning, and (c) a multi-huc model with no local fine tuning, using a stable set of variables and hyperparameters.  


## How to Reproduce The Results
The results for this expirement were produced using the snowML.LSTM package in this repo. The hyperparameters were set as shown in the section below. The training/validation/huc splits are also recorded below. The expirement was then run by importing the module multi-huc-expirement.py and by calling the function run_expirement(train_hucs, val_hucs, test_hucs) Note that during training data is split into batches and shuffled for randomness, so different runs of the same expirement may result in somewhat different outcomes.


[** TO BE INSERTED **]

