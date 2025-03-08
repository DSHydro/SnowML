## Expirement 3: Multi-Huc Training 

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


## Observations and Results

**Impact Of Variable Selection and Learning Rate** <br>
Figure 2 plots the results of each of the eight models by epoch, in terms of the "Median Test_Kge" observed accross the validation set of 54 Huc12 units for that epoch.  The results are relatively noisy. The main impact of variable selection and learning rate appears to be on the stability of the model over epochs.  All eight models reach a maximum value of median kge within a relatively tight range, from .78 (Solar Radiation, .0001 learning rate, epoch 9) to .82 (Humidity, .0003 learning rate, epoch 27). All models, except one, spend several epochs oscilatting up and down, before dropping precipitously in later epochs, likely due to overfitting in later epochs.  The exception is the Base Model plus Humidity, at the .0003 learning rate, which remains relatively stable throughout all 30 epochs. More generally, the modele using the lower .0003 learning rate appear noisier during early epohcs than the models run at the larger, .001 learning rate, contrary to what might be expected.  

** The Model Trained on Multiple Hucs Generalizes Relatively Well To Ungauged Basins ** <br>





## Figure 1 - Map of Training, Validation, and Test Sets ##

| Training Data | Validation Data |
|--------------|----------------|
| ![MapOfTrainingData](https://github.com/DSHydro/SnowML/blob/408f037565594e8bddcce673dc55bb12909509ed/notebooks/Ex3_MultiHucTraining/charts/TrainSet.png) | ![MapOfValidationData](https://github.com/DSHydro/SnowML/blob/408f037565594e8bddcce673dc55bb12909509ed/notebooks/Ex3_MultiHucTraining/charts/ValSet.png) |

| Test Set A | Test Set B |
|------------|------------|
| ![MapOfTestSetA](https://github.com/DSHydro/SnowML/blob/408f037565594e8bddcce673dc55bb12909509ed/notebooks/Ex3_MultiHucTraining/charts/TestSetA.png) | ![MapOfTestSetB](https://github.com/DSHydro/SnowML/blob/408f037565594e8bddcce673dc55bb12909509ed/notebooks/Ex3_MultiHucTraining/charts/TestSetB.png) |




## Figure 2 - Comparison of Model Performance by Variable, Learning Rate, and Epoch ##
| ![All Variables, .0003 Learning Rate](https://github.com/DSHydro/SnowML/blob/main/notebooks/Ex3_MultiHucTraining/charts/ModelChoice/Median_KGE_Comparison_By_Variable_Combo_Using_Learning_Rate_0.0003.png) | ![All Variables, .001 Learning Rate ](https://github.com/DSHydro/SnowML/blob/main/notebooks/Ex3_MultiHucTraining/charts/ModelChoice/Median_KGE_Comparison_By_Variable_Combo_Using_Learning_Rate_0.001.png) |
|---|---|


## Figure 3- Results on Ungauged Huc Units - Test Set A 
| ![Histogram of Median Test KGE in Naches and Upper Yakima](https://github.com/DSHydro/SnowML/blob/4ebc428282ec3eff50d5c7395bf80eab1adb47c7/notebooks/Ex3_MultiHucTraining/charts/histogram.png) 


## Figure 4 - Test Results on UnGauged Huc Units - Test Set B 

| ![Plot of Predicted vs. Actual SWE - Example of High Test KGE (90% percentile)](https://github.com/DSHydro/SnowML/blob/cdbdff61a60b5cb9df9935d9a1b0de16823d940e/docs/model_results/SWE_Predictions_for_huc170300010402.png) |
| ![Plot of Predicted vs. Actual SWE - Example Near Median Test KGE Value](https://github.com/DSHydro/SnowML/blob/cdbdff61a60b5cb9df9935d9a1b0de16823d940e/docs/model_results/SWE_Predictions_for_huc17030020106.png) |
| ![Plot of Predicted vs. Actual SWE - Example of Low Test KGE Value (10% percentile)](https://github.com/DSHydro/SnowML/blob/cdbdff61a60b5cb9df9935d9a1b0de16823d940e/docs/model_results/SWE_Predictions_for_huc170300010701.png) |



## Limitations and Questions For Further Research
- The eight model variations explored in this expirement barely scratch the surface in terms of the potential for tuning hyperparameters and investigating variable selection. - Future researchers may wish to pursue further tuning.  However, given lengthy training tiems for each model run, computational intensive methods, such as grid search among hyperparameters may not be practical. 
-


[** TO BE INSERTED **]

## How to Reproduce The Results
The results for this expirement were produced using the snowML.LSTM package in this repo. The hyperparameters were set as shown in the section below. The training/validation/huc splits are also recorded below. The expirement was then run by importing the module multi-huc-expirement.py and by calling the function run_expirement(train_hucs, val_hucs, test_hucs) Note that during training data is split into batches and shuffled for randomness, so different runs of the same expirement may result in somewhat different outcomes.


[** TO BE INSERTED **]

