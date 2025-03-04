## Expirement 3: Multi-Huc Training 

In Expirement 3, we considered whether model results could be generalized for use in ungauged basins.  We also expiremented with different variable combinations and trainging rates. 

For training, we focused on Huc08 basins where Maritime or Montane Forest snow predominated, including:  <br>
 - Maritme Basins: [Chelan](basin_fact_sheets/Chelan(17020009).md) (17020009), [Sauk](basin_fact_sheets/Sauk(17110006).md) (17110006), [Skagit](basin_fact_sheets/Skagit(17110005).md) (17110005), [Skykomish](basin_fact_sheets/Skykomish(1711009).md) (17110009, and [Wenatche](basin_fact_sheets/Wenatche(17020011).md) (17020011)
 - Montane Forest Basins: [Middle Salmon Chamberlain](basin_fact_sheets/Middle_Salmon-Chamberlain(17060207).md)(17060207), [St.Joe](basin_fact_sheets/St._Joe(17010304).md) (17010304), [South Fork Coeur d'ALene](basin_fact_sheets/South_Fork_Coeur_d'Alene(17010302).md) (17010302), [South Fork Salmon River](basin_fact_sheets/South_Fork_Salmon_River(17060208).md)(17060208) and [Upper Couer d'Alene](basin_fact_sheets/Upper_Coeur_d'Alene(17010301).md) (17010301)

Within each Huc8 region, we treated each Huc12 sub-unit as a  separate time series of data.  We excluded Huc12 subwatershed dominated by ephemral snow as we are primiarly interested in modelling persistent snow pack.  This resulted in **XX* Huc12 units available for training.

We randomly split these *XX* Huc12 sub-watersheds into training, validaton, and test groups using a 
60/20/20 split, resulting in **XX** Huc12 sub-watersheds were used in model training, **XX** in validation, and **XX** in Test Set A. Its worth noting that train/validation/test splits resulting from random selection resulted in a validation set that was somewhat overweighted in Motane Forest sub-watershed (

We also defined Test set B as the Huc12 units in  [Upper Yakima](basin_fact_sheets/UpperYakima(17030001).md) (17030001) and [Naches](basin_fact_sheets/Naches(17030002).md) (17030002), which contain a mix of Maritime and Montane Forest snowpack.  The Yakima/Naches region is **blah blah important** and plays the role of a completely ungauged basin in our expirement set up. 

For each of 8 different variable and learning rate combinations, we trained the model using the multiple basins in the training set, for 30 epochs, assessing performance against the validation set at the end of each epoch, and logging a separate model in mlflow at the end of each epoch.  After all runs were complete, we selected the "best" model by examining the median test_kge accross all the validation set hucs and choosing the model that resulted in the maximum value of median_test_kge. Finally, we used the selected model to predict results on both Test Set A and Test Set B. 



[** TO BE INSERTED **]

## Observations and Results
Several interesting observations result from this expirement: [** TO BE INSERTED **]

##Figure1##
[** TO BE INSERTED **]

Figure2
Limitations and Questions For Further Research

[** TO BE INSERTED **]

How to Reproduce The Results
The results for this expirement were produced using the snowML.LSTM package in this repo. The hyperparameters were set as shown in the section below. The training/validation/huc splits are also recorded below. The expirement was then run by importing the module multi-huc-expirement.py and by calling the function run_expirement(train_hucs, val_hucs, test_hucs) Note that during training data is split into batches and shuffled for randomness, so different runs of the same expirement may result in somewhat different outcomes.


[** TO BE INSERTED **]

