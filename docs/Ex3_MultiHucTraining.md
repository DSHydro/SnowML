Expirement 3: Multi-Huc Training 

In Expirement 3, we considered whether model results could be generalized and unsed in ungauged basins.  We also expiremented with different variable combinations and trainging rates. 

We focused on Huc12 sub-watershed units where Maritime or Montane Forest snow prrdominated.  We withheld from the training data all the Huc12 watersheds in the *XX** basin as a test set ("Test Set A"). The *XX* basin is **blah blah important** and plays the role of a completely ungauged basin in our expirement set up. 

From the reminaing *XX* Huc12 units with Maritime or Montane Foreests snow available for training, we split these into training, validaton, and test ("Test Set B") groups using a 
60/20/20 split, resulting in **XX** Huc12 sub-watersheds were used in model training, **XX** in validation, and **XX** in test set B.  

For each of 8 different variable and learning rate combinations, we trained the model using the multiple basins in the training set, for 30 epochs, assessing performance against the validation set at the end of each epoch, and logging a separate model in mlflow at the end of each epoch.  After all runs were complete, we selected the "best" model by examining the median test_kge accross all the validation set hucs and choosing the model that resulted in the maximum value of median_test_kge. Finally, we used the selected model to predict results on both Test Set A and Test Set B. 



[** TO BE INSERTED **]

Observations and Results
Several interesting observations result from this expirement: [** TO BE INSERTED **]

Figure1
[** TO BE INSERTED **]

Figure2
Limitations and Questions For Further Research

[** TO BE INSERTED **]

How to Reproduce The Results
The results for this expirement were produced using the snowML.LSTM package in this repo. The hyperparameters were set as shown in the section below. The training/validation/huc splits are also recorded below. The expirement was then run by importing the module multi-huc-expirement.py and by calling the function run_expirement(train_hucs, val_hucs, test_hucs) Note that during training data is split into batches and shuffled for randomness, so different runs of the same expirement may result in somewhat different outcomes.


[** TO BE INSERTED **]

