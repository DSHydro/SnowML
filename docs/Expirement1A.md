# ** Expirement 1A: Add more time series data****

The first adjustment we made to the proptoyped LSTM Model was to use the University of Arizona estimates of Snow Water Equivilent (SWE) [data]( https://climate.arizona.edu/data/UA_SWE/) as our target dataset for training and evaluating the model.  This dataset contains a longer time series of available SWE data than used in the protoyped model. 

We reran the protoype LSTM model with the new data, leaving all other hyperparmeters unchanged except that reduced the number of epochs to 10 after observing early convergence of the model likely due to the increased training data available. The results below compare the prior data run at 200 epochs with the new data run at 10 epochs, but the observations are robust to the number of epochs used (10 vs. 200) in both models.  Please refer to the [Viz10COmpare notebook](notebooks/Prototype_Model_Results/VizHuc10Compare.ipynb) for sensitivy analysis related to number of epochs used.  

The below charts show model results for HUC 1711000504, a watershed in the Skagitt basin, under the two approaches. The Prototyped Model with UA Data has a test_mean_squared_error of .007 which is a significnat improvement over [].  These differences appeared stable over multiple runs of the respective models.  

Improvements were also seen in each of the eight watersheds tested.  Interestingly, however, for both datasets, the model performed better in some watersheds than other. Again, these differences appeared stable over multiple runs of the respective model. We further explore differences in model performance by watershed is explored in later expirements. 

In this expirement 1A, all watersheds were trained *only* using the data from that sub-watershed,as noted below by the paramenter (pre_train_fraction = 0).  Train/test split was accomplished by reserving the final two thirds of the time period as test data. 

## ProtoTyped Model - UA Data 

![notebooks/charts/UAData_SWE_Post_Predictions_for_huc_1711000504.png]

## ProtoTyped Model - Original Data 
![notebooks/Prototype_Model_Results/charts/SWE_Post_Predictions_for_huc_1711000504.png]

## Test_MSE and KGE By Basin - UA Data
| HUC_ID      | Test MSE  | Test KGE  |
|------------|----------|----------|
| 1711000504 | 0.007089 | 0.923201 |
| 1711000505 | 0.004568 | 0.977445 |
| 1711000506 | 0.026291 | 0.701809 |
| 1711000507 | 0.015769 | 0.818750 |
| 1711000508 | 0.029625 | 0.644258 |
| 1711000509 | 0.010326 | 0.806219 |
| 1711000510 | 0.035046 | 0.621083 |
| 1711000511 | 0.003720 | 0.900734 |



## Model Details: ProtoTyped Model - UA Data 
| Parameter           | Value                       |
|---------------------|-----------------------------|
| hidden_size         | 64                          |
| num_class           | 1                           |
| num_layers          | 1                           |
| dropout             | 0.5                         |
| learning_rate       | 0.001                       |
| n_epochs            | 10                          |
| pre_train_fraction  | 0                           |
| train_size_fraction | 0.67                        |
| lookback            | 180                         |
| batch_size          | 16                          |
| n_steps             | 1                           |
| num_workers         | 8                           |
| var_list            | ['mean_pr', 'mean_tair']    |
| experiment_name     | BackToBasics_Skagitt        |
| run_mame            | likeable-vole-19            |
| run_id              | b1649da4415449c49ad0841fd230d950|


