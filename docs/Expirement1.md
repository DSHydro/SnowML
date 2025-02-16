# ** Expirement 1: Add more time series data****

The first adjustment we made to the proptoyped LSTM Model was to use the University of Arizona [add link] estimates of Snow Water Equivilent (SWE) as our
target dataset for training and evaluating the model.  This dataset contains a longer time series of available SWE data, thus augmenting the dataset.  This
change resulted in immediate improvement to the model.<br>

We reran the protoype LSTM model with the new data, leaving all other hyperparmeters unchanged except as follows: 
- Batch size increased from 8 to 16 to take advantage of increaased computing resources made available (thank you AWS!)
- We reduced the number of epochs to 10 after observing early convergence of the model likely due to the increased training data available.

The below charts show model results for HUC 1711000504, a sub-watershed in the Skagitt basin, under the two approaches. The Prototyped Model with UA Data has a test_mean_squared_error of .007 which is a significnat improvement over [].  These differences appeared stable over multiple runs of the respective models.  Improvements were also seen in each of the [five] sub-watersheds tested.  

In this expirement, all sub-watersheds were trained *only* using the data from that sub-watershed,as noted below by the paramenter (pre_train_fraction = 0).  

## ProtoTyped Model - Original Data 


## ProtoTyped Model - UA Data 

![SWE Predictions](docs/model_results/SWE_Predictions_for_huc1711000504.png)


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


