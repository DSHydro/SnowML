# ** Expirement 1: Add more time series data****

The first adjustment we made to the proptoyped LSTM Model was to use the University of Arizona [add link] estimates of Snow Water Equivilent (SWE) as our
target dataset for training and evaluating the model.  This dataset contains a longer time series of available SWE data, thus augmenting the dataset.  This
change resulted in immediate improvement to the model.<br>

We reran the protoype LSTM model with the new data, leaving all other hyperparmeters unchanged except as follows: 
- Batch size increased from 8 to 16 to take advantage of increaased computing resources made available (thank you AWS!)
- We reduced the number of epochs to 10 after observing early convergence of the model likely due to the increased training data available.

The below charts show model results for HUC 1711000504, a sub-watershed in the Skagitt basin, under the two approaches.  

