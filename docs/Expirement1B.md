# ** Expirement 1B: Increase Data Granularity **

In expirement 1B, we increased the geographic granularity of the model. Still working in the Skagit basin, we applied the proptyped LSTM model on each sub-watershed (Huc12) 
are rather than the Huc10 areas. Huc regions are defined on a nested structure, where each Huc10 watershed includes multiple smaller, Huc 12 watershed. 
All hyperparameters are unchanged between Expirement 1A and 1B. In this expirement 1B, all sub-watersheds were trained *only* using the data from that sub-watershed,
as noted below by the paramenter (pre_train_fraction = 0).  Train/test split was accomplished by reserving the final two thirds of the time period as test data. <br>




## ProtoTyped Model - Example Results 
  - ** BASIN XX **
  -  ** BASIN YY ** 



## Test_MSE and KGE By Basin - UA Data
| HUC_ID      | Test MSE  | Test KGE  |




## Model Details: ProtoTyped Model - UA Data, Skagitt12 
| Parameter           | Value                       |


