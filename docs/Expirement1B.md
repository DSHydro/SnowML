# ** Expirement 1B: Increase Data Granularity **

In expirement 1B, we increased the geographic granularity of the model. Still working in the Skagit basin, we applied the proptyped LSTM model on each sub-watershed (Huc12) 
are rather than the Huc10 areas. Huc regions are defined on a nested structure, where each Huc10 watershed includes multiple smaller, Huc 12 watershed. 
All hyperparameters are unchanged between Expirement 1A and 1B. In this expirement 1B, all sub-watersheds were trained *only* using the data from that sub-watershed,
as noted below by the paramenter (pre_train_fraction = 0).  Train/test split was accomplished by reserving the final two thirds of the time period as test data. <br>




## ProtoTyped Model - Example Results 
  - ** BASIN XX **
  -  ** BASIN YY ** 



## Average Test_MSE and KGE By Watershed - UA Data Modelled at Huc(12)
| HUC_ID       | Average Test MSE | Average Test KGE |
|--------------|------------------|------------------|
| 1711000504   | 0.005147         | 0.924250         |
| 1711000505   | 0.006710         | 0.931421         |
| 1711000506   | 0.026137         | 0.832154         |
| 1711000507   | 0.021886         | 0.836513         |
| 1711000508   | 0.019096         | 0.814897         |
| 1711000509   | 0.013426         | 0.741316         |
| 1711000510   | 0.041787         | 0.599238         |
| 1711000511   | 0.012682         | 0.655579         |

## Test_MSE and KGE By Watershed - UA Data Modelled at Huc(10)




## Model Details: ProtoTyped Model - UA Data, Skagitt12 
| Parameter           | Value                       |


