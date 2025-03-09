# Ungauged Basin?  No Problem! 

![High Creek-Naneum Creek](https://github.com/DSHydro/SnowML/blob/main/docs/model_results/SWE_Predictions_for_huc170300010402%20using%20Predict%20From%20PreTrained.png)

- Our multi-huc trained LSTM model predicts ungauged basins with accuracy comparable to the locally trained model
- The main drawback: longer training times. For forecasting SWE on gauged basins, use our lightweight local training model! 
- Resuts above reflect the High Creek-Naneum Creek sub-watershed near Yakima valley.
- Not all results are quite this good, but we acheived respectable goodness of fit statistics throughout the Naches and Upper Yakima sub-basins, an import region for Washington Agriculture: 

|       | Test MSE  | Test KGE  | Test R2  |
|-------|----------|----------|----------|
| count | 48 | 48| 48|
| mean  | 0.007  | 0.789 | 0.888  |
| 25%   | 0.002  | 0.707  | 0.864 |
| median   | 0.004 | 0.821| 0.919  |
| 75%   | 0.009  | 0.883  | 0.935  |
