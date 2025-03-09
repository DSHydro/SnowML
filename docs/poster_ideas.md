# Ungauged Basin?  No Problem! 

![High Creek-Naneum Creek](https://github.com/DSHydro/SnowML/blob/main/docs/model_results/SWE_Predictions_for_huc170300010402%20using%20Predict%20From%20PreTrained.png)

- Our multi-huc trained LSTM model predicts ungauged basins with accuracy comparable to the locally trained model
- The main drawback: longer training times. For predictions future SWE on gauged basins, use our lightweight local training model! 
- Graph above shows results from the High Creek-Naneum Creek sub-watershed near Yakima valley, an important watershed for Washington Agriculture.
- Of course, not all of our results are **quite** this good. Our goodness of fit statistics for the Naches and Upper Yakima regions:
