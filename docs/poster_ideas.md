# Ungauged Basin?  No Problem! 


- Our LSTM model trained on multiple Huc12 regions can predict totally ungauged basins with accuracy comprable to the locally trained model
- The drawback of this approach is that training times are much longer, however, so for gauged basins, predicting with a locally trained model is the way to go!
- The above graph shows the results from the [] a basin in the [] region near Yakima valley, an important watershed for Washington Agriculture.
- Of course, not all of our results are **quite** this good.  When testing on ungauged basins in the Upper Yakim and Naches sub-basins, our median Test KGE was [],
- with an interquartile range of []; corresponding to R-squared values of [](median) and interquartile range ().  THe above graph is toward the upper end of our results. 
