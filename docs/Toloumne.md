This document discusses results from local training LSTM model on Huc12 sub-watersheds in [Tuolumne valley](basin_fact_sheets/Tuolumne.md).  Huc12 watershed units where epheneral snow dominates were excluded, resulting in 22 sub-watersheds for analysis: 12 Maritime, 5 Prairie, 4 Montane Forest, and 1 Tundra.   

[Add map] 

Each Huc12 was trained with a simple LSTM model, using precipitation and air temperature as the feature inputs, and the University of Arizona re-analysis data for swe estimates. Median KGE was .88, similar to what was observed in the non-ephemeral sub-watersheds in Washington State discussed [here](Ex2_VariationByHuc.md) Charts/results for each of the 22 watersheds available [here](../notebooks/Toloumne/charts/Local_Training_Results).

An example of a well-performing sub-watershed: 
![Good Example](../notebooks/Toloumne/charts/Local_Training_Results/UA_Results_and_Lidar_for_huc_180400090105_w_UCLA_dat.png)

An example of a watershed with more moderate performance: 
![Mid Example](../notebooks/Toloumne/charts/Local_Training_Results/UA_Results_and_Lidar_for_huc_180400090401_w_UCLA_dat.png)

Boxplot of performance against UA data in test period accross all 22 sub-watershed, by snow-type 

![](../notebooks/Toloumne/charts/BoxPlot_of_Test_KGE__Locally_Trained_Models_in_Toloumne.png)

Boxplot of performance againes lidar data where available (19 watersheds) by snow-type
![](../notebooks/Toloumne/charts/BoxPlot_of_KGE_v_Lidar_Locally_Trained_Models_in_Toulumne.png)


