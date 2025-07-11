# Expirement 3: Multi-Watershed Training 

In Expirement 3, we considered whether model results could be generalized for use in ungauged basins.  We also expiremented with different variable combinations and training rates. 

For training, we focused on Huc08 sub-Basins in the Pacific Northwest where Maritime or Montane Forest snow predominated, including:  <br>
 - Maritme Basins: [Chelan](basin_fact_sheets/Chelan(17020009).md) (17020009), [Sauk](basin_fact_sheets/Sauk(17110006).md) (17110006), [Skagit](basin_fact_sheets/Skagit(17110005).md) (17110005), [Skykomish](basin_fact_sheets/Skykomish(1711009).md) (17110009), and [Wenatchee](basin_fact_sheets/Wenatche(17020011).md) (17020011)
 - Montane Forest Basins: [Middle Salmon Chamberlain](basin_fact_sheets/Middle_Salmon-Chamberlain(17060207).md)(17060207), [St.Joe](basin_fact_sheets/St._Joe(17010304).md) (17010304), [South Fork Coeur d'ALene](basin_fact_sheets/South_Fork_Coeur_d'Alene(17010302).md) (17010302), and [South Fork Salmon River](basin_fact_sheets/South_Fork_Salmon_River(17060208).md)(17060208) 

Within these regions, we treated each Huc12 sub-watershed unit as a separate time series of data.  We excluded Huc12 sub-watersheds dominated by ephemeral snow as we are primiarly interested in modelling persistent snow pack, and we learned from expirement 2 that ephemeral snow is less well modelled by our LSTM model.  This selection process reulted in 270 Huc12 units available for training.

We randomly split these 270 Huc12 sub-watersheds into training, validaton, and test groups using a 
60/20/20 split, resulting in 162 Huc12 sub-watersheds used in model training, 54 in validation, and 54 in Test Set A. Its worth noting that train/validation/test splits resulting from random selection resulted in a validation set that was somewhat overweighted in Motane Forest sub-watersheds (65%) compared to the training (52%) and test sets(56%).

We also defined Test set B as the Huc12 units in  [Upper Yakima](basin_fact_sheets/UpperYakima(17030001).md) (17030001) and [Naches](basin_fact_sheets/Naches(17030002).md) (17030002), which contain a mix of Maritime and Montane Forest snowpack, again excluding any Huc12 sub-watershed units predominated by ephemeral snow. The Yakima/Naches region is ecologically important in the Pacific Northwest and for farming in the Yakima valley. These regions play the role of completely ungauged regions in our expirement set up. 

In this expirement, we explored eight model different variations, consisting of four different variabe combinations, each run using a learning rate of .001 and .0003. 
| Name                | Features                                      |
|---------------------|----------------------------------------------|
| Base Model         | Precipitation, Temperature, Mean Elevation   |
| Base plus Srad     | Precipitation, Temperature, Elevation, Solar Radiation |
| Base plus WindSpeed | Precipitation, Temperature, Elevation, Wind Speed |
| Base plus Humidity | Precipitation, Temperature, Elevation, Humidity |


For each of 8 different variable and learning rate combinations, we trained the model using the multiple Huc12 units in the training set, for 30 epochs, assessing performance against the validation set at the end of each epoch, and logging a separate model in mlflow at the end of each epoch.  After all runs were complete, we selected the "best" model by examining the median test_kge accross all the validation set hucs and choosing the model that resulted in the maximum value of median_test_kge. Finally, we used the selected model to predict results on both Test Set A and Test Set B. 


# Observations and Results

## Impact of Variable Selection and Learning Rate

Figure 2 plots the results of each of the eight models by epoch, in terms of the "Median Test_Kge" observed accross the validation set of 54 Huc12 units for that epoch.  The results are relatively noisy. The main impact of variable selection and learning rate appears to be on the stability of the model over epochs.  All eight models reach a maximum value of median kge within a relatively tight range, from .78 (Solar Radiation, .0001 learning rate, epoch 9) to .82 (Humidity, .0003 learning rate, epoch 27). All models, except one, spend several epochs oscilatting up and down, before dropping precipitously in later epochs, likely due to overfitting in later epochs.  The exception is the Base Model plus Humidity, at the .0003 learning rate, which remains relatively stable throughout all 30 epochs. More generally, the modele using the lower .0003 learning rate appear noisier during early epohcs than the models run at the larger, .001 learning rate, contrary to what might be expected.  

## Model Generalization  
The LSTM model generalized reasonably well to untrained basins.  Figure 3 plots the Test KGE values for each of the Huc12 subunits in both Test Set A and Test Set B, as well as providing summary statistics for several goodness of fit measures incuding Test KGE, Test MSE, and Test R2.  For Test Set A, the median Test KGE value was .85 with an interquartile range of (0.63 to 0.90). Test Set B had a median Test KGE value of .82 with an interquartile range of (.70, .88). These distributions are slightly below the Test KGE values obtained in Expirement 2 for locally trained Huc12 subunits dominated by Maritime and Montane Forest snow.  In Expirement 2 we observed a median KGE of (.85) and interquartile range of (.76, .91) among Maritime HUc12 subunits and a median KGE of .91 and interquartile range of (.85, .95). As discussed below, however, results from Expirement 2 and 3 are not directly comprable.     

Figure 4 plots actual and predicted SWE from selected Huc12 regions in Test Set B spanning some of the worst results (Test KGE of .61, or 10% percentile) to some of the best (Test KGE of .94, or 90% percentile).  

## Differences In Model Performance By Snow Type 
In this Expiriment 3, there were not statistically significant differences in Test KGE performance between Maritime and Montane Forest Snow Classes, but significant differences in Test MSE were observed between the two classes. Figure 5 plots the results when Test Sets A and B are evaluated as combined test set.  Similar relationships hold when Test Set A and Test Set B are evaluated individually.  

## Difference in Model Perfrmance By Elevation 
With training on multiple Huc12 units and the explicite inclusion of mean elevation in the feature set, we no longer see a significant correlation bewteen model perforamance and mean elevation. Figure 6 plots Test KGE agains Mean Elevation for the combined Test Sets A and B.  Similar results were observed when examining the relationship between mean elevation and Test KGE on the individual Test Sets A and B.  

# Tables and Figures 

## Figure 1 - Map of Training, Validation, and Test Sets 

| Training Data | Validation Data |
|--------------|----------------|
| ![MapOfTrainingData](https://github.com/DSHydro/SnowML/blob/main/notebooks/Ex3_MultiHucTraining/charts/TrainSet.png) | ![MapOfValidationData](https://github.com/DSHydro/SnowML/blob/main/notebooks/Ex3_MultiHucTraining/charts/ValSet.png) |

| Test Set A | Test Set B (Naches and Upper Yakima) |
|------------|------------|
| ![MapOfTestSetA](https://github.com/DSHydro/SnowML/blob/main/notebooks/Ex3_MultiHucTraining/charts/TestSetA.png) | ![MapOfTestSetB](https://github.com/DSHydro/SnowML/blob/main/notebooks/Ex3_MultiHucTraining/charts/TestSetB.png) |




## Figure 2 - Comparison of Model Performance by Variable, Learning Rate, and Epoch 
| ![All Variables, .0003 Learning Rate](https://github.com/DSHydro/SnowML/blob/main/notebooks/Ex3_MultiHucTraining/charts/ModelChoice/Median_KGE_Comparison_By_Variable_Combo_Using_Learning_Rate_0.0003.png) | ![All Variables, .001 Learning Rate ](https://github.com/DSHydro/SnowML/blob/main/notebooks/Ex3_MultiHucTraining/charts/ModelChoice/Median_KGE_Comparison_By_Variable_Combo_Using_Learning_Rate_0.001.png) |
|---|---|



## Figure 3- Results on Independent HUc12 Sub-Watersheds
| Test Set A | Naches and Upper Yakima |
|------------|-------------------------|
| ![Test Set A](https://github.com/DSHydro/SnowML/blob/main/notebooks/Ex3_MultiHucTraining/charts/Historgram%20of%20Test%20KGE%20Values%20Among%20Test%20SetA.png) | ![Naches and Upper Yakima](https://github.com/DSHydro/SnowML/blob/main/notebooks/Ex3_MultiHucTraining/charts/Historgram%20of%20Test%20KGE%20For%20HUC12%20Sub-Watersheds%20In%20Test%20Set%20B.png) |

**Summary Statistics Test Set A**
|       | Test MSE  | Test KGE  | Test R2  |
|-------|----------|----------|----------|
| count | 54.000000 | 54.000000 | 54.000000 |
| mean  | 0.007387  | 0.723357  | 0.820370  |
| std   | 0.012596  | 0.311120  | 0.372874  |
| min   | 0.000522  | -0.614623 | -1.173180 |
| 25%   | 0.001691  | 0.635164  | 0.846439  |
| 50%   | 0.004524  | 0.845155  | 0.931609  |
| 75%   | 0.007354  | 0.900350  | 0.957855  |
| max   | 0.087530  | 0.982700  | 0.986104  |

**Summary Statistics Test Set B**
|       | Test MSE  | Test KGE  | Test R2  |
|-------|----------|----------|----------|
| count | 48.000000 | 48.000000 | 48.000000 |
| mean  | 0.007014  | 0.789205  | 0.887679  |
| std   | 0.008653  | 0.145063  | 0.076185  |
| min   | 0.000222  | 0.265396  | 0.550655  |
| 25%   | 0.002085  | 0.706568  | 0.863799  |
| 50%   | 0.004246  | 0.821192  | 0.918686  |
| 75%   | 0.008932  | 0.883383  | 0.934607  |
| max   | 0.050682  | 0.969418  | 0.963966  |



## Figure 4 - Example Prediction SWE vs. Actual SWE Plots 
Example prediction plots for Huc12 Units demonstrating s range of Test_KGE Values from Test Set B (Naches and Yakima).  Results shown range from from a low Test KGE of .61 (10% percentile within Test Set B) to a high of .94, (90% percentile).  

 ![Plot of Predicted vs. Actual SWE - Example of High Test KGE (90% percentile)](https://github.com/DSHydro/SnowML/blob/cdbdff61a60b5cb9df9935d9a1b0de16823d940e/docs/model_results/SWE_Predictions_for_huc170300010402.png) 
 ![Plot of Predicted vs. Actual SWE - Example Near Median Test KGE Value](https://github.com/DSHydro/SnowML/blob/cdbdff61a60b5cb9df9935d9a1b0de16823d940e/docs/model_results/SWE_Predictions_for_huc17030020106.png) 
![Plot of Predicted vs. Actual SWE - Example of Low Test KGE Value (10% percentile)](https://github.com/DSHydro/SnowML/blob/cdbdff61a60b5cb9df9935d9a1b0de16823d940e/docs/model_results/SWE_Predictions_for_huc170300010701.png) 

## Figure 5 
| KGE By Snow Type | MSE By Snow Type |
|----------------|----------------|
| ![KGE By Snow Type](https://github.com/DSHydro/SnowML/blob/f3751557a3b76c1f557363d3196ce197c533b243/notebooks/Ex3_MultiHucTraining/charts/Boxplot%20of%20Test%20KGE%20by%20Predominant%20Snow%20Type%20-%20MultiHucModel-CombinedTestSets.png) | ![MSE By Snow Type](https://github.com/DSHydro/SnowML/blob/main/notebooks/Ex3_MultiHucTraining/charts/Boxplot%20of%20Test%20MSE%20by%20Predominant%20Snow%20Type%20-%20MultiHucModel%20-%20CombinedTestSets.png) |
| Testing for Inequality of Mean Test KGE By Snow Type (Welch Test), p-value = .356 | Testing for Inequality of Mean Test MSE by Snow Type (Welch Test), p-value <.001 |








## Figure 6 - Model Performance By Mean Elevation  
![Model Performance By Mean Elevation](https://github.com/DSHydro/SnowML/blob/main/notebooks/Ex3_MultiHucTraining/charts/Test%20KGE_vs_mean_elevation_MultiHucModel_CombinedTestSets.png)

*Plot of Test KGE for each Huc12 sub-watershed plotted against mean elevation for that sub-watershed. Unlike expirement 2, in this expirement 3 there is no statistically significant relationship between mean elevation and Test KGE*




# Limitations and Questions for Further Research
- The eight model variations explored in this expirement barely scratch the surface in terms of the potential for tuning hyperparameters and investigating variable selection. Future researchers may wish to pursue further tuning.  However, given lengthy training tiems for each model run, computational intensive methods, such as grid search among hyperparameters may not be practical.
- Results between Expirement 2 and Expirement 3 are not directly comprable.  The second expirement introduced the enw variables of mean_humidity and mean_elevation (although mean elevation would not be relevant for a locally trained huc since it would always be a constant). Expirement 3 also used a smaller batch size as the introduction of new variables required smaller batches to stay within the limits of the computing power available to us.  An interesting area for future research will be to compare the performance of particular Huc12 sub-watersheds using (a) a locally trained model; (b) a multi-huc model model with local fine tuning, and (c) a multi-huc model with no local fine tuning, using a stable set of variables and hyperparameters. 
- The issues discussed with respect to Expirements 1 and 2 related to choosing the appropriate loss function, and potential errors in the training SWE set, remain relevant to Expirement 3.  


# How to Reproduce The Results
The results for this expirement were produced using the snowML package in this repo, using the steps below.  Note that during training data is split into batches and shuffled for randomness, so different runs of the same expirement may result in somewhat different outcomes.




1. **Set Up Your Run-Time Environment.** You will need an IDE that can execute python scripts and a terminal for bash commands, as well as an mlflow tracking server.  We recommend Sagemaker Studio which enables you to insantiate both an mlflow server and a Code Spaces IDE from within the Studio.  Take note of the tracking_uri for the mlflow server that you set up, as you'll need it in step 5. If working from Sagemaker Studio, the mlflow tracking_uri should look something like this: "arn:aws:sagemaker:us-west-2:677276086662:mlflow-tracking-server/dawgsML."

2.  **Clone the SnowML Repo and Install SnowML package.**
```
bash
git clone https://github.com/DSHydro/SnowML.git 
cd SnowML # make sure to switch into the snowML directory and run all subsequent code from there
pip install . #installs the SnowML package
```

3.  **Set up Earth Engine Credentials**  Go to https://developers.google.com/earth-engine/guides/auth to set up earth engine credentials and initialize a project.   Then run the following code: 
```
#python
import ee
ee.Authenticate(auth_mode = "notebook")
```
After successful authentication, Initialize your project.  Replace ee-frostydawgs with your project name you created above. 
```
ee.Initialize(project="ee-frostydawgs") # replace with your project name
```

4.  **Ensure Access To Model Ready Data**  The code in this github repo assumes you have access to the frosty-dawgs S3 buckets discussed in our [data pipeline notebook](https://github.com/DSHydro/SnowML/blob/main/notebooks/DataPipe.ipynb). If instead you are using your own model-ready data, plesae update the ```snowML.datapipe set_data_constants``` module to correctly point to the S3 buckets where your data is stored.   


5. **Create a dictionary called "params".** From within python, create a dictionary of "params" with the desired values of the relevant hyperparamenters (the values used in each run are shown in the table below). This can be acheived by updating the module ```snowML.LSTM.set_hyperparams``` in the snow.LSTM package or manually such as with the function below and updating the desired values. 

```
# python
def create_hyper_dict():
    param_dict = {
        "hidden_size": 2**6,
        "num_class": 1,
        "num_layers": 1,
        "dropout": 0.5,
        "learning_rate": 3e-4,  # 3e-3
        "n_epochs": 30,
        "lookback": 180,
        "batch_size": 32,
        "n_steps": 1,
        "num_workers": 8,
        "var_list": ["mean_pr", "mean_tair", "mean_hum", "Mean Elevation"],
        "expirement_name": "Multi_All-2",
        "loss_type": "mse",
        "mse_lambda": 1, 
        "train_size_dimension": "huc",
        "train_size_fraction": 1, 
        "mlflow_tracking_uri": "arn:aws:sagemaker:us-west-2:677276086662:mlflow-tracking-server/dawgsML"
    }
    return param_dict

params = create_hyper_dict()
```

6. **Define the hucs that will be used in the training, validation, and huc sets.**  To resuse the same hucs as discussed here, run the code below.  This code results in four lists of huc numbers, corresponding to the train, validation, and test sets A and B.  If you run this code from within an AWS enviornment, you may see warning messages about unclosed aiohttp connectors.  These are harmless.  (But annoying!  Please, help us out with a pull request if you know how to suppress, we've tried everything . . . )

```
from snowML.Scripts.load_hucs import load_huc_splits as lh
from snowML.Scripts.load_hucs import create_test_set_B as cb
tr, val, test_A = lh.huc_split()
test_B = cb.get_testB_huc_list()
```

7. **Train the model.**  Train the model, evaluating the results on the validation test set at the end of each epoch.   This will take a while!  The runs described in this expirement each took between 20-30 hours.

```
from snowML.Scripts import multi_huc_expirement as mhe
mhe.run_expirement(tr, val, params)  
```

**Repeat steps 3-5 for every combination of hyperparameters you wish to examine**

8. **Select Model To Use For Test Evaluation.**  To determine which model you want to use for testing, you'll want to examine the metrics logged in mlflow for each model run.  This can be done directly in the mlflow ui, or you can download metrics using the ```download_metrics``` module (discussed below) from the ```snowML.Scripts``` package and analyze the metrics offline as we did in this notebook [Choose Best Model](https://github.com/DSHydro/SnowML/blob/main/notebooks/Ex3_MultiHucTraining/Choose_Best_Model.ipynb).
  
9. **Get the identifiers for the chosen model.**  Once you have identified the model you want to test against, locate the model run_id from the MLflow server, and the model_uri for the model that corresponds to the epoch you want to use from that run. The model used for the metrics on this page was from epoch 27 using a learning rate of 3e-4 and feature variables temperature, precipitation, humidity, anbasin elevation. You'll also need your mlflow_tracking_uri again.  

```
model_uri = "s3://sues-test/298/51884b406ec545ec96763d9eefd38c36/artifacts/epoch27_model" # update with model you will use
run_id = "d71b47a8db534a059578162b9a8808b7" # update with run-id you will use 
mlflow_tracking_uri = "arn:aws:sagemaker:us-west-2:677276086662:mlflow-tracking-server/dawgsML" # update with your tracking_uri
```

10. **Run a new mlflow expirement to log the test metrics.**
```
from snowML.LSTM import LSTM_evaluate as eval
eval.predict_from_pretrain(test_B, run_id, model_uri, mlflow_tracking_uri)
```

11. **Download the test metrics from the mlflow server**
```
from snowML.Scripts import download_metrics as dm
run_dict = <new_run_id>  # insert the run_id for the run created in step 8 here
run_dict = {"Test_B": run_id} 
dm.download_all(run_dict, folder ="mlflow_data/run_id_data")  # update folder to your desired loca location
```
This will create a file called "metrics_from_{run_id}.csv" into the designated folder.  

12.  **Analyze Away!**  From here, we performed analytics in Jupyter Notebooks, with some helper scripts from the SnowML package.  Please refer to the notebooks [Assemble Metrics](https://github.com/DSHydro/SnowML/blob/main/notebooks/Ex3_MultiHucTraining/LSTM_By_Huc_Metric_Download_TestMetrics.ipynb), [Test Set A](https://github.com/DSHydro/SnowML/blob/main/notebooks/Ex3_MultiHucTraining/TestSetA.ipynb), [Test Set B](https://github.com/DSHydro/SnowML/blob/main/notebooks/Ex3_MultiHucTraining/TestSetB.ipynb) and [Combined Test Set](https://github.com/DSHydro/SnowML/blob/main/notebooks/Ex3_MultiHucTraining/TestSetA_and_B.ipynb) for details.  






# HyperParameters
| Parameter              | Base Model 1e-3 | Base Model 3e-4 | Base Plus Wind Speed ('vs') 1e-3 | Base Plus Wind Speed ('vs') 3e-4 | Base Plus Solar Radiation ('srad') 1e-3 | Base Plus Solar Radiation ('srad') 3e-4 | Base Plus Humidity 1e-3 | Base Plus Humidity 3e-4 |
|------------------------|----------------|----------------|--------------------------------|--------------------------------|--------------------------------------|--------------------------------------|----------------------|----------------------|
| hidden_size           | 64             | 64             | 64                             | 64                             | 64                                   | 64                                   | 64                   | 64                   |
| num_class            | 1              | 1              | 1                              | 1                              | 1                                    | 1                                    | 1                    | 1                    |
| num_layers           | 1              | 1              | 1                              | 1                              | 1                                    | 1                                    | 1                    | 1                    |
| dropout             | 0.5            | 0.5            | 0.5                            | 0.5                            | 0.5                                  | 0.5                                  | 0.5                  | 0.5                  |
| **learning_rate**       | **0.001**          | **0.0003**         | **0.001**                          | **0.0003**                         | **0.001**                                | **0.0003**                                | **0.001**                | **0.0003**                |
| n_epochs            | 30             | 30             | 30                             | 30                             | 30                                   | 30                                   | 30                   | 30                   |
| lookback            | 180            | 180            | 180                            | 180                            | 180                                  | 180                                  | 180                  | 180                  |
| batch_size          | 32             | 32             | 32                             | 32                             | 32                                   | 32                                   | 32                   | 32                   |
| n_steps             | 1              | 1              | 1                              | 1                              | 1                                    | 1                                    | 1                    | 1                    |
| num_workers         | 8              | 8              | 8                              | 8                              | 8                                    | 8                                    | 8                    | 8                    |
| **var_list**        | **['mean_pr', 'mean_tair', 'Mean Elevation']** | **['mean_pr', 'mean_tair', 'Mean Elevation']** | **['mean_pr', 'mean_tair', 'mean_vs', 'Mean Elevation']** | **['mean_pr', 'mean_tair', 'mean_vs', 'Mean Elevation']** | **['mean_pr', 'mean_tair', 'mean_srad', 'Mean Elevation']** | **['mean_pr', 'mean_tair', 'mean_srad', 'Mean Elevation']** | **['mean_pr', 'mean_tair', 'mean_hum', 'Mean Elevation']** | **['mean_pr', 'mean_tair', 'mean_hum', 'Mean Elevation']** |
| experiment_name      | Multi_All-2    | Multi_All-2    | Multi_All-2                    | Multi_All-2                    | Multi_All-2                          | Multi_All-2                          | Multi_All-2            | Multi_All-2            |
| loss_type           | mse            | mse            | mse                            | mse                            | mse                                  | mse                                  | mse                  | mse                  |
| mse_lambda          | 1              | 1              | 1                              | 1                              | 1                                    | 1                                    | 1                    | 1                    |
| train_size_dimension | huc            | huc            | huc                            | huc                            | huc                                  | huc                                  | huc                  | huc                  |
| train_size_fraction | 1              | 1              | 1                              | 1                              | 1                                    | 1                                    | 1                    | 1                    |
| **run_id** | **a6c611d4c4cf410e9666796e3a8892b7** | **e989030c272d4de59c84aff739d8063c** | **4653005687094d9ba54c295b943a4667** | **bc031cafad7445adb73173adc43b63c6** | **deed782fda71472fb47cf8670b668473** | **2b49d6cce3844ede8a66821ae9aec27b** | **d71b47a8db534a059578162b9a8808b7** | **51884b406ec545ec96763d9eefd38c36** |
| **run_name** | **debonair_dove** | **spiffy whale** | **puzzled cow** | **placid-croc** | **enchanting-roo** | **judicious_mare** | **peaceful stork** | **capricious snipe** |


