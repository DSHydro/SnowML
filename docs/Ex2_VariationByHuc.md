# Expirement 2: Investigate Model Performance Accross Hucs 

[** TO BE INSERTED **] 


# Observations and Results 

[** TO BE INSERTED **] 

## Figure1


## Figure2




# Limitations and Questions For Further Research
[** TO BE INSERTED **]

# How to Reproduce The Results
The results for this expirement were produced using the snowML.LSTM package in this repo.  The hyperparameters were set as shown in the section below. 
The training/validation/huc splits are also recorded below.  The expirement was then run by importing the module `multi-huc-expirement.py` and by calling the function
`run_expirement(train_hucs, val_hucs, test_hucs)` Note that during training data is split into batches and shuffled for randomness, so different runs of the same expirement may result in somewhat different outcomes. 

The results were gathered over three MLflow expirements(using the same parameters), each on a subset of the total hucs used, to make run_times manageable.


The metrics discussed above were downloaded from ML flow using [this notebook](https://github.com/DSHydro/SnowML/blob/d1653c0b190fa6e54b4473dc1d4808fe5c590e81/notebooks/Ex2_VarianceByHuc/DownloadMetrics.ipynb) and analyzed using [this notebook](https://github.com/DSHydro/SnowML/blob/24819a388afc00303ca350f9256376979f2f5298/notebooks/Ex2_VarianceByHuc/LSTM_By_Huc.ipynb). 

## Model Parameters

| Parameter             | Value                        |
|-----------------------|----------------------------|
| hidden_size          | 64                           |
| num_class           | 1                            |
| num_layers         | 1                            |
| dropout            | 0.5                          |
| learning_rate      | 0.001                        |
| n_epochs           | 10                           |
| lookback           | 180                          |
| batch_size         | 64                           |
| n_steps            | 1                            |
| num_workers        | 8                            |
| var_list           | ['mean_pr', 'mean_tair']     |
| expirement_name    | Single All                   |
| loss_type          | mse                          |
| mse_lambda         | 1                            |
| train_size_dimension | time                        |
| train_size_fraction | 0.67                         |
| MLFlow Expirement   | Single ALl                  |
| ML Flow Run IDs|  69d929bbdfdd43b4a2f45b823d945eb7, b0fa8180481a4dc6954f138e9ee934aa, aca5e4d1b42044a2aeabc055a0b14d8d|

 


