# SnowML
UW MSDS SnowML capstone project


## Data 
Read about our [data pipleine](docs/Data_Pipeline.md)

## Development

### Installation Instructions
Clone the repo and then do the following:
`cd SnowML`
`pip install -e .`

### MLflow
Follow [installation instructions](https://mlflow.org/docs/latest/getting-started/tracking-server-overview/index.html#minute-tracking-server-overview) or quickly install and run below:
1. Activiate a virtual environment through conda or pip first
2. `pip install mlflow`
3. Open a terminal, `cd` to a directory where you model will run, and then type the command `mlfow ui` and press entire.
4. Keep the termial alive, open a browser, and navigate to `localhost:5000` and press enter. This should direct to the MLflow enterface where all experiments and things live.
 
## New Teammates

### Where to Begin?
the **data** folder contains our outputs from loss functions. The **docs** folder contains the most information, check out Experiments 1-3 and the data-pipeline file for an in-depth description of our data pipeline. The **notebooks** file contains the initial model provided by our sponsor and our LSTM model jupyternotebook scripts. 

### Task Directory
Take a look at the task directory for ideas on where to take this project and interesting ideas to include in an academic poster. Sort the directory by completion status, type of project, and document your progress.
