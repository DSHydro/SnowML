
# # pylint: disable=C0103

# Script to run an expiriment
import importlib
from torch import optim
from torch import nn
import mlflow
from snowML.LSTM import LSTM_pre_process as pp
from snowML.LSTM import snow_LSTM as snow
from snowML.LSTM import set_hyperparams as sh


libs_to_reload = [snow, pp, sh]
for lib in libs_to_reload:
    importlib.reload(lib)


def set_inputs():
    #input_pairs = [[17020009, '12'], [17110005, '12'], [17030002, '12']]
    #input_pairs = [[17110005, '12']]
    input_pairs = [[17020009, '12']]
    return input_pairs


def prep_input_data(params):
    """
    Prepares input data for the experiment.


    Args:
        params (dict): A dictionary containing parameters for data preparation. 
                       Expected keys include var_list (list): List of variables to be used.

    Returns:
        df_dict: A dictionary where keys are HUCs and values are preprocessed dataframes.
    """
    input_pairs = set_inputs()
    hucs = pp.assemble_huc_list(input_pairs)
    df_dict = pp.pre_process(hucs, params["var_list"])
    return df_dict

def set_ML_server(params):
    """
    Configures the MLflow tracking server and sets the experiment.

    Returns:
        None
    """
    # Set our tracking server uri for logging
    mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")

    # Create a new MLflow Experiment called "LSTM"
    mlflow.set_experiment(params["expirement_name"])

def initialize_model(params):
    """
    Initializes the SnowModel with the given parameters.

    Args:
        params (dict): A dictionary containing the following keys:
            - "var_list" (list): List of variables for the input size.
            - "hidden_size" (int): The number of features in the hidden state.
            - "num_class" (int): The number of output classes.
            - "num_layers" (int): The number of recurrent layers.
            - "dropout" (float): The dropout probability.

    Returns:
        tuple: A tuple containing the initialized model, optimizer, and loss function:
            - model_dawgs (SnowModel): The initialized SnowModel.
            - optimizer_dawgs (torch.optim.Optimizer): The optimizer for the model.
            - loss_fn_dawgs (torch.nn.modules.loss._Loss): The loss function for the model.
    """
    input_size=len(params["var_list"])
    model_dawgs = snow.SnowModel(
        input_size,
        params['hidden_size'],
        params['num_class'],
        params['num_layers'],
        params['dropout']
    )
    optimizer_dawgs = optim.Adam(model_dawgs.parameters())
    loss_fn_dawgs = nn.MSELoss()
    return model_dawgs, optimizer_dawgs, loss_fn_dawgs


def run_expirement(param_dict = None):
    if param_dict is None:
        params = sh.create_hyper_dict()
    df_dict = prep_input_data(params)
    set_ML_server(params)
    model_dawgs, optimizer_dawgs, loss_fn_dawgs = initialize_model(params)

    with mlflow.start_run():
        # log all the params
        mlflow.log_params(params)

        # pre-train
        
        pre_train_epochs = int(params['n_epochs'] * params['pre_train_fraction'])
        for epoch in range(pre_train_epochs):
            print(f"Epoch {epoch}: Pre-training on multiple HUCs")

            # pre-train
            snow.pre_train(
                model_dawgs,
                optimizer_dawgs,
                loss_fn_dawgs,
                df_dict,
                params
                )

            # evaluate
            snow.evaluate(
                model_dawgs,
                df_dict,
                params,
                epoch)

        # fine_tune (if applicable)
        if params["pre_train_fraction"] < 1:
            fine_tune_epochs = int(params['n_epochs'] - pre_train_epochs)

            for epoch in range(pre_train_epochs, fine_tune_epochs):

                for target_key in df_dict.keys():
                    print(f"Fine-tuning on {target_key}")
                    snow.fine_tune(
                        model_dawgs,
                        optimizer_dawgs,
                        loss_fn_dawgs,
                        df_dict,
                        target_key,
                        params,
                        epoch)

                    snow.evaluate(
                        model_dawgs,
                        df_dict,
                        params,
                        epoch,
                        selected_keys = [target_key])

