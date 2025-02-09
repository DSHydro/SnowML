# pylint: disable=C0103

import random
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import LSTM_pre_process as pp


class SnowModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_class, num_layers, dropout):
        super(SnowModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.lstm1 = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            dropout=self.dropout,
            batch_first=True)
        self.linear = nn.Linear(hidden_size, num_class)
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x):
        device = x.device
        hidden_states = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        cell_states = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm1(x, (hidden_states, cell_states))
        out = self.linear(out[:, -1, :])
        out = self.leaky_relu(out)
        return out


# Helper Function: Load data into DataLoader
def create_dataloader(df, var_list, params):
    """ Creates a DataLoader for a given HUC dataset """
    train_data, _, _, _ = pp.train_test_split(df, params["train_size_fraction"])
    X_train, y_train = pp.create_tensor(train_data, params["lookback"], var_list)

    # Create a DataLoader
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_train, y_train),
        shuffle=True,
        batch_size=params["batch_size"],
        num_workers=params["num_workers"]  # Enable multi-threading for data loading
    )
    return loader

# Pre-training Phase: Train on multiple HUCs
def pre_train(model, optimizer, loss_fn, df_dict, target_key, var_list, params):
    """ Pre-train the model on multiple HUCs """

    # Calculate the number of pre-train epochs
    pre_train_epochs = int(params['n_epochs'] * params['pre_train_fraction'])

    # Initialize available keys for sampling without replacement
    available_keys = [key for key in df_dict.keys() if key != target_key]
    random.shuffle(available_keys)  # Shuffle for randomness

    for epoch in range(pre_train_epochs):
        model.train()  # Set model to training mode
        print(f"Epoch {epoch}: Pre-training on multiple HUCs")
        random.shuffle(available_keys)


        # Iterate over all the HUCs and train on them
        for i, selected_key in enumerate(available_keys, start=1):
            i+=1
            #if i % 5 == 0: 
            print(f"Epoch {epoch}: Training on HUC {i} {selected_key}")

            loader = create_dataloader(
                df_dict[selected_key],
                var_list,
                params,
            )


            # Training Loop
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                y_pred = model(X_batch)
                loss = loss_fn(y_pred, y_batch)
                loss.backward()
                optimizer.step()
            
            
            
        # Perform validation every 5 epochs
        df_target = df_dict[target_key]
        if epoch % 5 == 0:
            validate_model(model, loss_fn, df_target, var_list, params['lookback'])

# Fine-tuning Phase: Train on target HUC
def fine_tune(model, optimizer, loss_fn, df_dict, target_key, var_list, params):
    """ Fine-tune the model on the target HUC """

    n_epochs = int(params['n_epochs']*(1-params['pre_train_fraction']))
    df_target = df_dict[target_key]

    # Create DataLoader for fine-tuning (target HUC)
    loader = create_dataloader(
        df_target,
        var_list,
        params
        )

    for epoch in range(n_epochs):
        model.train()  # Set model to training mode
        print(f"Epoch {epoch}: Fine-tuning on target HUC {target_key}")

        # Training Loop on target HUC
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            loss.backward()
            optimizer.step()

        # Perform validation every 5 epochs
        if epoch % 5 == 0:
            validate_model(model, loss_fn, df_target, var_list, params['lookback'])


def validate_model(model, loss_fn, df_target, var_list, lookback):
    """ Performs validation on target datasets """
    model.eval()
    with torch.no_grad():
        # Evaluate on target dataset
        X_val_target, y_val_target = pp.create_tensor(df_target, lookback, var_list)
        y_pred_target = model(X_val_target)
        val_rmse_target = np.sqrt(loss_fn(y_pred_target, y_val_target))
        print(f"Validation RMSE on target dataset: {val_rmse_target:.4f}")


# End-to-End Model Training Function
def train_model(model, optimizer, loss_fn, df_dict, target_key, var_list, params):
    """Train the model using pre-train and fine-tune phases"""
    pre_train(model, optimizer, loss_fn, df_dict, target_key, var_list, params)
    fine_tune(model, optimizer, loss_fn, df_dict, target_key, var_list, params)

def predict(data, model, X_train, X_test, train_size, var_list, huc_id, params):
    data = data.astype(object)  # Keeping this line if needed for some other processing
    with torch.no_grad():
        # Initialize empty arrays with NaNs for plotting
        train_plot = np.full_like(data['mean_swe'].values, np.nan, dtype=float)
        test_plot = np.full_like(data['mean_swe'].values, np.nan, dtype=float)

        # Ensure X_train and X_test are tensors
        if isinstance(X_train, np.ndarray):
            X_train = torch.tensor(X_train, dtype=torch.float32)
        if isinstance(X_test, np.ndarray):
            X_test = torch.tensor(X_test, dtype=torch.float32)

        # Predict on training data
        y_pred = model(X_train)
        y_pred_new = y_pred[:, -1].unsqueeze(1)

        # Convert tensor to numpy safely
        train_plot[params["lookback"]:train_size] = y_pred_new.detach().cpu().numpy().flatten()

        # Predict on test data
        y_pred_test = model(X_test)[:, -1].unsqueeze(1)
        test_plot[train_size + params["lookback"] : len(data)] = y_pred_test.detach().cpu().numpy().flatten()

    # plot
    plt.figure(figsize=(12,  6))
    plt.plot(data.index, data['mean_swe'], c='b', label='Actual')
    plt.plot(data.index, train_plot, c='r', label='Train Predictions')
    plt.plot(
        data.index[train_size+params["lookback"]:],
        test_plot[train_size+params["lookback"]:],
        c='g',
        label='Test Predictions')
    plt.legend()
    plt.xlabel('Date')
    plt.ylabel('swe')
    # TO DO - DEAL WITH THE CLASSIFICATION VARS ALSO
    var_list_string= "_".join(var.split("_", 1)[1] for var in var_list[0:3]) + "snow_class_vars"
    plt.title(f'SWE_Predictions_for_huc{huc_id}_using_vars_{var_list_string}_and_self_only_is{params["self_only"]}')
    plt.savefig("predict_plot.png")
    #mlflow.log_figure(plt.gcf(), tit+".png")
    #plt.show()

def evaluate_metrics(model, X_train, y_train, X_test, y_test):

    with torch.no_grad():
        y_train_pred = model(X_train)
        y_test_pred = model(X_test)

        #var_list_string= "_".join(var.split("_", 1)[1] for var in var_list)

        train_mse = mean_squared_error(y_train.numpy(), y_train_pred.numpy())
        #mlflow.log_metric(f"train_mse_{str(huc)}_self_only_is{self_only}", train_mse, step=step_value)
        test_mse = mean_squared_error(y_test.numpy(), y_test_pred.numpy())
        #mlflow.log_metric(f"test_mse_{str(huc)}_self_only_is{self_only}", test_mse, step=step_value)

    return  [train_mse, test_mse]
