# pylint: disable=C0103

import random
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import mlflow
from sklearn.metrics import mean_squared_error
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
def create_dataloader(df, params):
    """ Creates a DataLoader for a given HUC dataset """
    train_data, _, _, _ = pp.train_test_split(df, params["train_size_fraction"])
    X_train, y_train = pp.create_tensor(train_data, params["lookback"], params["var_list"])

    # Create a DataLoader
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_train, y_train),
        shuffle=True,
        batch_size=params["batch_size"],
        num_workers=params["num_workers"]  # Enable multi-threading for data loading
    )
    return loader

# Pre-training Phase: Train on multiple HUCs
def pre_train(model, optimizer, loss_fn, df_dict, params):
    """ Pre-train the model on multiple HUCs """

    # Calculate the number of pre-train epochs
    pre_train_epochs = int(params['n_epochs'] * params['pre_train_fraction'])

    # Initialize available keys for sampling without replacement
    available_keys = [key for key in df_dict.keys()]
    random.shuffle(available_keys)  # Shuffle for randomness

    for epoch in range(pre_train_epochs):
        model.train()  # Set model to training mode
        print(f"Epoch {epoch}: Pre-training on multiple HUCs")
        random.shuffle(available_keys)


        # Iterate over all the HUCs and train on them
        for i, selected_key in enumerate(available_keys, start=1):
            if i % 5 == 0:
                print(f"Epoch {epoch}: Pre-training on HUC{i}: {selected_key}")

            loader = create_dataloader(
                df_dict[selected_key],
                params
            )

            # Training Loop
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                y_pred = model(X_batch)
                loss = loss_fn(y_pred, y_batch)
                loss.backward()
                optimizer.step()


        # Perform validation every 5 epochs; use the most recent selected key
        df_validation = df_dict[selected_key]
        if epoch % 5 == 0:
            validate_model(model, loss_fn, df_validation, params["var_list"], params['lookback'])

# Fine-tuning Phase: Train on target HUC
def fine_tune(model, optimizer, loss_fn, df_dict, target_key, params):
    """ Fine-tune the model on the target HUC """

    n_epochs = int(params['n_epochs']*(1-params['pre_train_fraction']))
    df_target = df_dict[target_key]

    # Create DataLoader for fine-tuning (target HUC)
    loader = create_dataloader(
        df_target,
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
            validate_model(model, loss_fn, df_target, params["var_list"], params['lookback'])


def validate_model(model, loss_fn, df_target, var_list, lookback):
    """ Performs validation on target datasets """
    model.eval()
    with torch.no_grad():
        # Evaluate on target dataset
        X_val_target, y_val_target = pp.create_tensor(df_target, lookback, var_list)
        y_pred_target = model(X_val_target)
        val_rmse_target = np.sqrt(loss_fn(y_pred_target, y_val_target))
        print(f"Validation RMSE on most recently trained huc: {val_rmse_target:.4f}")

def predict(data, model, X_train, X_test, train_size, huc_id, params):
    data = data.astype(object)
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
    ttle = f"SWE_Predictions_for_huc{huc_id} using Baseline Model" # TO DO: Make dynamic
    plt.title(ttle)
    #plt.savefig(f"{ttle}.png")
    mlflow.log_figure(plt.gcf(), ttle+".png")
    plt.close()

def kling_gupta_efficiency(y_true, y_pred):
    r = np.corrcoef(y_true.ravel(), y_pred.ravel())[0, 1] # Correlation coefficient
    alpha = np.std(y_pred) / np.std(y_true)  # Variability ratio
    beta = np.mean(y_pred) / np.mean(y_true)  # Bias ratio
    kg = 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
    #print(f"r: {r}, alpha: {alpha}, beta: {beta}")
    return kg, r, alpha, beta

def evaluate_metrics(model, X_train, y_train, X_test, y_test, target_key, step_value):

    with torch.no_grad():
        y_train_pred = model(X_train).cpu().numpy()
        y_test_pred = model(X_test).cpu().numpy()
        y_train_pred = model(X_train).numpy()
        y_test_pred = model(X_test).numpy()
        y_train_true = y_train.numpy()
        y_test_true = y_test.numpy()

        # check for NaNs
        # for arr, name in [(y_train_pred, "y_train_pred"),
        #                           (y_test_pred, "y_test_pred"),
        #                           (y_train_true, "y_train_true"),
        #                           (y_test_true, "y_test_true")]:
        #             print(f"{name}: NaNs: {np.isnan(arr).any()}, Infs: {np.isinf(arr).any()}")


        # Compute MSE
        train_mse = mean_squared_error(y_train_true, y_train_pred)
        test_mse = mean_squared_error(y_test_true, y_test_pred)

        # Compute KGE
        train_kge, _, _, _ = kling_gupta_efficiency(y_train_true, y_train_pred)
        test_kge, _, _, _ = kling_gupta_efficiency(y_test_true, y_test_pred)

        # Log metrics
        mlflow.log_metric(f"train_mse_{str(target_key)}", train_mse, step=step_value)
        mlflow.log_metric(f"test_mse_{str(target_key)}", test_mse, step=step_value)
        mlflow.log_metric(f"train_kge_{str(target_key)}", train_kge, step=step_value)
        mlflow.log_metric(f"test_kge_{str(target_key)}", test_kge, step=step_value)

        # Print metrics
        print(f"train_mse_{str(target_key)}: {train_mse}")
        print(f"test_mse_{str(target_key)}: {test_mse}")
        print(f"train_kge_{str(target_key)}: {train_kge}")
        print(f"test_kge_{str(target_key)}: {test_kge}")

        return (train_mse, test_mse, train_kge, test_kge)
    