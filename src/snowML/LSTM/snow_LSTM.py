import torch
from torch import nn
import random
import numpy as np
import LSTM_pre_process as pp
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class SnowModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_class, num_layers, dropout):
        super(SnowModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers, dropout=self.dropout, batch_first=True)
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

def train_model(model, optimizer, loss_fn, df_dict, target_key, n_epochs, \
                batch_size, lookback, var_list, train_size_fraction, self_only = True):

    if self_only:
        print("training_will_proceed_with_this_huc_only")
    else: 
        print("training will proceed with multi_hucs until the final 10% of epochs")
        
    # Initialize available keys for sampling without replacement
    available_keys = [key for key in df_dict.keys() if key != target_key]
    random.shuffle(available_keys)  # Shuffle for randomness
    used_keys = []  # Tracks used keys

    # Prepare validation data from df_target
    df_target = df_dict[target_key]
    train_target, _, _, _ = pp.train_test_split(df_target, train_size_fraction)
    X_val_target, y_val_target = pp.create_tensor(df_target, lookback, var_list)

    for epoch in range(n_epochs):
        
        # Determine which dataset to use for training
        if epoch < 0.9 * n_epochs and not self_only:
            # If available_keys is empty, reset it
            if not available_keys:
                available_keys = used_keys  # Refill from used keys
                random.shuffle(available_keys)  # Reshuffle
                used_keys = []  # Clear used list

            # Sample without replacement
            selected_key = available_keys.pop()
            used_keys.append(selected_key)  # Track used key
            if not self_only: 
                print(f"Epoch {epoch}: Training data used = {selected_key}")
                
            # prep data 
            data = df_dict[selected_key]
            train_main, _, _, _ = pp.train_test_split(data, train_size_fraction)
            X_train, y_train = pp.create_tensor(train_main, lookback, var_list)

        else:
            # Use df_target 
            X_train, y_train = X_val_target, y_val_target
            selected_key = target_key
            

        # Create DataLoader
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_train, y_train), shuffle=True, batch_size=batch_size
        )

        # Training Loop
        model.train()
        for X_batch, y_batch in loader:
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Perform validation every 5 epochs
        if epoch % 5 == 0:
            model.eval()
            with torch.no_grad():
                # Evaluate on target dataset
                y_pred_target = model(X_val_target)
                val_rmse_target = np.sqrt(loss_fn(y_pred_target, y_val_target))

                # Evaluate on current training dataset
                y_pred_train = model(X_train)
                val_rmse_train = np.sqrt(loss_fn(y_pred_train, y_train))

                print(f"Epoch {epoch}: Validation RMSE on target dataset ({target_key}): {val_rmse_target:.4f}")
                print(f"Epoch {epoch}: Validation RMSE on current training dataset ({selected_key}): {val_rmse_train:.4f}")

    return target_key

def predict(data, model, X_train,X_test, lookback, train_size, var_list, huc_id, self_only):
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
        train_plot[lookback:train_size] = y_pred_new.detach().cpu().numpy().flatten()

        # Predict on test data
        y_pred_test = model(X_test)[:, -1].unsqueeze(1)
        test_plot[train_size + lookback : len(data)] = y_pred_test.detach().cpu().numpy().flatten()

    # plot
    plt.figure(figsize=(12,  6))
    plt.plot(data.index, data['mean_swe'], c='b', label='Actual')
    plt.plot(data.index, train_plot, c='r', label='Train Predictions')
    plt.plot(data.index[train_size+lookback:], test_plot[train_size+lookback:], c='g', label='Test Predictions')
    plt.legend()
    plt.xlabel('Date')
    plt.ylabel('swe')
    var_list_string= "_".join(var.split("_", 1)[1] for var in var_list[0:3]) + "snow_class_vars"  # TO DO - DEAL WITH THE CLASSIFICATION VARS ALSO
    tit = f'SWE_Predictions_for_huc{huc_id}_using_vars_{var_list_string}_and_self_only_is{self_only}'
    plt.title(tit)
    f_out = "predict_plot.png"
    plt.savefig(f_out) 
    #mlflow.log_figure(plt.gcf(), tit+".png")
    #plt.show()

def evaluate_metrics(model, X_train, y_train, X_test, y_test, huc, step_value, self_only):

    with torch.no_grad():
        y_train_pred = model(X_train)
        y_test_pred = model(X_test)

        #var_list_string= "_".join(var.split("_", 1)[1] for var in var_list)
        
        train_mse = mean_squared_error(y_train.numpy(), y_train_pred.numpy())
        #mlflow.log_metric(f"train_mse_{str(huc)}_self_only_is{self_only}", train_mse, step=step_value)
        test_mse = mean_squared_error(y_test.numpy(), y_test_pred.numpy())
        #mlflow.log_metric(f"test_mse_{str(huc)}_self_only_is{self_only}", test_mse, step=step_value)
        
    return  [train_mse, test_mse]