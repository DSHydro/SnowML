""" Recreate Old Model File Name for Reloading"""

import torch
from torch import nn


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
        """
        Performs a forward pass through the LSTM model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, input_size),
                            representing a batch of sequences.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size), representing 
                      the model's predictions after passing through the LSTM and 
                      fully connected layers with a LeakyReLU activation.
    """

        device = x.device
        hidden_states = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        cell_states = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm1(x, (hidden_states, cell_states))
        out = self.linear(out[:, -1, :])
        out = self.leaky_relu(out)
        return out

class HybridLoss(nn.Module):
    def __init__(self, initial_lambda=1.0, final_lambda=1.0, total_epochs=30, eps=1e-6):
        """
        Hybrid loss combining -KGE and MSE.
        :param initial_lambda: Initial weight for MSE loss (higher at the start).
        :param final_lambda: Final weight for MSE loss (lower in later epochs).
        :param total_epochs: Total training epochs for lambda scheduling.
        :param eps: Small constant for numerical stability.
        """
        super(HybridLoss, self).__init__()
        self.initial_lambda = initial_lambda
        self.final_lambda = final_lambda
        self.total_epochs = total_epochs
        self.current_epoch = 0
        self.eps = eps
        self.mse_loss = nn.MSELoss()

    def set_epoch(self, epoch):
        """Update lambda dynamically per epoch."""
        self.current_epoch = epoch
        progress = epoch / self.total_epochs
        self.lambda_mse = self.initial_lambda * (1 - progress) + self.final_lambda * progress

    def forward(self, pred, obs):
        """
        Computes the hybrid loss function based on the Kling-Gupta Efficiency (KGE) 
        and Mean Squared Error (MSE).

        Args:
            pred (torch.Tensor): Predicted values tensor of shape (batch_size, ...).
            obs (torch.Tensor): Observed (ground truth) values tensor of the same shape as `pred`.

        Returns:
            torch.Tensor: Scalar loss value combining the negative KGE and weighted MSE.

        Notes:
            - The KGE is computed using Pearson correlation, variability ratio, and bias ratio.
            - Small epsilon values (`self.eps`) are added to prevent division by zero.
            - The final loss is `-KGE + lambda * MSE`, where `self.lambda_mse` controls 
            the weighting of MSE in the total loss.
        """

        # Ensure tensors are at least 1D
        if pred.ndim == 0 or obs.ndim == 0:
            return torch.tensor(float("nan"), device=pred.device)

        obs_mean = torch.mean(obs)
        pred_mean = torch.mean(pred)

        obs_std = torch.std(obs) + self.eps  # Avoid division by zero
        pred_std = torch.std(pred) + self.eps

        # Compute Pearson correlation manually
        covariance = torch.mean((pred - pred_mean) * (obs - obs_mean))
        r = covariance / (obs_std * pred_std + self.eps)  # Avoid zero denominator

        # Clamp r within [-1, 1] to prevent invalid values
        r = torch.clamp(r, -1 + self.eps, 1 - self.eps)

        alpha = pred_std / obs_std
        beta = pred_mean / (obs_mean + self.eps)

        # Compute KGE
        kge = 1 - torch.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)

        # Hybrid loss: -KGE + lambda * MSE
        mse = self.mse_loss(pred, obs)
        hybrid_loss = -kge + self.lambda_mse * mse

        return hybrid_loss
