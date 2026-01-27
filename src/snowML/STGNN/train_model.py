import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from mtgnn import MTGNN

import mlflow
import mlflow.pytorch


def create_mtgnn_model(num_features, num_nodes, seq_length, **kwargs):
    return MTGNN(
        gcn_true=True,
        build_adj=False,
        gcn_depth=2,
        num_nodes=num_nodes,
        kernel_set=[2, 3, 6, 7],
        kernel_size=7,
        dropout=0.3,
        subgraph_size=20,
        node_dim=40,
        dilation_exponential=1,
        conv_channels=32,
        residual_channels=32,
        skip_channels=16,
        end_channels=64,
        seq_length=seq_length,
        in_dim=num_features,
        out_dim=1,
        layers=3,
        propalpha=0.05,
        tanhalpha=3,
        layer_norm_affline=True,
        xd=None,
    )


def train_swe_model(
    X,
    y,
    adj_matrix,
    num_epochs=20,
    batch_size=32,
    seq_len=30,
    experiment_name="STGNN_SWE",
):
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        num_nodes = X.shape[2]
        num_features = X.shape[1]
        model = create_mtgnn_model(num_features, num_nodes, seq_len).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = torch.nn.MSELoss()
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        adj = torch.tensor(adj_matrix, dtype=torch.float32).to(device)

        # Log parameters
        mlflow.log_param("num_epochs", num_epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("seq_len", seq_len)
        mlflow.log_param("num_nodes", num_nodes)
        mlflow.log_param("num_features", num_features)

        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            for Xb, yb in loader:
                Xb, yb = Xb.to(device), yb.to(device)
                optimizer.zero_grad()
                out = model(Xb, adj).squeeze()  # [batch, num_nodes]
                loss = loss_fn(out, yb)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(loader)
            print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
            mlflow.log_metric("train_loss", avg_loss, step=epoch)

        # Log the model
        mlflow.pytorch.log_model(model, "model")
    return model
