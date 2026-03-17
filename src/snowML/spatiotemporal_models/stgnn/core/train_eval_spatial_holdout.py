"""
Final ST-GNN training and evaluation pipeline.
Supports spatial holdout: loss and metrics only on train_node_idx; adj_matrix is A_train (train-train edges only).
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import mlflow.pytorch
from .mtgnn import MTGNN
import matplotlib.pyplot as plt


class SlidingWindowDataset(Dataset):
    """Generate sliding windows on-the-fly to save memory."""

    def __init__(self, dynamic_features, target_tensor, seq_len):
        self.dynamic_features = dynamic_features  # [num_nodes, num_features, num_timesteps]
        self.target_tensor = target_tensor
        self.seq_len = seq_len
        self.num_samples = dynamic_features.shape[2] - seq_len

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        window = self.dynamic_features[:, :, idx: idx + self.seq_len]
        target = self.target_tensor[:, idx + self.seq_len]
        return window, target


def kling_gupta_efficiency(y_true, y_pred, eps=1e-8):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if np.isnan(y_true).any() or np.isnan(y_pred).any():
        return np.nan, np.nan, np.nan, np.nan
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return np.nan, np.nan, np.nan, np.nan
    r = np.corrcoef(y_true, y_pred)[0, 1]
    alpha = np.std(y_pred) / (np.std(y_true) + eps)
    mean_true = np.mean(y_true)
    beta = np.mean(y_pred) / (mean_true + eps) if not np.isclose(mean_true, 0.0) else np.nan
    kge = 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
    return kge, r, alpha, beta


def compute_metrics(y_true, y_pred):
    if np.isnan(y_true).any() or np.isnan(y_pred).any():
        return {"mse": np.nan, "mae": np.nan, "r2": np.nan, "kge": np.nan}
    kge, r, alpha, beta = kling_gupta_efficiency(y_true, y_pred)
    r2 = r2_score(y_true, y_pred) if np.std(y_true) != 0 else np.nan
    return {
        "mse": mean_squared_error(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2,
        "kge": kge,
    }


def train_model(
    dynamic_features,
    target_tensor,
    adj_matrix,
    static_features,
    train_node_idx,
    num_epochs=20,
    batch_size=32,
    seq_len=30,
    val_split=0.2,
    early_stopping=True,
    patience=5,
    rel_threshold=0.03,
    plot=True,
    log_model=False,
):
    """
    Train model with loss and val metrics only on train_node_idx (strict holdout).
    Caller passes A_train (only train-train edges); test nodes are isolated during training.

    Args:
        dynamic_features: [num_nodes, num_features, num_timesteps]
        target_tensor: [num_nodes, num_timesteps]
        adj_matrix: [num_nodes, num_nodes] — A_train (test rows/cols zeroed)
        static_features: [num_nodes, num_static_features]
        train_node_idx: 1D array or list of node indices used for loss and val metrics.
    """
    if num_epochs <= 0:
        raise ValueError("num_epochs must be greater than zero")
    train_node_idx = np.atleast_1d(np.asarray(train_node_idx, dtype=np.int64))

    print("training model")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    num_nodes, num_features, num_timesteps = dynamic_features.shape
    split_idx = int(num_timesteps * (1 - val_split))

    train_dataset = SlidingWindowDataset(
        dynamic_features[:, :, :split_idx],
        target_tensor[:, :split_idx],
        seq_len,
    )
    val_dataset = SlidingWindowDataset(
        dynamic_features[:, :, split_idx:],
        target_tensor[:, split_idx:],
        seq_len,
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    print(
        f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}, Train nodes: {len(train_node_idx)}"
    )

    model = MTGNN(
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
        seq_length=seq_len,
        in_dim=num_features,
        out_dim=1,
        layers=3,
        propalpha=0.05,
        tanhalpha=3,
        layer_norm_affline=True,
        xd=static_features.shape[1] if static_features is not None else None,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss(reduction="none")  # per-node loss, then mask

    adj = torch.tensor(adj_matrix, dtype=torch.float32).to(device)
    if static_features is not None:
        static_features = static_features.to(device)
    train_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    train_mask[train_node_idx] = True

    best_val_mse = float("inf")
    best_val_kge = -float("inf")
    best_epoch = -1
    epochs_no_improve = 0
    best_state = None
    train_loss_curve = []
    val_loss_curve = []
    epochs_curve = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_preds_list, train_targets_list = [], []

        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            Xb = Xb.permute(0, 2, 1, 3)
            optimizer.zero_grad()
            out = model(Xb, adj, FE=static_features).squeeze()
            # out, yb: [batch, num_nodes]
            per_node_loss = loss_fn(out, yb).mean(dim=0)
            loss = per_node_loss[train_mask].mean()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_preds_list.append(out.detach().cpu().numpy())
            train_targets_list.append(yb.detach().cpu().numpy())

        train_preds = np.concatenate(train_preds_list, axis=0)
        train_targets = np.concatenate(train_targets_list, axis=0)
        train_preds_train_nodes = train_preds[:, train_node_idx].flatten()
        train_targets_train_nodes = train_targets[:, train_node_idx].flatten()
        train_metrics = compute_metrics(train_targets_train_nodes, train_preds_train_nodes)

        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                Xb = Xb.permute(0, 2, 1, 3)
                out = model(Xb, adj, FE=static_features).squeeze()
                val_preds.append(out.cpu().numpy())
                val_targets.append(yb.cpu().numpy())

        val_preds = np.concatenate(val_preds, axis=0)
        val_targets = np.concatenate(val_targets, axis=0)
        val_preds_train_nodes = val_preds[:, train_node_idx].flatten()
        val_targets_train_nodes = val_targets[:, train_node_idx].flatten()
        val_metrics = compute_metrics(val_targets_train_nodes, val_preds_train_nodes)

        for metric_name, value in train_metrics.items():
            mlflow.log_metric(f"train_{metric_name}", value, step=epoch)
        for metric_name, value in val_metrics.items():
            mlflow.log_metric(f"val_{metric_name}", value, step=epoch)

        train_loss_curve.append(float(train_metrics["mse"]))
        val_loss_curve.append(float(val_metrics["mse"]))
        epochs_curve.append(epoch + 1)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"Epoch {epoch+1}/{num_epochs} Train - MSE: {train_metrics['mse']:.4f}, KGE: {train_metrics['kge']:.4f} "
                f"Val - MSE: {val_metrics['mse']:.4f}, KGE: {val_metrics['kge']:.4f}"
            )

        if early_stopping:
            current_mse = float(val_metrics["mse"])
            current_kge = (
                float(val_metrics["kge"]) if not np.isnan(val_metrics["kge"]) else -float("inf")
            )
            if np.isinf(best_val_mse):
                improved = True
            else:
                rel_impr = (best_val_mse - current_mse) / (abs(best_val_mse) + 1e-12)
                improved = rel_impr >= rel_threshold
            if improved:
                best_val_mse = current_mse
                best_val_kge = current_kge
                best_epoch = epoch
                epochs_no_improve = 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(
                    f"Early stopping at epoch {epoch+1}. Best epoch {best_epoch+1} "
                    f"val_mse={best_val_mse:.6f}, val_kge={best_val_kge:.6f}"
                )
                break

    if early_stopping and best_epoch >= 0:
        mlflow.log_metric("best_epoch", best_epoch + 1)
        mlflow.log_metric("best_val_mse", best_val_mse)
        mlflow.log_metric("best_val_kge", best_val_kge)
        val_metrics_return = {"val_mse": best_val_mse, "val_kge": best_val_kge}
    else:
        last_kge = float(val_metrics["kge"]) if not np.isnan(val_metrics["kge"]) else -float("inf")
        mlflow.log_metric("best_epoch", epoch + 1)
        mlflow.log_metric("best_val_mse", float(val_metrics["mse"]))
        mlflow.log_metric("best_val_kge", last_kge)
        val_metrics_return = {"val_mse": float(val_metrics["mse"]), "val_kge": last_kge}

    if early_stopping and best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)
        model.eval()

    if log_model:
        mlflow.pytorch.log_model(model, "model")

    if plot:
        plt.figure()
        plt.plot(epochs_curve, train_loss_curve, label="train_mse")
        plt.plot(epochs_curve, val_loss_curve, label="val_mse")
        plt.xlabel("Epoch")
        plt.ylabel("MSE")
        plt.legend()
        plt.grid(True)
        plt.show()

    return model, val_metrics_return


def run_test_predictions(
    model,
    dynamic_features,
    target_tensor,
    adj_matrix,
    static_features,
    test_node_idx,
    batch_size=32,
    seq_len=30,
):
    """
    Run inference; return predictions and targets for test_node_idx only.
    adj_matrix should be A_test (only test-test edges) for strict holdout.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_node_idx = np.atleast_1d(np.asarray(test_node_idx, dtype=np.int64))

    model = model.to(device).eval()
    test_dataset = SlidingWindowDataset(dynamic_features, target_tensor, seq_len)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    adj = torch.tensor(adj_matrix, dtype=torch.float32).to(device)
    if static_features is not None:
        static_features = static_features.to(device)

    preds_list, targets_list = [], []
    with torch.no_grad():
        for Xb, yb in test_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            Xb = Xb.permute(0, 2, 1, 3)
            out = model(Xb, adj, FE=static_features).squeeze()
            preds_list.append(out.cpu().numpy())
            targets_list.append(yb.cpu().numpy())

    preds = np.concatenate(preds_list, axis=0)
    targets = np.concatenate(targets_list, axis=0)
    # preds, targets: [num_samples, num_nodes]
    preds = preds.T
    targets = targets.T
    return preds[test_node_idx], targets[test_node_idx], test_node_idx

