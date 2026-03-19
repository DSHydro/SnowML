"""
Minimal Experiment Tracking for SageMaker Studio
Essential functionality only - easy to review and understand
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import mlflow.pytorch
from mtgnn import MTGNN
import matplotlib.pyplot as plt

class SlidingWindowDataset(Dataset):
    """Generate sliding windows on-the-fly to save memory"""
    def __init__(self, dynamic_features, target_tensor, seq_len):
        self.dynamic_features = dynamic_features  # [num_nodes, num_features, num_timesteps]
        self.target_tensor = target_tensor        # [num_nodes, num_timesteps]
        self.seq_len = seq_len
        self.num_samples = dynamic_features.shape[2] - seq_len
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        window = self.dynamic_features[:, :, idx:idx+self.seq_len]
        target = self.target_tensor[:, idx+self.seq_len]
        return window, target

def kling_gupta_efficiency(y_true, y_pred):
    # Flatten
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    # Check for NaNs
    if np.isnan(y_true).any() or np.isnan(y_pred).any():
        # print("Error: NaN values detected in y_true or y_pred")
        return np.nan, np.nan, np.nan, np.nan

    # Check for zero variance
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        # print("Error: Zero variance detected in y_true or y_pred")
        return np.nan, np.nan, np.nan, np.nan

    r = np.corrcoef(y_true, y_pred)[0, 1]
    alpha = np.std(y_pred) / np.std(y_true)
    beta = np.mean(y_pred) / np.mean(y_true)

    # mean(y_true) could be 0 => beta inf; handle it explicitly if you want:
    # if np.isclose(np.mean(y_true), 0.0): return np.nan, r, alpha, np.nan

    kge = 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
    return kge, r, alpha, beta



def compute_metrics(y_true, y_pred):
    """Calculate all evaluation metrics"""

    # KGE (Kling-Gupta Efficiency)
    # KGE = 1 - sqrt((r-1)^2 + (alpha-1)^2 + (beta-1)^2)
    # where: r = correlation, alpha = std ratio, beta = mean ratio
    if np.isnan(y_true).any() or np.isnan(y_pred).any():
        return {'mse': np.nan, 'mae': np.nan, 'r2': np.nan, 'kge': np.nan}

    kge, r, alpha, beta = kling_gupta_efficiency(y_true, y_pred)

    # R2 needs at least some variance in y_true; if std==0, r2_score returns 0.0 in sklearn,
    # but you may prefer NaN for consistency:
    if np.std(y_true) == 0:
        r2 = np.nan
    else:
        r2 = r2_score(y_true, y_pred)

    return {
        'mse': mean_squared_error(y_true, y_pred),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2,
        'kge': kge,
    }


def test_model(
    model,
    dynamic_features,
    target_tensor,
    adj_matrix,
    static_features,
    batch_size=32,
    seq_len=30,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = model.to(device)
    model.eval()

    test_dataset = SlidingWindowDataset(dynamic_features, target_tensor, seq_len)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Move graph + static features once
    adj = torch.tensor(adj_matrix, dtype=torch.float32).to(device)
    if static_features is not None:
        static_features = static_features.to(device)

    preds_list, targets_list = [], []
    with torch.no_grad():
        for Xb, yb in test_loader:
            Xb, yb = Xb.to(device), yb.to(device)

            # MTGNN expects: (batch, features, nodes, seq_len)
            Xb = Xb.permute(0, 2, 1, 3)
            out = model(Xb, adj, FE=static_features).squeeze()
            preds_list.append(out.detach().cpu().numpy())
            targets_list.append(yb.detach().cpu().numpy())

    preds = np.concatenate(preds_list, axis=0)
    targets = np.concatenate(targets_list, axis=0)

    # preds/targets are typically [num_samples, num_nodes]
    preds = preds.T
    targets = targets.T
    print(f"Predicts shape :{preds.shape}")
    print(f"Targets shape :{targets.shape}")
    return preds, targets


def train_model(
    dynamic_features,
    target_tensor,
    adj_matrix,
    static_features,
    num_epochs=20,
    batch_size=32,
    seq_len=30,
    val_split=0.2,
    experiment_name="STGNN_SWE",
    run_name=None,
    early_stopping=True,
    patience=5,
):
    """
    Train model with MLflow tracking
    
    Args:
        dynamic_features: [num_nodes, num_features, num_timesteps]
        target_tensor: [num_nodes, num_timesteps]
        adj_matrix: [num_nodes, num_nodes]
        static_features: [num_nodes, num_static_features]
    """
    print("training model")
    # Start MLflow tracking
    mlflow.set_experiment(experiment_name)
    mlflow.start_run(run_name=run_name)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Train/val split
    num_nodes, num_features, num_timesteps = dynamic_features.shape
    split_idx = int(num_timesteps * (1 - val_split))
    
    train_dataset = SlidingWindowDataset(
        dynamic_features[:, :, :split_idx],
        target_tensor[:, :split_idx],
        seq_len
    )
    
    val_dataset = SlidingWindowDataset(
        dynamic_features[:, :, split_idx:],
        target_tensor[:, split_idx:],
        seq_len
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Log basic config
    mlflow.log_params({
        'num_nodes': num_nodes,
        'num_features': num_features,
        'batch_size': batch_size,
        'num_epochs': num_epochs,
        'seq_len': seq_len
    })
    
    # Create model
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
    loss_fn = torch.nn.MSELoss()
    
    # Move to device
    adj = torch.tensor(adj_matrix, dtype=torch.float32).to(device)
    if static_features is not None:
        static_features = static_features.to(device)

    best_score = float("inf")
    best_epoch = -1
    epochs_no_improve = 0
    best_state = None
    train_loss_curve = []
    val_loss_curve = []
    epochs_curve = []
    
    # Training loop
    for epoch in range(num_epochs):
        # Train
        model.train()
        train_loss = 0
        train_preds_list, train_targets_list = [], []
        
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            Xb = Xb.permute(0, 2, 1, 3)
            
            optimizer.zero_grad()
            out = model(Xb, adj, FE=static_features).squeeze()
            loss = loss_fn(out, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            # Collect training predictions for KGE
            train_preds_list.append(out.detach().cpu().numpy())
            train_targets_list.append(yb.detach().cpu().numpy())

        # Calculate train metrics
        avg_train_loss = train_loss / len(train_loader)
        train_preds = np.concatenate(train_preds_list).flatten()
        train_targets = np.concatenate(train_targets_list).flatten()
        train_metrics = compute_metrics(train_targets, train_preds)
        
        # Validate
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                Xb = Xb.permute(0, 2, 1, 3)
                out = model(Xb, adj, FE=static_features).squeeze()
                val_preds.append(out.cpu().numpy())
                val_targets.append(yb.cpu().numpy())
        
        val_preds = np.concatenate(val_preds).flatten()
        val_targets = np.concatenate(val_targets).flatten()
        val_metrics = compute_metrics(val_targets, val_preds)
                
        # Log metrics
        mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
        # Log all train metrics (KGE, MSE, etc.)
        for metric_name, value in train_metrics.items():
            mlflow.log_metric(f"train_{metric_name}", value, step=epoch)
        # Log validation metrics
        for metric_name, value in val_metrics.items():
            mlflow.log_metric(f"val_{metric_name}", value, step=epoch)
        
        train_loss_curve.append(float(train_metrics['mse']))          # avg batch loss (MSELoss mean)
        val_loss_curve.append(float(val_metrics["mse"])) # true val MSE over full val set
        epochs_curve.append(epoch + 1)
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train - Loss: {avg_train_loss:.4f}, MSE: {train_metrics['mse']:.4f}, KGE: {train_metrics['kge']:.4f}")
        print(f"Val   - MSE: {val_metrics['mse']:.4f}, KGE: {val_metrics['kge']:.4f}")

        current = float(val_metrics["mse"])
        min_delta = 100
        if epoch != 0:
            min_delta = 0.01 * best_score
        improved = current < (best_score - min_delta)

        if improved:
            best_score = current
            best_epoch = epoch
            epochs_no_improve = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            mlflow.log_metric("best_score", best_score, step=epoch)
            mlflow.log_metric("best_epoch_so_far", best_epoch + 1)
        else:
            epochs_no_improve += 1

        # Early stopping check
        if early_stopping and epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1}. Best was epoch {best_epoch+1} with mse={best_score:.6f}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)
        model.eval()
        mlflow.log_metric("best_epoch", best_epoch + 1)
        mlflow.log_metric("best_final_score", best_score)

    # Save model
    mlflow.pytorch.log_model(model, "model")
    mlflow.end_run()

    plt.figure()
    plt.plot(epochs_curve, train_loss_curve, label="train_mse")
    plt.plot(epochs_curve, val_loss_curve, label="val_mse")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.title("Train vs Val MSE")
    plt.legend()
    plt.grid(True)
    plt.show()

    return model