# Helper functions to download and manipulate metrics/artifacts stored on mlflow 

import os
import boto3
import mlflow
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from snowML.datapipe import snow_types as st

#s3://sues-test/34/b1649da4415449c49ad0841fd230d950/artifacts/SWE_Predictions_for_huc1711000504 using Baseline Model.png
# b1649da4415449c49ad0841fd230d950 (Skagit 10)
# 215fe6997cc04f4493cce8d003dea9a5 (Skagit 12)
# b8c0693ac05e4d26b1011202ba551cfd (Chelan 12 - 64 )
# b1643a1474a247668feb4065db3975f1 (Skagit 10 - 64) *Best?
# c56b34c34f3d4988a9d5781fc7a78790 (Skagit 10 - 128)
# 7848ddbeadb84d358dfc7450df3ae9ab (Skagit 12 - 128, 5 epochs)
# d2f3f6f705014660917cfae0c0716236" (Skagit 12 - 64.  Use this one!)
# arn:aws:sagemaker:us-west-2:677276086662:mlflow-tracking-server/dawgsML

def load_ml_metrics(tracking_uri, run_id, save_local=False):
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.MlflowClient()
    # Get all metric keys from the run
    run_data = client.get_run(run_id).data
    metric_keys = run_data.metrics.keys()
    # Retrieve full metric history for each key
    all_metrics = []
    for metric in metric_keys:
        history = client.get_metric_history(run_id, metric)
        for record in history:
            all_metrics.append({
                "Metric": metric,
                "Step": record.step,
                "Value": record.value
            })
    
    # Convert to DataFrame
    metrics_df = pd.DataFrame(all_metrics)
    
    # Save to CSV if needed
    if save_local:
        f_out = f"run_id_data/metrics_from_{run_id}.csv"
        metrics_df.to_csv(f_out, index=False)

    return metrics_df

def summarize_by_step(df, step, agg_lev = 12):
    df_filtered = df[df["Step"] == step].copy()
    df_filtered["Metric_Type"] = df_filtered["Metric"].str.extract(r"(test_mse|test_kge|train_mse|train_kge)")
    df_filtered["HUC_ID"] = df_filtered["Metric"].str.extract(fr"(\d{{{agg_lev}}})")  

    # Take mean across HUC_ID if duplicates exist
    if df_filtered.duplicated(subset=["HUC_ID", "Metric_Type"]).any():
        df_filtered = df_filtered.groupby(["HUC_ID", "Metric_Type"], as_index=False)["Value"].mean()

    df_pivot = df_filtered.pivot(index="HUC_ID", columns="Metric_Type", values="Value")
    df_pivot.columns = ["Test KGE", "Test MSE", "Train KGE", "Train_MSE"]
    df_pivot_sorted = df_pivot.sort_index()
    df_selected = df_pivot_sorted[["Test MSE", "Test KGE"]]
    # print(df_selected)
    return df_selected




def plot_test_kge_histogram(df, output_file = "histogram.png"):
    """
    Plots a histogram of the test_kge values from a pandas DataFrame and saves it as 'histogram.png'.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing the column 'test_kge'.
    """
    if 'Test KGE' not in df.columns:
        raise ValueError("DataFrame must contain a 'Test KGE' column")
    
    test_kge_values = df['Test KGE'].dropna()
    median_kge = np.median(test_kge_values)
    
    plt.figure(figsize=(8, 6))
    plt.hist(df['Test KGE'].dropna(), bins=20, edgecolor='black', alpha=0.7)
    plt.axvline(median_kge, color='red', linestyle='dashed', linewidth=2, label=f'Median: {median_kge:.2f}')
    plt.text(median_kge, plt.ylim()[1] * 0.9, f'Median: {median_kge:.2f}', color='red', ha='right', fontsize=12, fontweight='bold')
    plt.xlabel('Test KGE')
    plt.ylabel('Frequency')
    plt.title('Histogram of Test KGE Values')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(output_file)
    return plt


def plot_metric(df, metric_type, output_file="plot.png"):
    """
    Plots a given metric type (test_mse, train_mse, test_kge, train_kge) over epochs for different HUCs,
    and includes an average line across all HUCs.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing 'Metric', 'Step', and 'Value' columns.
        metric_type (str): The metric type to plot (e.g., "test_mse", "train_mse", "test_kge", "train_kge").
        output_file (str): The filename to save the plot.
    """

    # Filter for the specified metric type
    metric_df = df[df["Metric"].str.startswith(f"{metric_type}_")].copy()

    # Extract HUC ID from the metric name
    metric_df["HUC"] = metric_df["Metric"].str.extract(rf"{metric_type}_(\d+)")

    # Calculate the average metric value across all HUCs for each epoch
    avg_metric_df = metric_df.groupby("Step")["Value"].mean().reset_index()
    
    # Print the average values in a readable format
    print(f"\nAverage {metric_type} over all HUCs by epoch:")
    print(avg_metric_df.to_string(index=False))

    # Plot each HUC separately
    for huc_id, group in metric_df.groupby("HUC"):
        plt.plot(group["Step"], group["Value"], marker="o", linestyle="-", label=f"HUC {huc_id}")

    # Plot the average metric as a thick black line
    plt.plot(avg_metric_df["Step"], avg_metric_df["Value"], "k-", linewidth=3, label=f"Avg {metric_type}")

    # Labels and title
    plt.xlabel("Epoch")
    plt.ylabel(metric_type.replace("_", " ").title())  # Format label nicely
    plt.title(f"{metric_type.replace('_', ' ').title()} over Epochs for Different HUCs")
    plt.legend(title="HUC ID", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True)

    # Save the plot
    plt.savefig(output_file, bbox_inches="tight")
    plt.close()  # Close the plot to free memory

    print(f"\nPlot saved as {output_file}")


def retrieve_plot (huc, key, local_file_path):  
    bucket = "sues-test" # TO DO - UPDATE TO DIFF S3 BUCKET
    s3 = boto3.client('s3')
    s3.download_file(bucket, key, local_file_path)
    s3.close()


def load_snow_type_data(input_pairs):
    snow_types = pd.DataFrame()
    for pair in input_pairs:
        huc_id = pair[0]
        huc_lev = pair[1]
        df, _ = st.process_all(huc_id, huc_lev)
        df.set_index("huc_id", inplace=True)
        snow_types = pd.concat([snow_types, df])
    return snow_types 

def plot_kge(df_merged, x_var, y_var, ttl = "Scatter Plot of test_kge vs Ephemeral"):
    # Define colors based on huc_id prefixes
    colors = df_merged.index.astype(str).map(lambda x: 'red' if x.startswith('1702') 
                                             else 'green' if x.startswith('1703') 
                                             else 'blue' if x.startswith('1711') 
                                             else 'gray')  # Default color for other values

    # Create scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(df_merged[x_var], df_merged[y_var], c=colors, alpha=0.7, edgecolors='k')
    
    # Labels and title
    plt.xlabel("Ephemeral")
    plt.ylabel("test_kge")
    plt.title(ttl)
    
    # Show plot
    return plt

def plot_kge_v_ephemeral(tracking_uri, run_id, huc_id, huc_lev):
    input_pairs = [[huc_id, huc_lev]]
    metrics = load_ml_metrics(tracking_uri, run_id)
    summary9 =  summarize_bystep(metrics, 9, huc_lev)
    df_class = load_snow_type_data(input_pairs)
    summary9.index = summary9.index.astype(str)
    df_class.index = df_class.index.astype(str)
    df_all = summary9.merge(df_class, left_index=True, right_index=True, how="inner")
    ttl = f"test_kge_vs_percent_ephemeral_in_regions{huc_id}"
    plt = plot_kge_w_trend2(df_all, "Ephemeral", "Test KGE", ttl)
    plt.savefig(f"{ttl}.png")
    return df_all

def plot_kge_v_peak(tracking_uri, run_id, huc_id, huc_lev, df_peaks):
    input_pairs = [[huc_id, huc_lev]]
    metrics = load_ml_metrics(tracking_uri, run_id)
    summary9 =  summarize_bystep(metrics, 9, huc_lev)
    summary9.index = summary9.index.astype(str)
    df_peaks.index = df_peaks.index.astype(str)
    df_all = summary9.merge(df_peaks, left_index=True, right_index=True, how="inner")
    return df_all

def hist_from_run(run_id, last_step, tracking_id = "arn:aws:sagemaker:us-west-2:677276086662:mlflow-tracking-server/dawgsML"):
    metrics = load_ml_metrics(tracking_id, run_id, save_local=True)
    metrics_last = summarize_by_step(metrics, last_step, agg_lev = 12) # TO DO REVERT 12
    f_out = f"kge_hist{run_id}_last_hist.png"
    plot_test_kge_histogram(metrics_last, output_file = f_out)


def stepwise_hists(run_id, epochs, tracking_id="arn:aws:sagemaker:us-west-2:677276086662:mlflow-tracking-server/dawgsML"):
    metrics = load_ml_metrics(tracking_id, run_id, save_local=True)
    temp_files = []
    
    for epoch in range(epochs): 
        metrics_epoch = summarize_by_step(metrics, epoch, agg_lev=12)  # TO DO REVERT 12
        temp_file = f"hist_epoch_{epoch}.png"
        plot_test_kge_histogram(metrics_epoch, output_file=temp_file)
        temp_files.append((epoch, temp_file))
    
    # Determine grid size
    num_histograms = len(temp_files)
    cols = 3  # Three per row
    rows = (num_histograms + cols - 1) // cols  # Calculate needed rows

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 5))  # Adjust figure size
    axes = np.array(axes).reshape(rows, cols)  # Ensure axes is always 2D

    for ax in axes.flat:
        ax.axis("off")  # Hide all axes initially

    for (epoch, temp_file), ax in zip(temp_files, axes.flat):
        img = plt.imread(temp_file)  # Load the image
        ax.imshow(img)
        ax.set_title(f"Epoch Number {epoch}", fontsize=14, fontweight="bold")
        ax.axis("off")  # Hide axis labels

    plt.tight_layout()
    f_out = f"kge_hist/{run_id}_all_hists.png"
    plt.savefig(f_out, dpi=300)
    plt.close()

    # Cleanup temporary files
    for _, f in temp_files:
        os.remove(f)

    print(f"Saved combined histogram image as {f_out}")

        