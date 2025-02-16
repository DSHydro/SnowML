# Helper functions to download and manipulate metrics/artifacts stored on mlflow 

import boto3
import mlflow
import pandas as pd
import matplotlib.pyplot as plt


#s3://sues-test/34/b1649da4415449c49ad0841fd230d950/artifacts/SWE_Predictions_for_huc1711000504 using Baseline Model.png
# b1649da4415449c49ad0841fd230d950



def load_ml_metrics(tracking_uri, run_id, save_local=False):
    import mlflow
import pandas as pd

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
        f_out = f"metrics_from_{run_id}.csv"
        metrics_df.to_csv(f_out, index=False)


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

























#s3_bucket = 'sues-test'
#s3_key = '34/b1649da4415449c49ad0841fd230d950/artifacts/SWE_Predictions_for_huc1711000504 using Baseline Model.png'
# tracking_uri = "arn:aws:sagemaker:us-west-2:677276086662:mlflow-tracking-server/dawgsML"

def retrieve_plot (bucket, key): 
    s3 = boto3.client('s3')
    local_file_path = "../docs/model_results/SWE_Predictions_for_huc1711000504.png"
    s3.download_file(bucket, key, local_file_path)
    s3.close()


