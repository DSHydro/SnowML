
# pylint: disable=C0103
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import mlflow
import matplotlib.dates as mdates

def assemble_plot_dict(y_vals, color, label, n_offset=180, sparse = False):
    plot_dict = {}
    try: 
        plot_dict["y_axis_vals"] = np.concatenate([np.full(n_offset, np.nan), y_vals.flatten()])
    except: 
        plot_dict["y_axis_vals"] = np.concatenate([np.full(n_offset, np.nan), y_vals])
    plot_dict["color"] = color
    plot_dict["label"] = label
    plot_dict["sparse"] = sparse
    return plot_dict

def plot3(x_axis_vals, y_dict_list, ttl, metrics_dict = None):
    plt.figure(figsize=(12, 6))
    for plot_dict in y_dict_list:
        if plot_dict is not None: 
            plt.plot(
                x_axis_vals,
                plot_dict["y_axis_vals"],
                c = plot_dict["color"],
                label = plot_dict["label"])
    plt.legend(loc='upper right')
    plt.xlabel('Date')
    plt.ylabel('SWE')
    plt.title(ttl)

    # Display metrics in the upper-right corner if metrics_dict is not None
    if metrics_dict is not None:
        ax = plt.gca()
        metric_text = "\n".join([f"{key}: {value:.3f}" for key, value in metrics_dict.items()])

        ax.text(
            0.02, 0.98, metric_text, transform=ax.transAxes, ha='left', va='top',
            fontsize=10, color='black', bbox=dict(facecolor='white', alpha=0.7, edgecolor='black')
        )
    mlflow.log_figure(plt.gcf(), ttl + ".png")
    plt.close()


def plot3b(x_axis_vals, y_dict_list, ttl, metrics_dict=None):
    # Ensure x values are datetime
    x_axis_vals = pd.to_datetime(x_axis_vals)
    
    plt.figure(figsize=(12, 6))
    for plot_dict in y_dict_list:
        if plot_dict is not None: 

            # Check if the data is sparse
            if plot_dict.get("sparse", False):
                # Plot as large dots if sparse
                plt.scatter(
                    x_axis_vals,
                    plot_dict["y_axis_vals"],
                    c=plot_dict["color"],
                    label=plot_dict["label"],
                    s=25  # Size of the dots (larger to make them visible)
                )
            else:
                # Plot as a line if not sparse
                plt.plot(
                    x_axis_vals,
                    plot_dict["y_axis_vals"],
                    c=plot_dict["color"],
                    label=plot_dict["label"]
                )
                
    plt.legend(loc='upper right')
    plt.xlabel('Date')
    plt.ylabel('SWE')
    plt.title(ttl)

    # Format x-axis to show only the year
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    # Display metrics in the upper-left corner if metrics_dict is not None
    if metrics_dict is not None:
        ax = plt.gca()
        metric_text = "\n".join([f"{key}: {value:.3f}" for key, value in metrics_dict.items()])

        ax.text(
            0.02, 0.98, metric_text, transform=ax.transAxes, ha='left', va='top',
            fontsize=10, color='black', bbox=dict(facecolor='white', alpha=0.7, edgecolor='black')
        )

    return plt

    