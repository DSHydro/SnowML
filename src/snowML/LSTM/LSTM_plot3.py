
# pylint: disable=C0103
import matplotlib.pyplot as plt
import numpy as np
import mlflow

def assemble_plot_dict(y_vals, color, label, n_offset=180): 
    plot_dict = {}
    plot_dict["y_axis_vals"] = np.concatenate([np.full(n_offset, np.nan), y_vals.flatten()])
    plot_dict["color"] = color
    plot_dict["label"] = label
    return plot_dict
 
def plot3(x_axis_vals, y_dict_list, ttl, metrics_dict = None): 
    plt.figure(figsize=(12, 6))
    for plot_dict in y_dict_list: 
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