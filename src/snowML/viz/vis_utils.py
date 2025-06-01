# pylint: disable=C0103
"Module to create basin and watershed visualizations; visualizations of metrics"

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from itertools import combinations
from scipy.stats import ttest_ind_from_stats
from snowML.datapipe.utils import snow_types as st
from snowML.datapipe.utils import get_geos as gg
from snowML.datapipe.utils import data_utils as du
from snowML.datapipe.utils import set_data_constants as sdc
from snowML.datapipe.utils import get_dem as gd

def plot_var(df, var, huc, initial_huc):
    plt.figure(figsize=(12,  6))
    plt.plot(df.index, df[var], c='b', label= f"Actual {var}")

    # Set y-axis limits for "mean_swe"
    if var == "mean_swe":
        plt.ylim(0, 2)

    plt.legend()
    plt.xlabel('Date')
    plt.ylabel(var)
    ttl = f'Actual {var} for huc{huc}'
    plt.title(ttl)
    # save file

    # Define the output directory and ensure it exists
    output_dir = os.path.join("docs", "var_plots_actuals", str(initial_huc))
    os.makedirs(output_dir, exist_ok=True)
    file_name = f"{ttl}.png"
    file_path = os.path.join(output_dir, file_name)
    plt.savefig(file_path)
    plt.close()  # Close the figure to free memory
    #print(f"Map saved to {file_path}")


def get_model_ready (huc, bucket_dict = None):
    if bucket_dict is None:
        bucket_dict = sdc.create_bucket_dict("prod")
    bucket_name = bucket_dict.get("model-ready")
    file_name = f"model_ready_huc{huc}.csv"
    df = du.s3_to_df(file_name, bucket_name)
    df['day'] = pd.to_datetime(df['day'])
    df.set_index('day', inplace=True)  # Set 'day' as the index
    return df

def plot_actual(huc, var, initial_huc, bucket_dict = None):
    df = get_model_ready(huc, bucket_dict= bucket_dict)
    plot_var(df, var, huc, initial_huc)

def summarize_swe(df):
    """
    Summarizes the mean_swe variable for each water year.
    A water year starts on October 1 and ends on September 30.
    
    Parameters:
        df (pd.DataFrame): DataFrame with index 'day' and column 'mean_swe'.
        
    Returns:
        pd.DataFrame: DataFrame with water year as index and columns 'annual_max_swe' and 'annual_mean_swe'.
    """
    # Ensure the index is a datetime index
    df = df.copy()
    df.index = pd.to_datetime(df.index)

    # Resample by water year (Oct 1 - Sep 30)
    summary = df.resample('YE-SEP').agg(
        annual_peak_swe=('mean_swe', 'max'),
        annual_mean_swe=('mean_swe', 'mean')
    )

    # Adjust index to represent the water year
    summary.index = summary.index.year

    # Calculate and print median values
    median_peak_swe = summary['annual_peak_swe'].median()
    median_ann_mean_swe = summary['annual_mean_swe'].median()
    #print(f"Median of annual max SWE: {median_peak_swe}")
    #print(f"Median of annual mean SWE: {median_ann_mean_swe}")

    return median_peak_swe, median_ann_mean_swe, summary

def basin_swe_summary(huc_id, final_huc_lev):
    geos = gg.get_geos(huc_id, final_huc_lev)
    hucs = geos["huc_id"]
    medians = []
    for huc in hucs:
        df = get_model_ready(huc)
        median_peak_swe, _, _, = summarize_swe(df)
        medians.append(median_peak_swe)
    results = pd.DataFrame({'huc_id': hucs, 'Median Peak Swe': medians})
    results.set_index('huc_id', inplace=True)
    f_out = f"docs/tables/Peak_annual_swe_huc{huc}.csv"
    results.to_csv(f_out)
    return results


def basic_map(geos, final_huc_lev, initial_huc):
    map_object = geos.explore()
    output_dir = os.path.join("docs", "basic_maps")
    file_name = f"Huc{final_huc_lev}_in_{initial_huc}.html"
    file_path = os.path.join(output_dir, file_name)
    map_object.save(file_path)
    print(f"Map saved to {file_path}")

def snow_colors():
    snow_class_colors_small = {
    3: "#FFFF00",  # Maritime (yellow)
    4: "#FFFFFF",  # Ephemeral (white)
    5: "#E31A1C",  # Prairie (red)
    6: "#FDBF6F",  # Montane Forest (orange)
    7: "#000000",  # Ice (black)
    }
    return snow_class_colors_small


def snow_colors_2():
    snow_class_colors_small = {
        3: "blue", # Maritime
        4: "#E6E6FA",  # Ephemeral (lavender)
        5: "lightgreen", # Prairie
        6: "darkgreen"  # Montane Forest
    }
    return snow_class_colors_small


def calc_bounds(geos):
    merged_geom = geos.geometry.union_all()
    outer_bound = merged_geom.convex_hull
    return outer_bound


def map_snow_types(ds, geos, huc, huc_lev = '12', class_colors = None, output_dir = None):

    # Set up the Cartopy projection
    fig, ax = plt.subplots(
    figsize=(10, 6),
    subplot_kw={"projection": ccrs.PlateCarree()}
    )

    # Add a baselayer

    if class_colors is None:
        class_colors = snow_colors_2()
    # Create a colormap and normalization based on the dictionary
    cmap = mcolors.ListedColormap([class_colors[i] for i in sorted(class_colors.keys())])
    bounds = list(class_colors.keys()) + [max(class_colors.keys()) + 1]  # Class boundaries
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    #Set extent
    outer_bound = calc_bounds(geos)
    minx, miny, maxx, maxy = outer_bound.bounds
    ax.set_extent([minx, maxx, miny, maxy], crs=ccrs.PlateCarree())

    # Plot the `snow_class` variable
    im = ax.pcolormesh(
        ds.lon,
        ds.lat,
        ds.SnowClass,
        cmap=cmap,
        norm=norm,
        transform=ccrs.PlateCarree()
    )

    # Plot the geometry outlines from the GeoDataFrame
    geos.plot(
        ax=ax,
        edgecolor="black",  # Color for the outlines
        facecolor="none",   # No fill
        linewidth=1.0,      # Thickness of the outlines
        transform=ccrs.PlateCarree()  # Ensure proper projection
    )

    # Set title and gridlines
    ax.set_title(f"Snow Classes In Huc {huc}", fontsize=14)
    ax.gridlines(draw_labels=True, linewidth=0.5, color="gray", alpha=0.5, linestyle="--")

    # Add a legend with a border around each color
    legend_patches = [
        mpatches.Patch(facecolor=color, edgecolor="black", label=label, linewidth=1)
        for label, color in zip(
            ["Maritime", "Ephemeral", "Prairie", "Montane Forest", "Ice"],
            [class_colors[i] for i in sorted(class_colors.keys())]
        )
    ]

    # Position the legend to the right, aligned with the top
    ax.legend(
        handles=legend_patches,
        title="Snow Classification",
        loc="upper left",
        bbox_to_anchor=(1.15, 1),
        borderaxespad=0
    )

    plt.tight_layout()

    # Save the plot
    if huc_lev != '12':
        file_name = f"Snow_classes_for_huc{huc_lev}in_{huc}.png"
    else:
        file_name = f"Snow_classes_in_{huc}.png"
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = os.path.join("docs", "basic_maps")
    file_path = os.path.join(output_dir, file_name)
    plt.savefig(file_path)
    plt.close(fig)  # Close the figure to free memory
    print(f"Map saved to {file_path}")

def plot_dem(dem_ds, geos, huc_id, f_out = None):
    _, ax = plt.subplots(figsize=(10, 6))
    dem_ds.plot(ax=ax, cmap='terrain')
    # Plot geometries in black outline
    geos.plot(ax=ax, edgecolor='black', facecolor='none', linewidth=2, zorder=5)
    ax.set_title(f"Digital Elevation Model (DEM) for huc {huc_id}")
    if f_out is None: 
        f_out = f"docs/basic_maps/dem_huc{huc_id}" # TO DO: Fix path to be dynamic
    plt.savefig(f_out, dpi=300, bbox_inches="tight")
    print(f"Map saved to {f_out}")
    plt.close()

def create_vis_all(initial_huc, final_huc_lev):
    geos = gg.get_geos(initial_huc, final_huc_lev)
    basic_map(geos, final_huc_lev, initial_huc) # create and save basic map
    ds_snow = st.snow_class_data_from_s3(geos)
    map_snow_types(ds_snow, geos, initial_huc, final_huc_lev) # create and save snow class map
    dem_ds = gd.get_dem(geos)
    plot_dem(dem_ds, geos, initial_huc) # create and save map of elevation
    # for huc in geos["huc_id"].tolist(): # create and save map of actuals
        # plot_actual(huc, "mean_swe", initial_huc, bucket_dict = None)
    # swe_summary = basin_swe_summary(initial_huc, final_huc_lev) # create & save csv 



def plot_scatter(df, x_var_name, y_var_name, color_map, title="Scatter_Plot", save_local=True, show_legend=True):
    """
    Creates a scatter plot of specified x and y variables, colored by Predominant Snow Type.

    Parameters:
    - df: DataFrame containing the x and y variables and "color_snow_type" column.
    - x_var_name: Column name for the x-axis variable.
    - y_var_name: Column name for the y-axis variable.
    - color_map: Dictionary mapping labels to their respective colors for the legend.
    - title: Title of the plot (default: "Scatter Plot" with underscores).
    - save_local: If True, saves the plot as a PNG file.
    - show_legend: If True, shows the legend (default is True).
    """
    plt.figure(figsize=(10, 6))

    # Use colors directly from the dataframe, default to white if missing
    colors = df["color_snow_type"].fillna("white")

    plt.scatter(df[x_var_name], df[y_var_name], c=colors, alpha=0.7, edgecolors="k")

    # Add labels and title
    plt.xlabel(x_var_name.replace("_", " "))
    plt.ylabel(y_var_name.replace("_", " "))
    plt.title(title.replace("_", " "))

    # Show legend if show_legend is True
    if show_legend:
        handles = [plt.Line2D([0], [0], marker='o', color=color, markersize=8, label=label) 
               for label, color in color_map.items()]
        plt.legend(handles=handles, title="Predominant Snow Type", loc='lower right', bbox_to_anchor=(0.95, 0.05))

    # Show or save plot
    if save_local:
        plt.savefig(f"charts/{title}.png", bbox_inches='tight')
        
    plt.show()





def plot_boxplot_by_group(df, parameter, title, groupby_column, color_map=None, category_order=None, trunc=False, save_local=True):
    """
    Plot a boxplot for the given parameter grouped by a specified column.
    
    Parameters:
    - df: DataFrame containing the data.
    - parameter: The name of the parameter to be plotted (as a column in the DataFrame).
    - title: The title of the plot.
    - groupby_column: The column by which to group the data.
    - color_map: A dictionary mapping categories to colors. Defaults to None.
    - category_order: Optional list of categories in the desired order for plotting.
    - trunc: If True, rotates labels 90 degrees and truncates them to 15 characters.
    - save_local: If True, saves the plot as a PNG file in the 'charts/' directory.
    """
    
    grouped_data = df.groupby(groupby_column)[parameter].agg(['count', 'median', 'mean', 'std'])
    #print(f"\nParameter Summary for '{parameter}' by '{groupby_column}':")
    #print(grouped_data, "\n")
    
    # Define a default color map if none is provided
    if color_map is None:
        default_palette = sns.color_palette("twilight", len(df[groupby_column].unique()))
        color_map = dict(zip(sorted(df[groupby_column].unique()), default_palette))

    # If category_order is not provided, use the keys of the color_map in order
    if category_order is None:
        category_order = list(color_map.keys())

    # Create the boxplot
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x=groupby_column, y=parameter, palette=color_map, order=category_order)

    # Set the title and labels
    plt.title(title)
    plt.xlabel(groupby_column)
    plt.ylabel(parameter)

    # Adjust x-axis labels based on truncation
    if trunc:
        plt.xticks(rotation=90)  # Rotate labels 90 degrees
    else:
        plt.xticks(rotation=0)  # Keep labels horizontal

    # Ensure layout is clean
    plt.tight_layout()

    # Save the plot locally if save_local is True
    if save_local:
        os.makedirs("charts", exist_ok=True)  # Ensure the directory exists
        safe_title = "".join(c if c.isalnum() or c in (" ", "-", "_") else "_" for c in title)  # Remove invalid filename characters
        plt.savefig(f"charts/{safe_title}.png", bbox_inches='tight')
        

    # Show the plot
    plt.show()

    return grouped_data

def plot_scatter(df, x_var_name, y_var_name, color_map, title="Scatter_Plot", save_local=True, show_legend=True):
    """
    Creates a scatter plot of specified x and y variables, colored by Predominant Snow Type.

    Parameters:
    - df: DataFrame containing the x and y variables and "color_snow_type" column.
    - x_var_name: Column name for the x-axis variable.
    - y_var_name: Column name for the y-axis variable.
    - color_map: Dictionary mapping labels to their respective colors for the legend.
    - title: Title of the plot (default: "Scatter Plot" with underscores).
    - save_local: If True, saves the plot as a PNG file.
    - show_legend: If True, shows the legend (default is True).
    """
    plt.figure(figsize=(10, 6))

    # Use colors directly from the dataframe, default to white if missing
    colors = df["color_snow_type"].fillna("white")

    plt.scatter(df[x_var_name], df[y_var_name], c=colors, alpha=0.7, edgecolors="k")

    # Add labels and title
    plt.xlabel(x_var_name.replace("_", " "))
    plt.ylabel(y_var_name.replace("_", " "))
    plt.title(title.replace("_", " "))

    # Show legend if show_legend is True
    if show_legend:
        handles = [plt.Line2D([0], [0], marker='o', color=color, markersize=8, label=label) 
               for label, color in color_map.items()]
        plt.legend(handles=handles, title="Predominant Snow Type", loc='lower right', bbox_to_anchor=(0.95, 0.05))

    # Show or save plot
    if save_local:
        plt.savefig(f"charts/{title}.png", bbox_inches='tight')
        

    plt.show()

def plot_scatter_w_R2(df, x_var_name, y_var_name, color_map, title="Scatter_Plot", save_local=True, show_legend=True):
    """
    Creates a scatter plot of specified x and y variables, colored by Predominant Snow Type.
    Adds a best-fit line and displays the R-squared value.

    Parameters:
    - df: DataFrame containing the x and y variables and "Snow_Type_Color" column.
    - x_var_name: Column name for the x-axis variable.
    - y_var_name: Column name for the y-axis variable.
    - color_map: Dictionary mapping labels to their respective colors for the legend.
    - title: Title of the plot (default: "Scatter Plot" with underscores).
    - save_local: If True, saves the plot as a PNG file.
    - show_legend: If True, shows the legend (default is True).
    """
    plt.figure(figsize=(10, 6))

    # Use colors directly from the dataframe, default to white if missing
    colors = df["color_snow_type"].fillna("white")

    plt.scatter(df[x_var_name], df[y_var_name], c=colors, alpha=0.7, edgecolors="k", label="Data Points")

    # Compute best-fit line
    x_values = df[x_var_name].values.reshape(-1, 1)
    y_values = df[y_var_name].values

    model = LinearRegression()
    model.fit(x_values, y_values)
    y_pred = model.predict(x_values)

    # Compute R-squared
    r2 = r2_score(y_values, y_pred)

    # Plot best-fit line
    plt.plot(df[x_var_name], y_pred, color="red", linestyle="--", linewidth=2, label=f"Best Fit Line (R²={r2:.2f})")

    # Add labels and title
    plt.xlabel(x_var_name.replace("_", " "))
    plt.ylabel(y_var_name.replace("_", " "))
    plt.title(title.replace("_", " "))

    # Annotate with R-squared value
    plt.text(0.05, 0.9, f"R² = {r2:.2f}", transform=plt.gca().transAxes, fontsize=12, color="red", bbox=dict(facecolor="white", alpha=0.7))

    # Show legend if show_legend is True
    if show_legend:
        handles = [plt.Line2D([0], [0], marker='o', color=color, markersize=8, label=label) 
               for label, color in color_map.items()]
        plt.legend(handles=handles, title="Predominant Snow Type", loc='lower right', bbox_to_anchor=(0.95, 0.05))


    # Show or save plot
    if save_local:
        plt.savefig(f"charts/{title}.png", bbox_inches='tight')

    plt.show()



def pairwise_welch_t_test(grouped_data):
    """
    Performs Welch's t-test for unequal variances on all pairwise comparisons 
    of groups in grouped_data.

    Parameters:
    - grouped_data: DataFrame with a 'mean' and 'std' column, indexed by group.

    Returns:
    - A DataFrame with columns "Group1", "Group2", and "P-Value".
    """
    results = []

    # Get unique groups
    groups = grouped_data.index

    # Perform pairwise Welch's t-test
    for group1, group2 in combinations(groups, 2):
        # Extract data for each group
        mean1, std1, n1 = grouped_data.loc[group1, ['mean', 'std', 'count']]
        mean2, std2, n2 = grouped_data.loc[group2, ['mean', 'std', 'count']]

        # Compute Welch's t-test
        t_stat, p_value = ttest_ind_from_stats(mean1, std1, n1, mean2, std2, n2, equal_var=False)

        # Append results as a row in the list
        results.append({'Group1': group1, 'Group2': group2, 'P-Value': p_value})

    # Convert the results list to a DataFrame
    return pd.DataFrame(results)

def plot_boxplot_by_2group(df, parameter, title, groupby_column_1, groupby_column_2, color_map=None, category_order=None, trunc=False, save_local=True):
    """
    Plot a boxplot for the given parameter grouped by two specified columns.
    
    Parameters:
    - df: DataFrame containing the data.
    - parameter: The name of the parameter to be plotted (as a column in the DataFrame).
    - title: The title of the plot.
    - groupby_column_1: The first column by which to group the data (determines left-to-right order).
    - groupby_column_2: The second column for creating side-by-side box plots within each group of groupby_column_1.
    - color_map: A dictionary mapping groupby_column_2 categories to colors. Defaults to None.
    - category_order: Optional list of categories in the desired order for plotting groupby_column_1.
    - trunc: If True, rotates labels 90 degrees and truncates them to 15 characters.
    - save_local: If True, saves the plot as a PNG file in the 'charts/' directory.
    """
    
    # Define a default color map if none is provided
    if color_map is None:
        unique_categories = sorted(df[groupby_column_2].unique())
        default_palette = sns.color_palette("twilight", len(unique_categories))
        color_map = dict(zip(unique_categories, default_palette))

    # If category_order is not provided, use the sorted unique values from groupby_column_1
    if category_order is None:
        category_order = sorted(df[groupby_column_1].unique())

    # Apply truncation if enabled
    if trunc:
        category_order = [cat[:15] for cat in category_order]

    # Create the boxplot
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x=groupby_column_1, y=parameter, hue=groupby_column_2, 
                palette=color_map, order=category_order, dodge=True)

    # Set the title and labels
    plt.title(title)
    plt.xlabel(groupby_column_1)
    plt.ylabel(parameter)

    # Adjust x-axis labels based on truncation
    if trunc:
        plt.xticks(rotation=90)  # Rotate labels 90 degrees
    else:
        plt.xticks(rotation=0)  # Keep labels horizontal

    # Move the legend outside the plot for better visibility
    plt.legend(title=groupby_column_2, bbox_to_anchor=(1.05, 1), loc='upper left')

    # Ensure layout is clean
    plt.tight_layout()

    # Save the plot locally if save_local is True
    if save_local:
        os.makedirs("charts", exist_ok=True)  # Ensure the directory exists
        safe_title = "".join(c if c.isalnum() or c in (" ", "-", "_") else "_" for c in title)  # Remove invalid filename characters
        plt.savefig(f"charts/{safe_title}.png", bbox_inches='tight')

    # Show the plot
    plt.show()

  


    