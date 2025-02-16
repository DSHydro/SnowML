import pandas as pd 
import mlflow
import os
from snowML import snow_types as st
import matplotlib.pyplot as plt

input_pairs = [[17020009, '12'], [17110005, '12'], [17030002, '12']]

# exp0 
desc1 = "2 epochs, no snow type data" 
uri_1 = "mlflow-artifacts:/549974439488203583/7761d31b7c31466cacbe9fcd23c4d16e/artifacts/results.csv"

# exp 2 (unleashed stork)
desc2 =  "2 epochs and snow type data" 
uri_2 = "mlflow-artifacts:/549974439488203583/984cb9981ded4b8f9b802f542db8f020/artifacts/results.csv"

# exp3 (youthful hog)
desc3 =  "10 epochs and includes snow type data" 
uri_3 = "mlflow-artifacts:/549974439488203583/1c23001dbbb84b8f8f4d2728ba226067/artifacts/results.csv"

#exp4 (delicate roo)
desc4 = "10 epochs and includes snow type data & two layers"
uri_4 = "mlflow-artifacts:/549974439488203583/d49c3c889ebb40a18dae2cd69cef6cdf/artifacts/results.csv"

uris = [uri_1, uri_2, uri_3, uri_4]
desc = [desc1, desc2, desc3, desc4]
results_dict = dict(zip(desc, uris))    

def load_ml_metrics(artifact_uri): 
    mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")
    path = mlflow.artifacts.download_artifacts(artifact_uri)
    df = pd.read_csv(path)
    df.rename(columns={df.columns[0]: "huc_id"}, inplace=True)
    df.set_index("huc_id", inplace=True)
    #print(df.head())
    os.remove(path)
    return df

def summarize_metrics(df):
    print(df.describe())
    #return df.describe()

def load_snow_type_data(input_pairs):
    snow_types = pd.DataFrame()
    for pair in input_pairs:
        huc_id = pair[0]
        huc_lev = pair[1]
        df, _ = st.process_all(huc_id, huc_lev)
        df.set_index("huc_id", inplace=True)
        snow_types = pd.concat([snow_types, df])
    return snow_types 

def plot_kge(df_merged, x_var, y_var):
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
    plt.title("Scatter Plot of test_kge vs Ephemeral")
    
    # Show plot
    return plt

# print results 
for desc, uri in results_dict.items():
    print(f"Summary for {desc}:")
    df = load_ml_metrics(uri)
    summarize_metrics(df)
    df_class = load_snow_type_data(input_pairs)
    df.index = df.index.astype(str)
    df_class.index = df_class.index.astype(str)
    df_all = df_merged = df.merge(df_class, left_index=True, right_index=True, how="inner")
    plt = plot_kge(df_all)
    plt.savefig(f"{desc}.png")
    


