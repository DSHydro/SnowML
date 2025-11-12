""" Module With Helper Functions For Dashboard"""

# pylint: disable=C0103

import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import geopandas as gpd
from snowML.datapipe.utils import data_utils as du
from snowML.datapipe.utils import get_geos as gg



def is_valid_huc8(value):
    return value.isdigit() and len(value) == 8

@st.cache_data
def cached_region_geos():
    url = "s3://snowml-dashboard/R17_huc8.geo.parquet"
    R17 = gpd.read_parquet(url)
    return R17

@st.cache_data
def cached_get_geos(huc12):
    """Get geospatial data with caching."""
    url = f"s3://snowml-dashboard/{huc12}_huc12.geo.parquet"
    geos = gpd.read_parquet(url)
    return geos 

@st.cache_data
def cached_get_data(huc):

    """Get model-ready S3 data with caching."""

    # Get UA Model Ready Data
    f_UA = f"model_ready_huc{huc}.csv"
    bucket_name = "snowml-model-ready"
    if not du.isin_s3(bucket_name, f_UA):
        df_UA =  None
    else:
        df_UA = du.s3_to_df(f_UA, bucket_name)
        df_UA.set_index("day", inplace=True)

    # Get UCLA Model Ready Data
    f_UCLA = f"model_ready_huc{huc}_ucla.csv"
    if not du.isin_s3(bucket_name, f_UCLA):
        df_UCLA = None
    else:
        df_UCLA = du.s3_to_df(f_UCLA, bucket_name)
        df_UCLA.set_index("day", inplace=True)

    return df_UA, df_UCLA

def slim_df(df):
    df_slim = df[["mean_swe"]]
    return df_slim

def make_plot_df(df_UA, df_UCLA):
    df_UA_slim = slim_df(df_UA)
    if df_UCLA is None:
        return df_UA_slim
    df_UCLA_slim = slim_df(df_UCLA)
    df_UCLA_slim = df_UCLA_slim.rename(columns={"mean_swe": "mean_swe_UCLA"})
    df_joined = df_UA_slim.join(df_UCLA_slim, how="inner")
    return df_joined

# Cached SWE plot
@st.cache_data
def cached_plot_swe(df_UA, df_UCLA, huc12):
    """
    Generate the SWE plot for a given HUC12 and cache it.
    Returns a matplotlib Figure.
    """

    df_joined = make_plot_df(df_UA, df_UCLA)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df_joined.index, df_joined["mean_swe"], c='b', label="Mean_SWE_UAData")
    if df_UCLA is not None: 
        ax.plot(df_joined.index, df_joined["mean_swe_UCLA"], c='black', label="Mean_SWE_UCLA_Data")
    ax.set_ylim(0, 2)
    ax.legend()
    ax.set_xlabel('Date')
    ax.set_ylabel('SWE (m)')
    ax.set_title(f"Mean SWE for HUC {huc12}")

    # Format x-axis to show only years
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    plt.setp(ax.get_xticklabels(), rotation=45, ha="center")
    fig.tight_layout()

    return fig






