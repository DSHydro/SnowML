""" Module to select hucs for train, test, validate for Expirement2"""

import random
import json
from snowML.datapipe import snow_types as st
from snowML.datapipe import get_geos as gg

# Define constants

huc_maritime = [17020009, 17020011, 17110005, 17110006,  17110009]
huc_montane =  [17010304, 17010302, 17060207, 17060208]
huc_mixed = [17030001, 17030002, 17110008]
huc_ephemeral = [17030003, 17020010, 17110007]
huc_all = huc_montane + huc_maritime + huc_mixed+huc_ephemeral
INPUT_PAIRS = [[huc, '12'] for huc in huc_all]


# Excluded Hucs Due to Missing SWE Data (Canada)
EXCLUDED_HUCS_CA = ["1711000501", "1711000502", "1711000503", "171100050101", "171100050102", \
                "171100050201", "171100050202", "171100050203", "171100050301", \
                "171100050302", "171100050303", "171100050304", "171100050305", \
                "171100050306", "171100050401"] 
# Plus Six Inadvertently (?) Excluded Hucs
EXCLUDED_HUCS_ADDTL = ['171100060305', '171100060401', '171100060402', '171100060403', '171100060404', '171100060405', '171100060406']
EXCLUDED_HUCS = EXCLUDED_HUCS_CA + EXCLUDED_HUCS_ADDTL


def assemble_huc_list(input_pairs=INPUT_PAIRS):
    """
    Assembles a list of HUC (Hydrologic Unit Code) IDs from a list of input pairs.

    Args:
        input_pairs (list of tuples): A list of tuples where each tuple contains two elements:
            - The first element is a string representing the huc code for the region of interest.
            - The second element is a string or intrepresenting the lowest huc subunit to study.

    Returns:
        list: HUC IDs extracted from the geojson files corresponding to the input pairs.

    Example:
        input_pairs = [("RegionA", "10"), ("RegionB", "12")]
        huc_list = assemble_huc_list(input_pairs)
    """
    hucs = []
    for pair in input_pairs:
        geos = gg.get_geos(pair[0], pair[1])
        hucs.extend(geos["huc_id"].to_list())
    # Filter out excluded hucs
    hucs = [huc for huc in hucs if huc not in EXCLUDED_HUCS]
    return hucs

def find_nan_dataframes(df_dict):
    nan_dfs = []
    for key, df in df_dict.items():
        nan_count = df.isnull().sum().sum()
        if nan_count > 0:
            print(f"Warning: The DataFrame for '{key}' has {nan_count} NaN values.")
            nan_dfs.append(key)
    return nan_dfs

