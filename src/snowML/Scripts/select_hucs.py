""" Module to select hucs for train, test, validate for multi-huc expirement"""

import random
import json
from snowML.datapipe import snow_types as st
from snowML.datapipe import get_geos as gg

# Define constants

huc_maritime = [17020009, 17110006, 17110005, 17110009, 17020011]
huc_montane =  [17060207, 17010304, 17010302, 17060208]
huc_all = huc_montane + huc_maritime
INPUT_PAIRS = [[huc, '12'] for huc in huc_all]

TRAIN_SIZE_FRACTION = 0.6
VAL_SIZE_FRACTION = 0.2
TEST_SIZE_FRACTION = 0.2
EXLCUDED_EPHEMERAL = True
# Excluded Hucs Due to Missing SWE Data (Canada)
EXCLUDED_HUCS = ["1711000501", "1711000502", "1711000503", "171100050101", "171100050102", \
                "171100050201", "171100050202", "171100050203", "171100050301", \
                "171100050302", "171100050303", "171100050304", "171100050305", \
                "171100050306", "171100050401"] 


def assemble_huc_list(input_pairs):
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
        if EXLCUDED_EPHEMERAL is True:
            geos = snowclass_filter(geos)
            #print(f"filtered shape for {pair[0]} is {geos.shape}")
        hucs.extend(geos["huc_id"].to_list())
    # Filter out excluded hucs
    hucs = [huc for huc in hucs if huc not in EXCLUDED_HUCS]
    return hucs

# function that filters geos to exlcude hucs where ephemeral is >= 50
def snowclass_filter(geos):
    df_snow_types = st.snow_class(geos)
    # Filter huc_ids where Ephemeral < 50
    valid_huc_ids = df_snow_types.loc[df_snow_types["Ephemeral"] < 50, "huc_id"]
    geos_filtered = geos[geos["huc_id"].isin(valid_huc_ids)]
    return geos_filtered


def split_by_huc(hucs, train_size_frac, val_size_frac):
    random.shuffle(hucs)  # Shuffle the HUC IDs randomly
    train_end = int(len(hucs) * train_size_frac)
    val_end = train_end + int(len(hucs) * val_size_frac)
    train_hucs = hucs[:train_end]
    val_hucs = hucs[train_end:val_end]
    test_hucs = hucs[val_end:]
    return train_hucs, val_hucs, test_hucs

def select_hucs(input_pairs, f_out="hucs_data.json"):
    hucs = assemble_huc_list(input_pairs)
    train_hucs, val_hucs, test_hucs = split_by_huc(hucs, TRAIN_SIZE_FRACTION, VAL_SIZE_FRACTION)
   # Combine the lists into a dictionary
    hucs_dict = {
        "train_hucs": train_hucs,
        "val_hucs": val_hucs,
        "test_hucs": test_hucs
    }
    # Save the dictionary to a single JSON file
    with open(f_out, "w") as json_file:
        json.dump(hucs_dict, json_file, indent=2)

    return train_hucs, val_hucs, test_hucs

def find_nan_dataframes(df_dict):
    nan_dfs = []
    for key, df in df_dict.items():
        nan_count = df.isnull().sum().sum()
        if nan_count > 0:
            print(f"Warning: The DataFrame for '{key}' has {nan_count} NaN values.")
            nan_dfs.append(key)
    return nan_dfs
