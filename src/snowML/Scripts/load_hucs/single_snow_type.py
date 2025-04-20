"Model Optimized Per Snow Type"

# DEFINE SOME CONSTANTS 

import random
import json
from snowML.datapipe import snow_types as st
from snowML.datapipe import get_geos as gg

# Define constants

HUC_MARITIME = [17020009, 17020011, 17110005, 17110006,  17110009]
HUC_MONTANE =  [17010304, 17010302, 17060207, 17060208]
HUC_MIXED = [17030001, 17030002, 17110008]
HUC_EPHEMERAL = [17030003, 17020010, 17110007]
HUC_ALL =  HUC_MARITIME + HUC_MIXED + HUC_EPHEMERAL + HUC_MONTANE  
INPUT_PAIRS = [[huc, '12'] for huc in HUC_ALL]

TRAIN_SIZE_FRACTION = 0.6
VAL_SIZE_FRACTION = 0.2
TEST_SIZE_FRACTION = 0.2



# Excluded Hucs Due to Missing SWE Data (Canada)
EXCLUDED_HUCS_CA = ["1711000501", "1711000502", "1711000503", "171100050101", "171100050102", \
                "171100050201", "171100050202", "171100050203", "171100050301", \
                "171100050302", "171100050303", "171100050304", "171100050305", \
                "171100050306", "171100050401"] 
# Plus Six Inadvertently (?) Excluded Hucs
EXCLUDED_HUCS_ADDTL = ['171100060305', '171100060401', '171100060402',
                        '171100060403', '171100060404', '171100060405',
                        '171100060406']
EXCLUDED_HUCS = EXCLUDED_HUCS_CA + EXCLUDED_HUCS_ADDTL



def assemble_huc_list(snow_type, input_pairs = INPUT_PAIRS):
    hucs = []
    for pair in input_pairs:
        geos = gg.get_geos(pair[0], pair[1])
        _, _, df_predom = st.process_all(pair[0], pair[1])
        valid_hucs = df_predom.loc[df_predom["Predominant_Snow"] == snow_type, "huc_id"]
        geos_filtered = geos[geos["huc_id"].isin(valid_hucs)]
        hucs.extend(geos_filtered["huc_id"].to_list())
    # Filter out excluded hucs
    hucs = [huc for huc in hucs if huc not in EXCLUDED_HUCS]
    return hucs




def split_by_huc(hucs, train_size_frac, val_size_frac):
    """
    Splits a list of Hydrologic Unit Codes (HUCs) into training, validation, 
    and test sets.

    Args:
        hucs (list): List of HUC IDs to be split.
        train_size_frac (float): Fraction of the HUCs to be used for the 
            training set.
        val_size_frac (float): Fraction of the HUCs to be used for the 
            validation set.

    Returns:
        tuple: A tuple containing three lists:
            - train_hucs (list): List of HUC IDs for the training set.
            - val_hucs (list): List of HUC IDs for the validation set.
            - test_hucs (list): List of HUC IDs for the test set.
    """
    random.shuffle(hucs)  # Shuffle the HUC IDs randomly
    train_end = int(len(hucs) * train_size_frac)
    val_end = train_end + int(len(hucs) * val_size_frac)
    train_hucs = hucs[:train_end]
    val_hucs = hucs[train_end:val_end]
    test_hucs = hucs[val_end:]
    return train_hucs, val_hucs, test_hucs

def select_hucs(input_pairs=INPUT_PAIRS, f_out="hucs_data_maritime.json"):
    hucs = assemble_huc_list(input_pairs)
    train_hucs, val_hucs, test_hucs = split_by_huc(hucs, TRAIN_SIZE_FRACTION, VAL_SIZE_FRACTION)
   # Combine the lists into a dictionary
    hucs_dict = {
        "train_hucs": train_hucs,
        "val_hucs": val_hucs,
        "test_hucs": test_hucs
    }
    # Save the dictionary to a single JSON file
    with open(f_out, "w", encoding="utf-8") as json_file:
        json.dump(hucs_dict, json_file, indent=2)

    return train_hucs, val_hucs, test_hucs


