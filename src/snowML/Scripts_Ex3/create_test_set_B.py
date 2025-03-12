""" Module to create and save huc_list for Test_Set B """
# pylint: disable=C0103

import pandas as pd
from snowML.datapipe import get_geos as gg
from snowML.datapipe import snow_types as st

# Define Constants
NACHES = 17030002
UPPER_YAKIMA = 17030001


def get_testB_huc_list():
    """ get the sub_basins, names, and geos for Test Set B"""
    geos1 = gg.get_geos_with_name(NACHES, '12')
    geos2 = gg.get_geos_with_name(UPPER_YAKIMA, '12')
    geos_all = pd.concat([geos1, geos2])
    print(f"Before exclude ephemeral, number of Huc12 units is {geos_all.shape[0]}")

    # Add snow type data and exlcude ephemeral basins
    _, _, df_predom1 = st.process_all(NACHES, '12')
    _, _, df_predom2 = st.process_all(UPPER_YAKIMA, '12')
    df_snow_types_testB = pd.concat([df_predom1, df_predom2])
    testB_df = df_snow_types_testB[["huc_id", "Predominant_Snow"]]
    testB_df = testB_df[testB_df["Predominant_Snow"] != "Ephemeral"]
    print(testB_df.shape)

    # Extract just the hucs
    testB_hucs = list(testB_df["huc_id"])
    return testB_hucs
