""" Moduel to get snowtypes for a set of HUCS and save into the specified S3 gold bucket"""


# pylint: disable=C0103

import pandas as pd
import time
from snowML.datapipe import data_utils as du
from snowML.datapipe import snow_types as st
from snowML.datapipe import get_geos as gg


def load_current_snow_type_data(save_ttl):
    b = "snowml-gold"  # TO DO - make dynamic
    f = save_ttl + ".csv"
    df = du.s3_to_df(f, b)
    return df


def snow_types_all (huc_list, save_ttl = "Snow_Types", first = False):
    time_start = time.time()
    
    # initialize existing df and huc_processed list
    if not first:
        existing_df = load_current_snow_type_data(save_ttl)
        processed_hucs = list(existing_df["huc_id"])
    else:
        existing_df = pd.DataFrame()
        processed_hucs = []

    results_df = existing_df
    tot = len(huc_list)
    count = 0
    for huc_id in huc_list:
        count += 1
        print(f"processing huc {count} of {tot}")
        if int(huc_id) in processed_hucs:
            print("already exists")
        else:
            _, _, df_predom = st.process_all(huc_id, str(len(str(huc_id))).zfill(2))
            df_predom = df_predom[["huc_id", "Predominant_Snow"]]
            results_df = pd.concat([results_df, df_predom], ignore_index=True)
    results_df.set_index("huc_id", inplace=True)
    results_df.index = results_df.index.astype(str)
    results_df.sort_index(inplace=True)
    if save_ttl is not None:
        b = "snowml-gold"  # TO DO - make dynamic
        du.dat_to_s3(results_df, b, save_ttl, file_type="csv")
    du.elapsed(time_start)
    return results_df


def get_region_ls(region = 17):
    huc8 = gg.get_geos(region, '08')
    ls = list(huc8['huc_id'])
    print(f"There are {len(ls)} huc08 units to process") 
    return ls 

def process_regions(region_ls, save_ttl = "Snow_Types_Region", first = False): 
    error_regions = []
    count = 0
    for region in region_ls: 
        count += 1
        print(f"processing region no {count} : {region}")
        geos = gg.get_geos(region, '12')
        huc_list = list(geos["huc_id"])
        try: 
            snow_types_all(huc_list, save_ttl = save_ttl, first = first)
        except: 
            error_regions.append(region)
    print(f"The following regions had errors {error_regions}")
    return error_regions

    
    





              
    return 3