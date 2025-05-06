""" Module to load data direct to silver bucket """

# pylint: disable=C0103
import time
import pandas as pd
from snowML.datapipe.utils import data_utils as du
from snowML.datapipe.utils import get_geos as gg
from snowML.datapipe.utils import snow_types as st
from snowML.datapipe.utils import get_dem as dem
from snowML.datapipe.utils import forest_cover as cover


# DEFINE SOME CONSTANTS
SAVE_DICT = {
    "geos": "Geos", 
    "snow_types": "Snow_Types", 
    "dem": "Dem", 
    "forest_cover": "Forest_Cover", 
}

B = "snowml-silver" # Bucket Name.  To do, make dynamic

def load_current_data(save_ttl):
    f = save_ttl + ".csv"
    df = du.s3_to_df(f, B)
    return df


def process_list (huc_list, save_ttl, var_name, first = False):
    time_start = time.time()

    # initialize existing df and huc_processed list
    if not first:
        existing_df = load_current_data(save_ttl)
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
            huc_lev = str(len(str(huc_id))).zfill(2)
            if var_name == "geos":
                row = gg.get_geos_with_name(huc_id, huc_lev)
                row = row[["row_id", "huc_name", "geometry"]]
            elif var_name == "snow_types":
                _, _, df_predom = st.process_all(huc_id, huc_lev)
                row = df_predom[["huc_id", "Predominant_Snow"]]
            elif var_name == "dem":
                elev = dem.process_dem_all(huc_id, huc_lev)
                row = pd.DataFrame({"huc_id": [huc_id], "Mean Elevation": [elev]})
            elif var_name == "forest_cover":
                row = cover.forest_cover_huc(huc_id)
            else:
                print("unknown variable")
                row = pd.DataFrame()
            results_df = pd.concat([results_df, row], ignore_index=True)

    results_df.set_index("huc_id", inplace=True)
    results_df.index = results_df.index.astype(str)
    results_df.sort_index(inplace=True)
    print("time to save")
    du.dat_to_s3(results_df, B, save_ttl, file_type="csv")
    print("finished saving")
    du.elapsed(time_start)
    return results_df

def get_region_ls(region = 17):
    huc8 = gg.get_geos(region, '08')
    ls = list(huc8['huc_id'])
    print(f"There are {len(ls)} huc08 units to process")
    return ls

def process_regions(region_ls, var_name, region = 17, first = False):
    save_ttl = SAVE_DICT[var_name]
    save_ttl = save_ttl + "_" + str(region)
    error_regions = []
    count = 0
    for reg in region_ls:
        count += 1
        print(f"processing region no {count} : {reg}")
        geos = gg.get_geos(reg, '12')
        huc_list = list(geos["huc_id"])
        #try:
        results_df = process_list(huc_list, save_ttl, var_name, first = first)
        #except:
        #error_regions.append(reg)
    
    print(f"The following regions had errors {error_regions}")
    return error_regions
