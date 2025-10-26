""" Module to load data direct to silver bucket """

# pylint: disable=C0103
import time
import pandas as pd
from snowML.datapipe.utils import data_utils as du
from snowML.datapipe.utils import set_data_constants as sdc
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

BUCKET_DICT = sdc.create_bucket_dict("prod")
B = BUCKET_DICT["silver"]

def load_current_data(save_ttl, bucket = B):
    f = save_ttl + ".csv"
    df = du.s3_to_df(f, bucket)
    return df


def process_list (huc_list, save_ttl, var_name, tif_path, bucket = B):
    time_start = time.time()

    # initialize existing df and huc_processed list
    f= save_ttl + ".csv"
    if du.isin_s3(bucket, f):
        #print("loading existing file")
        existing_df = load_current_data(save_ttl, bucket = bucket)
        processed_hucs = list(existing_df["huc_id"])
    else:
        existing_df = pd.DataFrame()
        processed_hucs = []

    results_df = existing_df
    tot = len(huc_list)
    count = 0

    for huc_id in huc_list:
        count += 1
        #print(f"processing huc {count} of {tot}")

        if int(huc_id) in processed_hucs:
            #print("already exists")
            pass
        else:
            huc_lev = str(len(str(huc_id))).zfill(2)
            if var_name == "geos":
                row = gg.get_geos_with_name(huc_id, huc_lev)
                row = row[["huc_id", "huc_name", "geometry"]]
            elif var_name == "snow_types":
                _, _, df_predom = st.process_all(huc_id, huc_lev)
                row = df_predom[["huc_id", "Predominant_Snow"]]
            elif var_name == "dem":
                elev = dem.process_dem_all(huc_id, huc_lev)
                row = pd.DataFrame({"huc_id": [huc_id], "Mean Elevation": [elev]})
            elif var_name == "forest_cover":
                row = cover.forest_cover_huc(huc_id, tif_path)
            else:
                print("unknown variable")
                row = pd.DataFrame()
            results_df = pd.concat([results_df, row], ignore_index=True)
    results_df.set_index("huc_id", inplace=True)
    results_df.index = results_df.index.astype(str)
    results_df.sort_index(inplace=True)
    du.dat_to_s3(results_df, bucket, save_ttl, file_type="csv")
    #du.elapsed(time_start)
    return results_df

def get_region_ls(region = 17):
    huc8 = gg.get_geos(region, '08')
    ls = list(huc8['huc_id'])
    print(f"There are {len(ls)} huc08 units to process")
    return ls

def drop_CA_hucs(geos):
    ca_rows = geos[geos["huc_name"] == "Canada"]
    geos_small = geos[geos["huc_name"] != "Canada"]
    excluded_hucs = ca_rows["huc_id"].tolist()
    return geos_small, excluded_hucs


def process_regions(region_ls, var_name, region = 17, bucket =B):
    save_ttl = SAVE_DICT[var_name]
    save_ttl = save_ttl + "_" + str(region)
    all_canada_hucs = []
    all_error_hucs = []
    count = 0
    for reg in region_ls:
        count += 1
        print(f"processing region no {count} : {reg}")
        geos = gg.get_geos_with_name(reg, '12')
        huc_list = list(geos["huc_id"])
        error_hucs, canada_hucs = process_single_hucs(huc_list, var_name, bucket = bucket)
        all_canada_hucs = all_canada_hucs + canada_hucs
        all_error_hucs = all_error_hucs + error_hucs

    print(f"The following {len(all_error_hucs)} had errors {all_error_hucs}")
    print(f"The following {len(all_canada_hucs)} hucs were excluded as being in Canada {all_canada_hucs}")
    return all_canada_hucs, all_error_hucs


def process_single_hucs(huc_ls, var_name, region = 17, bucket = B, tif_path = "notebooks/Land_Cover/nlcd_tcc_conus_2021_v2021-4.tif"):
    save_ttl = SAVE_DICT[var_name]
    save_ttl = save_ttl + "_" + str(region)
    error_hucs = []
    canada_hucs = []
    count = 0
    for huc in huc_ls:
        count += 1
        #print(f"processing huc no {count} : {huc}")
        geos = gg.get_geos_with_name(huc, '12')
        geos_small, excluded_hucs = drop_CA_hucs(geos)
        huc_list = list(geos_small["huc_id"])
        canada_hucs = canada_hucs + excluded_hucs
        try:
            process_list(huc_list, save_ttl, var_name, tif_path, bucket = bucket)
        except:
            error_hucs.append(huc)

    print(f"The following hucs had errors {error_hucs}")
    print(f"The following hucs were excluded as being in Canada {canada_hucs}")
    return error_hucs, canada_hucs

