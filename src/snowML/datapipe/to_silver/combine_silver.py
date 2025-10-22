""" Module to combine static variables for each huc into a single dataframe and store in S3 """

# pylint: disable=C0103

from snowML.datapipe.utils import data_utils as du
from snowML.datapipe.utils import set_data_constants as sdc


# DEFINE SOME CONSTANTS
VAR_NAMES  = ["Geos", "Snow_Types", "Dem", "Forest_Cover"]
VAR_NAMES_NO_GEO = ["Snow_Types", "Dem", "Forest_Cover"]
BUCKET_DICT = sdc.create_bucket_dict("prod")
B = BUCKET_DICT["silver"]

def load_df(var_name, region, bucket = B):
    f = f"{var_name}_{region}.csv"
    df = du.s3_to_df(f, bucket)
    return df

def combine(region = 17, bucket = B):
    results_df = None
    for var_name in VAR_NAMES:
        df = load_df(var_name, region=region, bucket = bucket)
        df.set_index("huc_id", inplace=True)

        if results_df is None:
            results_df = df
        else:
            results_df = results_df.merge(df, left_index=True, right_index=True, how="outer")
        print(results_df.head(2))

    f_out = f"Static_All_Region_{region}"
    du.dat_to_s3(results_df, bucket, f_out, file_type="csv")

    return results_df

def combine_no_geo(var_names = VAR_NAMES_NO_GEO, region = 17, bucket = B):
    results_df = None
    for var_name in var_names:
        df = load_df(var_name, region=region, bucket = bucket)
        df.set_index("huc_id", inplace=True)

        if results_df is None:
            results_df = df
        else:
            results_df = results_df.merge(df, left_index=True, right_index=True, how="outer")
    #print(results_df.head(2))

    f_out = f"Static_No_Geo_Region_{region}"
    du.dat_to_s3(results_df, bucket, f_out, file_type="csv")

    return results_df
