""" Module to combine static variabes for each huc into a single dataframe and store in S3 """

# pylint: disable=C0103

from snowML.datapipe.utils import data_utils as du


# DEFINE SOME CONSTANTS
VAR_NAMES  = ["Geos", "Snow_Types", "Dem", "Forest_Cover"]
B = "snowml-silver"  # TO DO - Make Dynamic

def load_df(var_name, region):
    f = f"{var_name}_{region}.csv"
    df = du.s3_to_df(f, B)
    return df

def combine(region = 17):
    results_df = None
    for var_name in VAR_NAMES:
        df = load_df(var_name, region=region)
        df.set_index("huc_id", inplace=True)

        if results_df is None:
            results_df = df
        else:
            results_df = results_df.merge(df, left_index=True, right_index=True, how="outer")
        print(results_df.head(2))

    f_out = f"Static_All_Region_{region}"
    du.dat_to_s3(results_df, B, f_out, file_type="csv")

    return results_df
