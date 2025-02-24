from snowML.datapipe import get_geos as gg
from snowML.datapipe import snow_types as st

def create_bucket_dict(b_type):
    permitted_types = ["prod", "test"]
    assert b_type in permitted_types, f"Type must be one of {permitted_types}"
    bucket_types = ["shape-bronze", "bronze", "gold", "model-ready"]
    if b_type == "prod":
        bucket_names = ["snowml-shape", "snowml-bronze",  "snowml-gold", "snowml-model-ready"]
    else: # if b_type is not prod, then b_type == "test"
        bucket_names = ["sues-test"] * len(bucket_types)
    bucket_dict = dict(zip(bucket_types, bucket_names))
    return bucket_dict

def create_var_dict():
    var_list = ["pr", "tmmn", "tmmx", "vs", "srad", "rmax", "rmin", "swe"]
    var_names = ["precipitation_amount", "air_temperature", "air_temperature", \
                 "wind_speed", "surface_downwelling_shortwave_flux_in_air",  \
                  "relative_humidity", "relative_humidity", "SWE"]
    var_dict = dict(zip(var_list, var_names))
    return var_dict

import pandas as pd

def count_huc_types():
    hucs = [17010301, 17010302, 17010303, 17010304,
            17020009, 17020010, 17020011,
            17030001, 17030002, 17030003,
            17110005, 17110006, 17110007, 17110008]
    
    names = []
    Maritime_counts = []
    Ephemeral_counts = []
    Montane_counts = []
    Prairie_counts = []
    for huc in hucs:
        print(f"processing huc{huc}")
        name = gg.get_geos_with_name(huc, '08').iloc[0, 2]  # To do - make '08' dynamic
        names.append(name)  
        
        df, counts, predom = st.process_all(huc, '12')  # To do - make '12' dynamic
        counts = predom["Predominant_Snow"].value_counts()
        
        Maritime_counts.append(counts.get("Maritime", 0))
        Ephemeral_counts.append(counts.get("Ephemeral", 0))
        Montane_counts.append(counts.get("Montane Forest", 0))
        Prairie_counts.append(counts.get("Prairie", 0))

    # Create a DataFrame with hucs as an index and the respective columns
    df = pd.DataFrame({
        "huc_id": hucs,
        "name": names,
        "Maritime_counts": Maritime_counts,
        "Ephemeral_counts": Ephemeral_counts,
        "Montane_counts": Montane_counts,
        "Prairie_counts": Prairie_counts
    }).set_index("huc_id")  # Set huc_id as the index
    
    return df
