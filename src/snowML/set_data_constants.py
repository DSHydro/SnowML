

def create_bucket_dict(b_type):
    permitted_types = ["prod", "test"]
    assert b_type in permitted_types, f"Type must be one of {permitted_types}"
    bucket_types = ["shape-bronze", "bronze", "silver", "gold", "model-ready"]
    if b_type == "prod":
        bucket_names = ["shape-bronze", "dawgs-bronze", "dawgs-silver", "dawgs-gold", "dawgs-model-ready"]
    else: # if b_type is not prod, then b_type == "test"
        bucket_names = ["sues-scratch"] * len(bucket_types)
    bucket_dict = dict(zip(bucket_types, bucket_names))
    return bucket_dict

def create_var_dict():
    var_list = ["pr", "tmmn", "tmmx", "vs", "srad", "rmax", "rmin", "swe"]
    var_names = ["precipitation_amount", "air_temperature", "air_temperature", \
                 "wind_speed", "surface_downwelling_shortwave_flux_in_air",  \
                  "relative_humidity", "relative_humidity", "SWE"]
    var_dict = dict(zip(var_list, var_names))
    return var_dict

# def create_f_out_dict():
#     f_out_types = ["bronze", "silver", "gold"]
#     f_out_patterns = ["{var}_all.zarr", "raw_{var}_in_{huc_id}.csv", "mean_{var}_in_{huc_id}.csv"]
#     f_out_dict = dict(zip(f_out_types, f_out_patterns))
#     return f_out_dict
