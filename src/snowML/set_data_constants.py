

def create_bucket_dict(b_type):
    permitted_types = ["prod", "test"]
    assert b_type in permitted_types, f"Type must be one of {permitted_types}"
    bucket_types = ["shape-bronze", "swe-bronze",  "swe-silver", "swe-gold", \
                "wrf-bronze", "wrf-silver", "wrf-gold", "model-ready", "sues-test"]
    if b_type == "prod":
        bucket_names = bucket_types
        bucket_dict = dict(zip(bucket_types, bucket_names))
        bucket_dict['model-ready'] = 'dawgs-model-ready'
        return bucket_dict
    # if b_type is not prod, then b_type == "test"
    bucket_names = ["sues-scratch"] * len(bucket_types)
    bucket_dict = dict(zip(bucket_types, bucket_names))
    bucket_dict['swe-bronze'] = 'swe-bronze'
    bucket_dict['wrf-bronze'] = 'wrf-bronze'
    return bucket_dict

def create_var_dict():
    #var_list = ["swe", "pr", "tmmn", "sph"]
    #var_names = ["SWE", "precipitation_amount", "air_temperature", "daily_mean_specific_humidity"]
    var_list = ["swe", "pr", "tmmn"]
    var_names = ["SWE", "precipitation_amount", "air_temperature"]
    var_dict = dict(zip(var_list, var_names))
    return var_dict

def create_f_out_dict():
    f_out_types = ["bronze", "silver", "gold"]
    f_out_patterns = ["{var}_all.zarr", "raw_{var}_in_{huc_id}.csv", "mean_{var}_in_{huc_id}.csv"]
    f_out_dict = dict(zip(f_out_types, f_out_patterns))
    return f_out_dict
