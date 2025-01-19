

def create_bucket_dict(type):
    permitted_types = ["prod", "test"]
    assert type in permitted_types, f"Type must be one of {permitted_types}"
    bucket_types = ["shape-bronze", "swe-bronze",  "swe-silver", "swe-gold", \
                "wrf-bronze", "wrf-silver", "wrf-gold", "model-ready", "sues-test"]
    if type == "prod": 
        bucket_names = bucket_types
        bucket_dict = dict(zip(bucket_types, bucket_names))
        bucket_dict['model-ready'] = 'dawgs-model-ready'
        return bucket_dict
    else: # type == "test"
        bucket_names = ["sues-scratch"] * len(bucket_types)
        bucket_dict = dict(zip(bucket_types, bucket_names))
        bucket_dict['swe-bronze'] = 'swe-bronze'
        bucket_dict['wrf-bronze'] = 'wrf-bronze'        
        return bucket_dict

def create_var_dict():
    vars = ["swe", "pr", "tmmn"] 
    var_names = ["SWE", "precipitation_amount", "air_temperature"]   
    var_dict = dict(zip(vars, var_names))
    return var_dict

def create_f_out_dict():
    f_out_types = ["bronze", "silver", "gold"]
    f_out_patterns = ["{var}_all.zarr", "raw_{var}_in_{huc_id}", "mean_{var}_in_{huc_id}"]
    f_out_dict = dict(zip(f_out_types, f_out_patterns)) 
    return f_out_dict


