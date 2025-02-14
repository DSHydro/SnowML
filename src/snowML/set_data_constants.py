

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

