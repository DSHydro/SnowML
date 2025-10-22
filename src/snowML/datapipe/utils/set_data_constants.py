# pylint: disable=C0103
"""
This module provides utility functions for creating dictionaries that map bucket 
types to bucket names and variable abbreviations to their full names. 

Functions:
    create_bucket_dict(b_type): Creates a dictionary mapping 
        bucket types to bucket names based on the specified type.
    create_var_dict(): Creates a dictionary mapping 
        variable abbreviations to their full names.
        variable abbreviations to their full names.
"""


def create_bucket_dict(b_type):
    """
    Creates a dictionary mapping bucket types to bucket names based on the specified type.

    Args:
        b_type (str): The type of bucket configuration. Must be one of "prod", "test", or "sue-private".

    Returns:
        dict: A dictionary where the keys are bucket types and the values are bucket names.

    Raises:
        AssertionError: If `b_type` is not one of the permitted types ("prod", "test", "sue-private").
    """
    permitted_types = ["prod", "test", "sue-private"]
    assert b_type in permitted_types, f"Type must be one of {permitted_types}"
    bucket_types = ["shape-bronze", "bronze", "silver", "gold", "model-ready"]
    if b_type == "prod":
        bucket_names = ["snowml-shape", "snowml-bronze", "snowml-silver", "snowml-gold", "snowml-model-ready"]
    elif b_type == "test":
         bucket_names = ["datapipe-demo"] * len(bucket_types)
    else: 
        bucket_names = ["snowy-data-demo"] * len(bucket_types)
    
    bucket_dict = dict(zip(bucket_types, bucket_names))
    return bucket_dict

def create_var_dict():
    """
    Creates a dictionary mapping variable abbreviations to their full names.

    Returns:
        dict: A dictionary where the keys are variable abbreviations (e.g., "pr", "tmmn") 
              and the values are the corresponding full names (e.g., "precipitation_amount", 
              "air_temperature").
    """
    var_list = ["pr", "tmmn", "tmmx", "vs", "srad", "rmax", "rmin", "swe"]
    var_names = ["precipitation_amount", "air_temperature", "air_temperature", \
                 "wind_speed", "surface_downwelling_shortwave_flux_in_air",  \
                  "relative_humidity", "relative_humidity", "SWE"]
    var_dict = dict(zip(var_list, var_names))
    return var_dict
