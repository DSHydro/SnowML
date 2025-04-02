""" Moduel that loads and returns train, validate, and test hucs from a saved json file"""
# pylint: disable=C0103

import json


def huc_split(f="src/snowML/Scripts/hucs_data.json"):
    """
    Splits HUCs (Hydrologic Unit Codes) data into training, validation, and 
    test sets.

    Args:
        f (str): The file path to the JSON file containing HUCs data. 
                 Defaults to "src/snowML/Scripts_Ex3/hucs_data.json".

    Returns:
        tuple: A tuple containing three lists:
            - tr (list): List of training HUCs.
            - val (list): List of validation HUCs.
            - te (list): List of test HUCs.
    """
    with open(f, 'r', encoding='utf-8') as file:
        data = json.load(file)
    tr = data["train_hucs"]
    val = data["val_hucs"]
    te = data["test_hucs"]
    return tr, val, te