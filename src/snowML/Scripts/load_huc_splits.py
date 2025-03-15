""" Moduel that loads and returns train, validate, and test hucs from a saved json file"""

import json


def huc_split(f="src/snowML/Scripts/hucs_data.json"):
    with open(f, 'r') as file:
        data = json.load(file)
    tr = data["train_hucs"]
    #print(tr[0:4])
    val = data["val_hucs"]
    #print(val[0:4])
    te = data["test_hucs"]
    #print(te[0:4])
    return tr, val, te
    