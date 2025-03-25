""" Script to update gold files with newly released data """

from snowML.datapipe import bronze_to_gold as btg
from snowML.datapipe import get_geos as gg
from snowML.datapipe import set_data_constants as sdc

# DEFINE CONSTANTS 
HUC_MARITIME = [17020009, 17020011, 17110005, 17110006,  17110009]
HUC_MONTANE =  [17010304, 17010302, 17060207, 17060208]
HUC_MIXED = [17030001, 17030002, 17110008]
HUC_EPHEMERAL = [17030003, 17020010, 17110007]
HUC_ALL = HUC_MONTANE + HUC_MARITIME + HUC_MIXED + HUC_EPHEMERAL

def update(): 
    for huc in HUC_ALL[0:2]: 
        var_list = sdc.create_var_dict()
        geos = gg.get_geos(huc, '12')
        for var in var_list: 
            if var != "swe": 
                btg.process_geos(geos, var, max_wk = 4, append_start = "2024-01-01")




    