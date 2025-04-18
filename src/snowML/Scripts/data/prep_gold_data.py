""" Script to process gold data for new hucs """

from snowML.datapipe import data_utils as du
from snowML.datapipe import set_data_constants as sdc
from snowML.datapipe import bronze_to_gold as btg
from snowML.datapipe import get_geos as gg

def process_gold(huc_list): 
    var_dict = sdc.create_var_dict()
    var_list = list(var_dict.keys())
    for huc in huc_list: 
        geos = gg.get_geos(huc, '12')
        for var in var_list: 
            btg.process_geos(geos, var, max_wk =4)

