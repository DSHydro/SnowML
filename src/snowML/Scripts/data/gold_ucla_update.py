""" Script to update model_ready UCLA data """

from snowML.datapipe import to_model_ready_UCLA as ucla_mr
import importlib

importlib.reload(ucla_mr)

def update_all(huc_list, overwrite = False): 
    error_ls = []
    for huc in huc_list: 
        try: 
            ucla_mr.process_one_huc(huc, overwrite=overwrite) 
        except: 
           error_ls.append(huc)
    return error_ls