
"""Module that creates a list of Huc12 units from selected Huc08 Basins"""

from snowML.datapipe import get_geos as gg

def make_list():
    hucs = []
    for huc in [17030003, 17110007]:
        geos = gg.get_geos(huc, '12')
        new_hucs = list(geos["huc_id"])
        print(f"for {huc} there are {geos.shape[0]} subhucs")
        hucs = hucs + new_hucs
    return hucs
