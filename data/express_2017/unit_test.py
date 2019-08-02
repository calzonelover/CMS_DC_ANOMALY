import os
import re, json
import sqlite3
import numpy as np
import pandas as pd

import runregistry as rr

from data.express_2017 import settings as express_2017_settings

def main():
    conn = sqlite3.connect(os.path.join(express_2017_settings.RAW_DATA_DIRECTORY, express_2017_settings.SQLITE_RAW_DATA_NAME))
    c = conn.cursor()

    c.execute('SELECT fromrun, torun, fromlumi, tolumi, value, name FROM monitorelements')
    # c.execute('SELECT fromrun, torun, fromlumi, tolumi, value, name FROM monitorelements WHERE name = "RPC/AllHits/SummaryHistograms/Occupancy_for_Endcap"')

    table_rows = c.fetchall()

    print(len(table_rows))
    input()

    for row in range(len(table_rows)):
        n_extended_binsx, n_extended_grids = int(json.loads(table_rows[row][4])['fXaxis']['fNbins']+2), int(len(json.loads(table_rows[row][4])['fArray']))
        n_binsx, n_binsy = int(n_extended_binsx-2), int(n_extended_grids/n_extended_binsx-2)
        extended_data_vec = np.array(json.loads(table_rows[row][4])['fArray'])
        extended_data_grid = np.reshape(extended_data_vec, [int(n_extended_grids/n_extended_binsx), n_extended_binsx])
        data_grid = extended_data_grid[1:-1,1:-1]
        
        print(table_rows[row][5])
        print(table_rows[row][0])
        print(table_rows[row][1])
        print(table_rows[row][2])
        print(table_rows[row][3])
        print("\n")
        print(n_extended_binsx, n_extended_grids)
        print(n_binsx, n_binsy)
        print(extended_data_vec)
        print(extended_data_grid)
        print(data_grid)
        print("\n\n\n")