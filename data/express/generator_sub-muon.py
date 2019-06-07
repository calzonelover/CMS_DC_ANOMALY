import json
import sqlite3
import h5py
import numpy as np

conn = sqlite3.connect("dqmio.sqlite")
c = conn.cursor()

# "CSC/CSCOfflineMonitor/Occupancy/hOStripsAndWiresAndCLCT",
# "RPC/AllHits/SummaryHistograms/Occupancy_for_Barrel",
# "RPC/AllHits/SummaryHistograms/Occupancy_for_Endcap",

c.execute(
    """
    SELECT * FROM monitorelements 
    WHERE name = "RPC/AllHits/SummaryHistograms/Occupancy_for_Endcap"
    """
    )

col_value = 6

table_rows = c.fetchall()
x_data = []

for row in range(len(table_rows)):
    value = table_rows[row][col_value]
    dictionary = json.loads(value)
    x_data.append(np.divide(dictionary['fArray'], 100000.0))

f = h5py.File('RPC_normalize.hdf5', 'w')

dset = f.create_dataset("ENDCAP", data=x_data)

f.close()