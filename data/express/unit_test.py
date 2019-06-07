import json
import sqlite3

conn = sqlite3.connect("dqmio.sqlite")
c = conn.cursor()

# c.execute('SELECT Count(*) FROM monitorelements WHERE name = "RPC/AllHits/SummaryHistograms/Occupancy_for_Endcap"')
c.execute('SELECT * FROM monitorelements WHERE name = "RPC/AllHits/SummaryHistograms/Occupancy_for_Endcap"')


# row = 13
col_value = 6

table_rows = c.fetchall()
# print(table_rows)

for row in range(len(table_rows)):
    value = table_rows[row][col_value]
    dictionary = json.loads(value)
    # print(json.dumps(dictionary, indent=4, sort_keys=True))
    print(table_rows[row][0])
    print(dictionary['fArray'])