import json
import sqlite3

def main():
    conn = sqlite3.connect("express_2018.sqlite")
    c = conn.cursor()

    # c.execute('SELECT Count(*) FROM monitorelements WHERE name = "RPC/AllHits/SummaryHistograms/Occupancy_for_Endcap"')
    # c.execute('SELECT * FROM monitorelements WHERE name = "RPC/AllHits/SummaryHistograms/Occupancy_for_Endcap"')
    c.execute('SELECT fromrun, torun, fromlumi, tolumi, value, name FROM monitorelements WHERE name = "RPC/AllHits/SummaryHistograms/Occupancy_for_Endcap"')

    # row = 13
    col_value = 6

    table_rows = c.fetchall()

    for row in range(20):
        print(table_rows[row][5])
        print(table_rows[row][0])
        print(table_rows[row][1])
        print(table_rows[row][2])
        print(table_rows[row][3])
        print(json.loads(table_rows[row][4])['fArray'], len(json.loads(table_rows[row][4])['fArray']))
        print("\n\n\n")
#         value = table_rows[row][col_value]
#         print(value)
#         dictionary = json.loads(value)
#         print(dictionary.keys())
#         input()

if __name__ == '__main__':
    main()