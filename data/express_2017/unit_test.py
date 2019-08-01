import os
import json
import sqlite3
import pandas as pd

from data.express_2017 import settings as express_2017_settings

def main():
    conn = sqlite3.connect(os.path.join(express_2017_settings.RAW_DATA_DIRECTORY, express_2017_settings.SQLITE_RAW_DATA_NAME))
    c = conn.cursor()

    c.execute('SELECT fromrun, torun, fromlumi, tolumi, value, name FROM monitorelements WHERE name = "RPC/AllHits/SummaryHistograms/Occupancy_for_Endcap"')

    table_rows = c.fetchall()
    print(len(table_rows))