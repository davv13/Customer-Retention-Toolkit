from etl.db.schema import *
from etl.db.sql_interactions import  SqlHandler
from etl.logger import *
# from etl.db.FillTables import *
from FillTables import *

from etl.api import app
import pandas as pd
import random

# test.py
# data = pd.read_csv('telecom_data.csv')
if __name__ == "__main__":
    
    create_database()
    InsertToTables()
    
   


    