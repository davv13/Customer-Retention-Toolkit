from customer_retention_toolkit.db.schema import *
from customer_retention_toolkit.db.sql_interactions import  SqlHandler
from customer_retention_toolkit.logger import *
# from customer_retention_toolkit.db.FillTables import *
from FillTables import *

from customer_retention_toolkit.api import app
import pandas as pd
import random

# test.py
# data = pd.read_csv('telecom_data.csv')
if __name__ == "__main__":
    
    create_database()
    InsertToTables()
    
   


    