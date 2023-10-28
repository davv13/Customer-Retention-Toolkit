from etl.data_preperation.schema import *
from etl.data_preperation.sql_interactions import  SqlHandler
from etl.data_preperation.data_subseting import *
from etl.logger import *
import pandas as pd
import random

# test.py
data = pd.read_csv('telecom_data.csv')
if __name__ == "__main__":

    create_database()
    # print("Database created successfully!")  
    # print(data_State(data))