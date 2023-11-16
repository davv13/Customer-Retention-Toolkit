#this code is used to create updated_telecom_data.csv to fill db

import sqlite3
import pandas as pd
from etl.db.sql_interactions import SqlHandler

dbname = 'temp'
table_names = ['State', 'PlanDetails', 'DayUsage', 'EveUsage', 'NightUsage', 'IntlUsage']
cnxn = sqlite3.connect(f'{dbname}.db')

dataframes = {}

for table_name in table_names:
    sql_handler = SqlHandler(dbname=dbname, table_name=table_name)
    df = sql_handler.from_sql_to_pandas()
    dataframes[table_name] = df
    print(f"Data for {table_name}:")
    # print(df.head())  # Display the first few rows of each DataFrame

# Now, dataframes dictionary contains a DataFrame for each table
combined_df = pd.concat(dataframes.values(), axis=1)
orig = pd.read_csv('telecom_data.csv')
metrics = orig.loc[:,['CustomerServiceCalls','ChurnStatus']]
final = pd.concat([combined_df,metrics],axis =1)
# print(final.head())
final.to_csv('updated_telecom_data.csv',index=False)
# combined_df.to_csv('updated_telecom_data.csv',index=False)

# Display the combined DataFrame
# print(combined_df.head())