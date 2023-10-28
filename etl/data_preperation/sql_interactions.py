import sqlite3
import logging 
import pandas as pd
import numpy as np
import os
from ..logger import CustomFormatter

logger = logging.getLogger(os.path.basename(__file__))
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(CustomFormatter())
logger.addHandler(ch)

class SqlHandler:

    def __init__(self, dbname:str, table_name:str) -> None:
        self.cnxn = sqlite3.connect(f'{dbname}.db')
        self.cursor = self.cnxn.cursor()
        self.dbname = dbname
        self.table_name = table_name

    def close_cnxn(self) -> None:
        logger.info('commiting the changes')
        self.cnxn.commit()
        self.cnxn.close()
        logger.info('the connection has been closed')

    def get_table_columns(self) -> list:
        self.cursor.execute(f"PRAGMA table_info({self.table_name});")
        columns = self.cursor.fetchall()
        column_names = [col[1] for col in columns]
        logger.info(f'the list of columns: {column_names}')
        return column_names
    
    def truncate_table(self) -> None:
        query = f"DELETE FROM {self.table_name};"
        self.cursor.execute(query)
        logger.info(f'the {self.table_name} is truncated')

    def drop_table(self) -> None:
        query = f"DROP TABLE IF EXISTS {self.table_name};"
        self.cursor.execute(query)
        logger.info(f"table '{self.table_name}' deleted.")

    def insert_many(self, df: pd.DataFrame) -> None:
        df = df.replace(np.nan, None)
        df.rename(columns=lambda x: x.lower(), inplace=True)
        columns = list(df.columns)
        sql_column_names = [i.lower() for i in self.get_table_columns()]
        columns = list(set(columns) & set(sql_column_names))
        ncolumns = ','.join(['?'] * len(columns))
        data_to_insert = df[columns].values.tolist()
        query = f"INSERT INTO {self.table_name} ({','.join(columns)}) VALUES ({ncolumns});"
        self.cursor.executemany(query, data_to_insert)
        logger.info('Data inserted successfully')

    def from_sql_to_pandas(self, chunksize: int = None, order_by: str = None) -> pd.DataFrame:
        if chunksize:
            offset = 0
            dfs = []
            while True:
                query = f"SELECT * FROM {self.table_name}"
                if order_by:
                    query += f" ORDER BY {order_by}"
                query += f" LIMIT {chunksize} OFFSET {offset}"
                data = pd.read_sql_query(query, self.cnxn)
                if data.empty:
                    break
                dfs.append(data)
                offset += chunksize
            return pd.concat(dfs, ignore_index=True)
        else:
            return pd.read_sql_query(f"SELECT * FROM {self.table_name}", self.cnxn)

    def update_table(self, condition: str, update_values: dict) -> None:
        set_clause = ", ".join([f"{key} = ?" for key in update_values.keys()])
        query = f"UPDATE {self.table_name} SET {set_clause} WHERE {condition};"
        self.cursor.execute(query, list(update_values.values()))
        logger.info('Data updated successfully')


   
        



