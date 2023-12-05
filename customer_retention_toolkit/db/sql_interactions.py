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
    """
    A class used to interact with a SQLite database.

    Attributes:
        cnxn (sqlite3.Connection): The SQLite connection object.
        cursor (sqlite3.Cursor): The SQLite cursor object for executing SQL commands.
        dbname (str): The name of the database.
        table_name (str): The name of the table to interact with.

    Methods:
        close_cnxn(): Commits changes and closes the database connection.
        get_table_columns(): Retrieves the column names of the specified table.
        truncate_table(): Deletes all data from the specified table.
        drop_table(): Deletes the specified table from the database.
        insert_many(df): Inserts multiple records into the specified table from a pandas DataFrame.
        from_sql_to_pandas(chunksize, order_by): Fetches data from the specified table and returns a pandas DataFrame.
        update_table(condition, update_values): Updates records in the specified table based on a condition.
    """

    def __init__(self, dbname: str, table_name: str) -> None:
        """
        Initializes the SqlHandler with a specific database and table.

        Args:
            dbname (str): The name of the database.
            table_name (str): The name of the table to interact with.
        """
        self.cnxn = sqlite3.connect(f'{dbname}.db')
        self.cursor = self.cnxn.cursor()
        self.dbname = dbname
        self.table_name = table_name

    def close_cnxn(self) -> None:
        """
        Commits changes and closes the database connection.
        """
        logger.info('Committing the changes')
        self.cnxn.commit()
        self.cnxn.close()
        logger.info('The connection has been closed')

    def get_table_columns(self) -> list:
        """
        Retrieves the column names of the specified table.

        Returns:
            list: A list of column names of the table.
        """
        self.cursor.execute(f"PRAGMA table_info({self.table_name});")
        columns = self.cursor.fetchall()
        column_names = [col[1] for col in columns]
        logger.info(f'The list of columns: {column_names}')
        return column_names
    
    def truncate_table(self) -> None:
        """
        Deletes all data from the specified table.
        """
        query = f"DELETE FROM {self.table_name};"
        self.cursor.execute(query)
        logger.info(f'The {self.table_name} is truncated')

    def drop_table(self) -> None:
        """
        Deletes the specified table from the database.
        """
        query = f"DROP TABLE IF EXISTS {self.table_name};"
        self.cursor.execute(query)
        logger.info(f"Table '{self.table_name}' deleted.")

    def insert_many(self, df: pd.DataFrame) -> None:
        """
        Inserts multiple records into the specified table from a pandas DataFrame.

        Args:
            df (pd.DataFrame): A pandas DataFrame containing the records to be inserted.
        """
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
        """
        Fetches data from the specified table and returns a pandas DataFrame.

        Args:
            chunksize (int, optional): The number of rows to fetch at a time. Fetches all rows if not specified.
            order_by (str, optional): SQL ORDER BY clause to sort the results.

        Returns:
            pd.DataFrame: A pandas DataFrame containing the fetched data.
        """
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
        """
        Updates records in the specified table based on a condition.

        Args:
            condition (str): The SQL condition to apply for the update.
            update_values (dict): A dictionary of column-value pairs to update.
        """
        set_clause = ", ".join([f"{key} = ?" for key in update_values.keys()])
        query = f"UPDATE {self.table_name} SET {set_clause} WHERE {condition};"
        self.cursor.execute(query, list(update_values.values()))
        logger.info('Data updated successfully')
