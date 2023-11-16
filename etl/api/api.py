from fastapi import FastAPI, HTTPException
import sqlite3
import logging
#from ..logger import CustomFormatter
import os
from pydantic import BaseModel
from typing import Any

#create instance called app
app=FastAPI()

    
logger = logging.getLogger(os.path.basename(__file__))
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
#ch.setFormatter(CustomFormatter())
logger.addHandler(ch)

# Defining a function to open a connection to our SQLite database
def get_db():
    db = sqlite3.connect('temp.db')
    return db

@app.get("/")
async def root():
    return {"message": "Initializing"}



@app.get("/get_data/{CustomerID}")
async def get_record(CustomerID: int):
    # Open a connection to the database
    with get_db() as db:
        cursor = db.cursor()
        # Executing a SQL query to fetch the data
        cursor.execute(f"SELECT * FROM CustomerMetrics WHERE CustomerID = {CustomerID}")
        record = cursor.fetchone()

    if record is None:
        return {"error": "Record not found"}

    # Converting the record to a dictionary 
    column_names = [description[0] for description in cursor.description]
    record_dict = dict(zip(column_names, record))
    return record_dict


#Defining a Pydantic model for the data
class CreatemovieRequest(BaseModel):
    CustomerID: int
    ChurnStatus: str
    StateID: int
    PlanID: int
    DayUsageID: int
    EveUsageID: int
    NightUsageID: int
    IntlUsageID: int
    CustomerServiceCalls: int


# Columns: ['CustomerID', 'ChurnStatus', 'StateID', 'PlanID', 'DayUsageID', 'EveUsageID', 'NightUsageID', 'IntlUsageID', 'CustomerServiceCalls']

@app.post("/create_data")
async def create_record(new_data: CreatemovieRequest):
    try:
        # Opening a database connection using the get_db function
        db = get_db()
        cursor = db.cursor()

        # Defining the SQL query to insert data into the table
        insert_query = """
        INSERT INTO CustomerMetrics (
            CustomerID, ChurnStatus, StateID, PlanID, DayUsageID, EveUsageID, NightUsageID ,IntlUsageID, CustomerServiceCalls
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """

        # Executing the insert query with the data from the new_data parameter
        cursor.execute(insert_query, (
            new_data.CustomerID, new_data.ChurnStatus, new_data.StateID,
            new_data.PlanID, new_data.DayUsageID, new_data.EveUsageID,
            new_data.NightUsageID, new_data.IntlUsageID, new_data.CustomerServiceCalls
        ))

        # Committing the transaction to save the data in the database
        db.commit()

        return {"message": "Record created successfully"}
    except Exception as e:
        logger.error(f"Failed to insert data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to insert data: {str(e)}")
    finally:
        # Closing the database connection in the 'finally' block
        db.close()


class UpdateRecordRequest(BaseModel):
    column_name: str  # The type that column gets
    new_value: Any # As it can be both integer and str, defined as Any. Later can find better solution
    CustomerID: int  # The type that the CustomerID is

@app.put("/update_data")
async def update_record(update_request: UpdateRecordRequest):
    try:
        # Opening a database connection using the get_db function
        db = get_db()
        cursor = db.cursor()

        # Defining the SQL query to update the specified column for the given record ID
        update_query = f"UPDATE CustomerMetrics SET {update_request.column_name} = ? WHERE CustomerID = ?"

        # Executing the update query with the new value and record ID
        cursor.execute(update_query, (update_request.new_value, update_request.CustomerID))

        # Committing the transaction to save the data in the database
        db.commit()

        return {"message": "Record updated successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update data: {str(e)}")
    finally:
        # Closing the database connection in the 'finally' block
        db.close()


# Columns: ['CustomerID', 'ChurnStatus', 'StateID', 'PlanID', 'DayUsageID', 'EveUsageID', 'NightUsageID', 'IntlUsageID', 'CustomerServiceCalls']
# import sqlite3

# # Replace 'your_file.db' with the actual name of your .db file
# db_file = 'temp.db'
# table_name = 'CustomerMetrics'  # Replace with the actual name of the table

# # Connect to the SQLite database
# connection = sqlite3.connect(db_file)
# cursor = connection.cursor()

# # Check if the table exists
# cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}';")
# table_exists = cursor.fetchone()

# if table_exists:
#     # Fetch and print the contents of the specified table
#     cursor.execute(f"SELECT * FROM {table_name};")
#     rows = cursor.fetchall()

#     # Print column names
#     column_names = [description[0] for description in cursor.description]
#     print("Columns:", column_names)

#     # Print data
#     print(f"Data in the '{table_name}' table:")
#     for row in rows:
#         print(row)
# else:
#     print(f"The table '{table_name}' does not exist in the database.")

# # Close the connection
# connection.close()





# @app.get("/get_info/")
# def get_info(ID:int):
#     """THis code is to get the id of a person from the data """
#     info=data[data.id==ID].to_dict('records')
#     return (info)



# import sqlite3

# # Replace 'your_file.db' with the actual name of your .db file
# db_file = 'temp.db'

# # Connect to the SQLite database
# connection = sqlite3.connect(db_file)
# cursor = connection.cursor()

# # Fetch the list of tables
# cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
# tables = cursor.fetchall()

# # Print the list of tables
# print("Tables in the database:")
# for table in tables:
#     print(table[0])

# # Close the connection
# connection.close()
