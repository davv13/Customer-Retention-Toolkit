from fastapi import FastAPI, HTTPException
import sqlite3
import pandas as pd
import logging
import os
from pydantic import BaseModel
from typing import Any
from ..models.MLWorkflow import MLWorkflow

app = FastAPI()

logger = logging.getLogger(os.path.basename(__file__))
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
# ch.setFormatter(CustomFormatter())
logger.addHandler(ch)

def get_db():
    """
    Creates and returns a connection to the SQLite database.

    Returns:
        sqlite3.Connection: The connection to the SQLite database.
    """
    db = sqlite3.connect('temp.db')
    return db

@app.get("/")
async def root():
    """
    Root endpoint for the FastAPI application.

    Returns:
        dict: A dictionary with a welcome message.
    """
    return {"message": "Initializing"}

@app.get("/get_data/{CustomerID}")
async def get_record(CustomerID: int):
    """
    Fetches a record from the CustomerMetrics table based on CustomerID.

    Args:
        CustomerID (int): The ID of the customer to fetch.

    Returns:
        dict: A dictionary containing the customer's data or an error message if not found.
    """
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

class UserRequest(BaseModel):
    """
    Pydantic model representing a user request for creating a record in the CustomerMetrics table.

    Attributes:
        CustomerID (int): The customer's ID.
        ChurnStatus (str): The churn status of the customer.
        StateID (int): The state ID associated with the customer.
        PlanID (int): The plan ID associated with the customer.
        DayUsageID (int): The day usage ID associated with the customer.
        EveUsageID (int): The evening usage ID associated with the customer.
        NightUsageID (int): The night usage ID associated with the customer.
        IntlUsageID (int): The international usage ID associated with the customer.
        CustomerServiceCalls (int): The number of customer service calls made by the customer.
    """
    CustomerID: int
    ChurnStatus: str
    StateID: int
    PlanID: int
    DayUsageID: int
    EveUsageID: int
    NightUsageID: int
    IntlUsageID: int
    CustomerServiceCalls: int

@app.post("/create_data")
async def create_record(new_data: UserRequest):
    """
    Creates a new record in the CustomerMetrics table.

    Args:
        new_data (UserRequest): The data to be inserted into the table.

    Returns:
        dict: A message indicating the success or failure of the operation.
    """
    try:
        # Opening a database connection using the get_db function
        db = get_db()
        cursor = db.cursor()

        # Defining the SQL query to insert data into the table
        insert_query = """
        INSERT INTO CustomerMetrics (
            CustomerID, ChurnStatus, StateID, PlanID, DayUsageID, EveUsageID, NightUsageID ,IntlUsageID, CustomerServiceCalls
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
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
    """
    Pydantic model representing a request to update a record in the CustomerMetrics table.

    Attributes:
        column_name (str): The name of the column to be updated.
        new_value (Any): The new value to be set for the column.
        CustomerID (int): The ID of the customer whose record is to be updated.
    """
    column_name: str
    new_value: Any
    CustomerID: int

@app.put("/update_data")
async def update_record(update_request: UpdateRecordRequest):
    """
    Updates a record in the CustomerMetrics table based on the provided update request.

    Args:
        update_request (UpdateRecordRequest): The request containing the update details.

    Returns:
        dict: A message indicating the success or failure of the operation.
    """
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

@app.get("/predict_churn/{CustomerID}")
async def predict_churn(CustomerID: int):
    """
    Predicts churn for a given customer ID using a machine learning workflow.

    Args:
        CustomerID (int): The ID of the customer for whom churn is to be predicted.

    Returns:
        dict: A dictionary containing the CustomerID and the churn prediction.
    """
    dbname = 'temp'
    ml_workflow = MLWorkflow(dbname)
    ml_workflow.run_workflow(['State', 'PlanDetails', 'DayUsage', 'EveUsage', 'NightUsage', 'IntlUsage', 'CustomerMetrics']) 
    try:
        # Use MLWorkflow to predict churn for the given CustomerID
        prediction, error = ml_workflow.predict_for_customer(CustomerID)

        if error:
            raise HTTPException(status_code=404, detail=error)

        return {"CustomerID": CustomerID, "ChurnPrediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
