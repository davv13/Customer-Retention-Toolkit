from fastapi import FastAPI, HTTPException
import sqlite3
import pandas as pd
import logging
#from ..logger import CustomFormatter
import os
from pydantic import BaseModel
from typing import Any
from ..models.MLWorkflow import MLWorkflow
dbname = 'temp'
ml_workflow = MLWorkflow(dbname)

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

# ... (existing API code) ...

# Endpoint to predict churn based on CustomerID
@app.get("/predict_churn/{CustomerID}")
async def predict_churn(CustomerID: int):
    try:
        # Fetch data for the specific customer
        with get_db() as db:
            cursor = db.cursor()
            cursor.execute(f"SELECT * FROM CustomerMetrics WHERE CustomerID = {CustomerID}")
            customer_record = cursor.fetchone()

        if customer_record is None:
            raise HTTPException(status_code=404, detail="Customer not found")

        # Convert the record to a DataFrame
        column_names = [description[0] for description in cursor.description]
        customer_data = pd.DataFrame([customer_record], columns=column_names)

        # Preprocess the data as per your MLWorkflow requirements
        # Note: Adjust the preprocessing if needed
        preprocessed_data = ml_workflow.preprocess_data(customer_data)

        # Make a prediction
        prediction = ml_workflow.predict(preprocessed_data)

        return {"CustomerID": CustomerID, "ChurnPrediction": prediction[0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

