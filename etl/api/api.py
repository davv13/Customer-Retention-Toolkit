import pandas as pd
import numpy as np
from fastapi import FastAPI

#Read the data

data=pd.read_csv('telecom_data.csv')

#create instance called app
app=FastAPI()

@app.get('/')
def read_root():
    return{"message":"Hello, World"}

@app.get("/get_info/")
def get_info(ID:int):
    """THis code is to get the id of a person from the data """
    info=data[data.id==ID].to_dict('records')
    return (info)

@app.get("/get_data/{record_id}")
async def get_record(record_id: int):
    record = data[data['ID'] == record_id].to_dict(orient='records')
    if len(record) == 0:
        return {"error": "Record not found"}
    return record[0]

# Update a specific record by ID
@app.put("/update_data/{record_id}")
async def update_record(record_id: int, new_data: dict):
    # Find the index of the record to be updated
    index = data[data['ID'] == record_id].index
    if len(index) == 0:
        raise HTTPException(status_code=404, detail="Record not found")
    
    # Update the record
    data.loc[index] = new_data

    # Save the updated data back to the CSV file
    data.to_csv('telecom_data.csv', index=False)

    return {"message": "Record updated successfully"}



# Delete a specific record by ID
@app.delete("/delete_data/{record_id}")
async def delete_record(record_id: int):
    # Find the index of the record to be deleted
    index = data[data['ID'] == record_id].index
    if len(index) == 0:
        raise HTTPException(status_code=404, detail="Record not found")

    # Delete the record
    data.drop(index, inplace=True)

    # Save the updated data back to the CSV file
    data.to_csv('telecom_data.csv', index=False)

    return {"message": "Record deleted successfully"}


# Create a new record
@app.post("/create_data")
async def create_record(new_data: dict):
    # Check if the required fields exist in the new_data dictionary
    required_fields = ['ID', 'name', 'age']  # Customize this based on your CSV structure
    for field in required_fields:
        if field not in new_data:
            raise HTTPException(status_code=400, detail=f"Missing required field: {field}")

    # Add the new record to the data DataFrame
    data = data.append(new_data, ignore_index=True)

    # Save the updated data back to the CSV file
    data.to_csv('telecom_data.csv', index=False)

    return {"message": "Record created successfully"}