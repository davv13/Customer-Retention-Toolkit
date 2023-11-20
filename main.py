import pandas as pd
import random
from customer_retention_toolkit.db.schema import *
from customer_retention_toolkit.db.sql_interactions import  SqlHandler
from customer_retention_toolkit.logger import *
from FillTables import *
from run_ml import run_ml_workflow
from customer_retention_toolkit.api import app
from customer_retention_toolkit.models.model_final import (
    load_data_from_db,
    preprocess_data,
    split_data,
    train_decision_tree,
    train_random_forest,
    train_logistic_regression,
    evaluate_model,
    save_predictions_to_db  
)



# test.py
# data = pd.read_csv('telecom_data.csv')
if __name__ == "__main__":
    
    create_database()
    InsertToTables()
    run_ml_workflow()
   


    