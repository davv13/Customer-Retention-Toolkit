import pandas as pd
import random
from customer_retention_toolkit.db.schema import *
from customer_retention_toolkit.db.sql_interactions import  SqlHandler
from customer_retention_toolkit.logger import *
from FillTables import *
from customer_retention_toolkit.models.MLWorkflow import MLWorkflow
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
# if __name__ == "__main__":
    
#     create_database()
#     InsertToTables()
#     run_ml_workflow()

# Example usage
# Configure the logger
logger = logging.getLogger(os.path.basename(__file__))
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(CustomFormatter())
logger.addHandler(ch)

if __name__ == '__main__':
    create_database()
    InsertToTables()
    dbname = 'temp'
    workflow = MLWorkflow(dbname)

    # Run the workflow
    metrics, X_test, y_test, best_model_predictions = workflow.run_workflow()

    # Log model metrics
    for model_name, model_metrics in metrics.items():
        logger.info(f"{model_name} Metrics:")
        for metric_name, metric_value in model_metrics.items():
            logger.info(f"{metric_name}: {metric_value}")

    # Save predictions to the database
    workflow.save_predictions_to_db(X_test, y_test, best_model_predictions, table_name='PredictionResults')
    

   

