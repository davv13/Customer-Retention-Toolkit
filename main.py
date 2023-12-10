import pandas as pd
import random
from customer_retention_toolkit.db.schema import *
from customer_retention_toolkit.db.sql_interactions import  SqlHandler
from customer_retention_toolkit.logger import *
from FillTables import *
from customer_retention_toolkit.models.MLWorkflow import MLWorkflow
from customer_retention_toolkit.api import app

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
    metrics, X_test, y_test, best_model_predictions = workflow.run_workflow(['State', 'PlanDetails', 'DayUsage', 'EveUsage', 'NightUsage', 'IntlUsage', 'CustomerMetrics'])

    # Log model metrics
    logger.info("Random Forest Model Metrics:")
    for metric_name, metric_value in metrics.items():
        logger.info(f"{metric_name}: {metric_value}")

    # Save predictions to the database
    workflow.save_predictions_to_db(X_test, y_test, best_model_predictions, table_name='PredictionResults')    

   

