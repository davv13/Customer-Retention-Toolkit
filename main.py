from customer_retention_toolkit.db.schema import *
from customer_retention_toolkit.db.sql_interactions import  SqlHandler
from customer_retention_toolkit.logger import *
# from customer_retention_toolkit.db.FillTables import *
from FillTables import *
from customer_retention_toolkit.models.Prediction_Model import *
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
from run_ml import run_ml_workflow
from customer_retention_toolkit.api import app
import pandas as pd
import random

# test.py
# data = pd.read_csv('telecom_data.csv')
if __name__ == "__main__":
    
    create_database()
    InsertToTables()
    run_ml_workflow()
    # ML workflow
    # dbname = 'temp'
    # table_name = 'PredictionResults'

    # data = load_data_from_db(dbname)
    # data = preprocess_data(data)
    # print("Columns in data after preprocessing:", data.columns)
    # X_train, X_test, y_train, y_test = split_data(data)

    # models = {
    #     'Decision Tree': train_decision_tree(X_train, y_train),
    #     'Random Forest': train_random_forest(X_train, y_train),
    #     'Logistic Regression': train_logistic_regression(X_train, y_train)
    # }

    # model_metrics = {}
    # for model_name, model in models.items():
    #     accuracy, conf_matrix, classification_rep = evaluate_model(model, X_test, y_test)
    #     model_metrics[model_name] = {
    #         'Accuracy': accuracy,
    #         'Confusion Matrix': conf_matrix,
    #         'Classification Report': classification_rep
    #     }
    #     print(f"{model_name} Metrics:")
    #     print("Accuracy:", accuracy)
    #     print("Confusion Matrix:\n", conf_matrix)
    #     print("Classification Report:\n", classification_rep)

    # best_model_name = max(model_metrics, key=lambda k: model_metrics[k]['Accuracy'])
    # best_model = models[best_model_name]

    # best_model_predictions = best_model.predict(X_test)
    # save_predictions_to_db(X_test, best_model_predictions, best_model_name, dbname, table_name)
   


    