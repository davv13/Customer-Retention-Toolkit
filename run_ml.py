# Import necessary modules
from customer_retention_toolkit.db.schema import *
from customer_retention_toolkit.db.sql_interactions import SqlHandler
from customer_retention_toolkit.logger import *
from FillTables import *
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
from customer_retention_toolkit.api import app
import pandas as pd
import random

def run_ml_workflow(dbname='temp', table_name='PredictionResults'):
    # Load and preprocess data
    table_names = ['State', 'PlanDetails', 'DayUsage', 'EveUsage', 'NightUsage', 'IntlUsage', 'CustomerMetrics']
    data = load_data_from_db(dbname,table_names)
    data = preprocess_data(data)
    print("Columns in data after preprocessing:", data.columns)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = split_data(data,target="ChurnStatus")

    # Train models
    models = {
        'Decision Tree': train_decision_tree(X_train, y_train),
        'Random Forest': train_random_forest(X_train, y_train),
        'Logistic Regression': train_logistic_regression(X_train, y_train)
    }

    # Evaluate models and select the best one
    model_metrics = {}
    for model_name, model in models.items():
        accuracy, conf_matrix, classification_rep = evaluate_model(model, X_test, y_test)
        model_metrics[model_name] = {
            'Accuracy': accuracy,
            'Confusion Matrix': conf_matrix,
            'Classification Report': classification_rep
        }
        print(f"{model_name} Metrics:")
        print("Accuracy:", accuracy)
        print("Confusion Matrix:\n", conf_matrix)
        print("Classification Report:\n", classification_rep)

    best_model_name = max(model_metrics, key=lambda k: model_metrics[k]['Accuracy'])
    best_model = models[best_model_name]

    # Make predictions with the best model
    best_model_predictions = best_model.predict(X_test)

    # Save predictions to database
    save_predictions_to_db(X_test, y_test,best_model_predictions, best_model_name, dbname, table_name)

    return model_metrics