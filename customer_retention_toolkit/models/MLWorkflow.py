import pandas as pd
import os
import sqlite3
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from customer_retention_toolkit.db.sql_interactions import SqlHandler
from ..logger.logger import CustomFormatter,logging

# Configure logging
logger = logging.getLogger(os.path.basename(__file__))
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(CustomFormatter())
logger.addHandler(ch)

class MLWorkflow:
    def __init__(self, dbname):
        self.dbname = dbname
        self.models = {
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'Random Forest': RandomForestClassifier(random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42,max_iter=1000)
        }
        self.model_metrics = {}

    def load_data_from_db(self, table_names):
        cnxn = sqlite3.connect(f'{self.dbname}.db')
        dataframes = {}
        for table_name in table_names:
            df = pd.read_sql_query(f"SELECT * FROM {table_name}", cnxn)
            dataframes[table_name] = df
        cnxn.close()
        df_comb = pd.concat(dataframes.values(), axis=1)
        return df_comb

    def preprocess_data(self, df_comb, target='ChurnStatus'):
        exclude_columns = [col for col in df_comb.columns if 'ID' in col or col in ['CustomerID', target]]
        categorical_columns = df_comb.select_dtypes(include=['object']).columns.tolist()
        categorical_columns = [col for col in categorical_columns if col not in exclude_columns]
        df_comb = pd.get_dummies(df_comb, columns=categorical_columns, drop_first=True, dtype=int)
        return df_comb

    def split_data(self, df_comb, target='ChurnStatus'):
        X = df_comb.drop(target, axis=1)
        y = df_comb[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test

    def train_models(self, X_train, y_train):
        for model_name, model in self.models.items():
            model.fit(X_train, y_train)
            logger.info(f"{model_name} trained successfully.")

    def evaluate_models(self, X_test, y_test):
        model_metrics = {}
        for model_name, model in self.models.items():
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            conf_matrix = confusion_matrix(y_test, y_pred)
            classification_rep = classification_report(y_test, y_pred)
            model_metrics[model_name] = {
                'Accuracy': accuracy,
                'Confusion Matrix': conf_matrix,
                'Classification Report': classification_rep
            }
        return model_metrics

    def predict(self, X):
        # Assuming you want to use the best performing model for prediction
        # best_model_name = max(self.model_metrics, key=lambda k: self.model_metrics[k]['Accuracy'])
        # best_model = self.models[best_model_name]
        # return best_model.predict(X)
        


        # def predict(self, X):
        #     if not self.model_metrics:
        #         logger.warning("Model metrics are empty. Falling back to default model for prediction.")
        #         default_model = self.models['Logistic Regression']  # for example
        #     return default_model.predict(X)
        if not self.model_metrics:
            logger.warning("Model metrics are empty. Falling back to default model.")
            default_model = self.models.get('Logistic Regression')
            if default_model:
                return default_model.predict(X)
            else:
                logger.error("Default model is not available.")
                return None



    def run_workflow(self):
            # Load and preprocess data
        table_names = ['State', 'PlanDetails', 'DayUsage', 'EveUsage', 'NightUsage', 'IntlUsage', 'CustomerMetrics']
        data = self.load_data_from_db(table_names)
        data = self.preprocess_data(data)

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = self.split_data(data)

        self.train_models(X_train, y_train)
        metrics = self.evaluate_models(X_test, y_test)

        # Select the best model
        self.best_model_name = max(metrics, key=lambda k: metrics[k]['Accuracy'])
        self.best_model = self.models[self.best_model_name]

        # Make predictions with the best model
        best_model_predictions = self.best_model.predict(X_test)

        return metrics, X_test, y_test, best_model_predictions

    def save_predictions_to_db(self, X_test, y_test, predictions, table_name):
        results_df = pd.DataFrame({
            'CustomerID': X_test['CustomerID'],
            'PredictedLabel': predictions,
            'ChurnStatus': y_test,
            'ModelName': [self.best_model_name] * len(predictions),
        })
        sql_handler = SqlHandler(dbname=self.dbname, table_name=table_name)
        sql_handler.insert_many(results_df)
        sql_handler.close_cnxn()
        logger.info(f"Predictions saved to {table_name}")

