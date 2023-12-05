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
ch.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(ch)

class MLWorkflow:
    def __init__(self, dbname):
        self.dbname = dbname
        self.model = RandomForestClassifier(random_state=42)

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
        exclude_columns = [col for col in df_comb.columns if 'ID' in col or col == 'CustomerID' or col == target]
        categorical_columns = df_comb.select_dtypes(include=['object']).columns.tolist()
        categorical_columns = [col for col in categorical_columns if col not in exclude_columns]
        df_comb = pd.get_dummies(df_comb, columns=categorical_columns, drop_first=True, dtype=int)
        return df_comb

    def split_data(self, df_comb, target='ChurnStatus'):
        X = df_comb.drop(target, axis=1)
        y = df_comb[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test

    def train_model(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        logger.info("Model trained successfully.")

    def evaluate_model(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        classification_rep = classification_report(y_test, y_pred)
        return {
            'Accuracy': accuracy,
            'Confusion Matrix': conf_matrix,
            'Classification Report': classification_rep
        }

    def predict(self, X):
        return self.model.predict(X)

    def run_workflow(self,table_names):
        # table_names = ['State', 'PlanDetails', 'DayUsage', 'EveUsage', 'NightUsage', 'IntlUsage', 'CustomerMetrics']
        data = self.load_data_from_db(table_names)
        data = self.preprocess_data(data)

        X_train, X_test, y_train, y_test = self.split_data(data)

        self.train_model(X_train, y_train)
        metrics = self.evaluate_model(X_test, y_test)
        self.trained_columns = X_train.columns.tolist()


        best_model_predictions = self.model.predict(X_test)
        return metrics, X_test, y_test, best_model_predictions

    def save_predictions_to_db(self, X_test, y_test, predictions, table_name):
        results_df = pd.DataFrame({
            'CustomerID': X_test['CustomerID'],
            'PredictedLabel': predictions,
            'ChurnStatus': y_test,
            'ModelName': ['Random Forest'] * len(predictions),
        })
        sql_handler = SqlHandler(dbname=self.dbname, table_name=table_name)
        sql_handler.insert_many(results_df)
        sql_handler.close_cnxn()
        logger.info(f"Predictions saved to {table_name}")

    def predict_for_customer(self, CustomerID):
            # Ensure that trained_columns is set
            if not hasattr(self, 'trained_columns'):
                raise Exception("The model is not trained yet. Please run the workflow first.")

            # Construct the SQL query to fetch all necessary customer data for prediction
            query = f"""
            SELECT cm.CustomerID, pd.AreaCode, pd.InternationalPlan, pd.VoiceMailPlan, pd.NumberVMailMessages,
                du.TotalDayMinutes, du.TotalDayCalls, du.TotalDayCharge,
                eu.TotalEveMinutes, eu.TotalEveCalls, eu.TotalEveCharge,
                nu.TotalNightMinutes, nu.TotalNightCalls, nu.TotalNightCharge,
                iu.TotalIntlMinutes, iu.TotalIntlCalls, iu.TotalIntlCharge,
                s.StateName
            FROM CustomerMetrics cm
            LEFT JOIN State s ON cm.StateID = s.StateID
            LEFT JOIN PlanDetails pd ON cm.PlanID = pd.PlanID
            LEFT JOIN DayUsage du ON cm.DayUsageID = du.DayUsageID
            LEFT JOIN EveUsage eu ON cm.EveUsageID = eu.EveUsageID
            LEFT JOIN NightUsage nu ON cm.NightUsageID = nu.NightUsageID
            LEFT JOIN IntlUsage iu ON cm.IntlUsageID = iu.IntlUsageID
            WHERE cm.CustomerID = {CustomerID}
            """

            # Execute the query and fetch the data
            with sqlite3.connect(f'{self.dbname}.db') as conn:
                df_customer = pd.read_sql_query(query, conn)

            # Check if customer data was found
            if df_customer.empty:
                return None, "Customer not found"

            # Preprocess the customer data to match training data
            df_customer_processed = self.preprocess_data(df_customer)

            # Align customer data columns with trained columns
            for col in self.trained_columns:
                if col not in df_customer_processed.columns:
                    df_customer_processed[col] = 0  # Add missing columns with a default value of 0

            # Ensure the columns are in the same order as during training
            df_customer_processed = df_customer_processed[self.trained_columns]

            # Make a prediction using the trained model
            prediction = self.model.predict(df_customer_processed)

            # Return the prediction
            return prediction[0], None  # Assuming prediction is a numpy array and you want the first element