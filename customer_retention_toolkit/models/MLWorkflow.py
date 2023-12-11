import pandas as pd
import os
import sqlite3
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
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
    """
    A class for managing the machine learning workflow, including data loading,
    preprocessing, model training, evaluation, and prediction.

    Attributes:
        dbname (str): Database name to connect to for data loading.
        model (RandomForestClassifier): The machine learning model to be used.
        trained_columns (list): List of columns used for training the model.
    """
    def __init__(self, dbname: str):
        """
        Initializes the MLWorkflow class with a database name and model.

        Args:
            dbname (str): The name of the database to connect to.
        """    
        self.dbname = dbname
        self.model = RandomForestClassifier(random_state=42)

    def load_data_from_db(self, table_names: list) -> pd.DataFrame:
        """
        Loads data from a SQLite database from the given table names.

        Args:
            table_names (list): A list of table names to load data from.

        Returns:
            DataFrame: A pandas DataFrame containing the concatenated data from the tables.
        """
        cnxn = sqlite3.connect(f'{self.dbname}.db')
        dataframes = {}
        for table_name in table_names:
            df = pd.read_sql_query(f"SELECT * FROM {table_name}", cnxn)
            dataframes[table_name] = df
        cnxn.close()
        df_comb = pd.concat(dataframes.values(), axis=1)
        return df_comb

    def preprocess_data(self, df_comb: pd.DataFrame, target: str = 'ChurnStatus') -> pd.DataFrame:
        """
        Preprocesses the data by removing specified columns and one-hot encoding categorical variables.

        Args:
            df_comb (DataFrame): The DataFrame to preprocess.
            target (str, optional): The target variable to exclude from preprocessing. Defaults to 'ChurnStatus'.

        Returns:
            DataFrame: The preprocessed DataFrame.
        """
        exclude_columns = [col for col in df_comb.columns if 'ID' in col or col == 'CustomerID' or col == target]
        categorical_columns = df_comb.select_dtypes(include=['object']).columns.tolist()
        categorical_columns = [col for col in categorical_columns if col not in exclude_columns]
        df_comb = pd.get_dummies(df_comb, columns=categorical_columns, drop_first=True, dtype=int)
        imputer = SimpleImputer(strategy='most_frequent')  # or use 'median' or 'most_frequent'
        df_comb = pd.DataFrame(imputer.fit_transform(df_comb), columns=df_comb.columns)
        return df_comb

    def split_data(self, df_comb: pd.DataFrame, target: str = 'ChurnStatus') -> tuple:
        """
        Splits the data into training and testing sets.

        Args:
            df_comb (DataFrame): The DataFrame to split.
            target (str): The target variable.

        Returns:
            tuple: A tuple containing the split data (X_train, X_test, y_train, y_test).
        """
        X = df_comb.drop(target, axis=1)
        y = df_comb[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test

    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Trains the RandomForestClassifier model on the given data.

        Args:
            X_train (DataFrame): Training features.
            y_train (Series): Training target variable.

        Returns:
            None
        """
        self.model.fit(X_train, y_train)
        logger.info("Model trained successfully.")

    def evaluate_model(self, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        """
        Evaluates the trained model on the test dataset.

        Args:
            X_test (DataFrame): Test features.
            y_test (Series): Test target variable.

        Returns:
            dict: A dictionary containing accuracy, confusion matrix, and classification report.
        """
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        classification_rep = classification_report(y_test, y_pred)
        return {
            'Accuracy': accuracy,
            'Confusion Matrix': conf_matrix,
            'Classification Report': classification_rep
        }

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Makes predictions using the trained RandomForestClassifier model.

        Args:
            X (DataFrame): Features for making predictions.

        Returns:
            array: Predicted labels.
        """
        return self.model.predict(X)

    def run_workflow(self, table_names: list) -> tuple:
        """
        Runs the entire machine learning workflow.

        Args:
            table_names (list): A list of table names to load data from.

        Returns:
            tuple: A tuple containing model metrics, test features, test labels, and predictions.
        """
        # table_names = ['State', 'PlanDetails', 'DayUsage', 'EveUsage', 'NightUsage', 'IntlUsage', 'CustomerMetrics']
        data = self.load_data_from_db(table_names)
        data = self.preprocess_data(data)

        X_train, X_test, y_train, y_test = self.split_data(data)

        self.train_model(X_train, y_train)
        metrics = self.evaluate_model(X_test, y_test)
        self.trained_columns = X_train.columns.tolist()


        best_model_predictions = self.model.predict(X_test)
        return metrics, X_test, y_test, best_model_predictions

    def save_predictions_to_db(self, X_test: pd.DataFrame, y_test: pd.Series, predictions: np.ndarray, table_name: str):
        """
        Saves the model predictions to the specified database table.

        Args:
            X_test (DataFrame): Test features.
            y_test (Series): Test target variable.
            predictions (array): Predicted labels.
            table_name (str): The name of the table to save predictions to.

        Returns:
            None
        """
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

    def predict_for_customer(self, CustomerID: int) -> tuple:
        """
        Makes a prediction for a specific customer using the trained model.

        Args:
            CustomerID (int): The ID of the customer for whom the prediction is made.

        Returns:
            tuple: A tuple containing the prediction and an error message if applicable.
        """    
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

        # # Make a prediction using the trained model
        # prediction = self.model.predict(df_customer_processed)

        # # Return the prediction
        # return prediction[0], None  # Assuming prediction is a numpy array and you want the first element
        # Make a prediction using the trained model
        prediction = self.model.predict(df_customer_processed)

        # Convert numpy.int64 to int (or use int(prediction[0]) if prediction is a single-value array)
        if isinstance(prediction, np.ndarray) and prediction.shape:
            # Assuming prediction is a 1D array with a single element
            prediction = int(prediction[0])
        else:
            # If prediction is a scalar
            prediction = int(prediction)

        # Return the prediction
        return prediction, None