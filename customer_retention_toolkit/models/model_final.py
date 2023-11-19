import sqlite3
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from ..db.sql_interactions import SqlHandler


def load_data_from_db(dbname):
    table_names = ['State', 'PlanDetails', 'DayUsage', 'EveUsage', 'NightUsage', 'IntlUsage', 'CustomerMetrics']
    cnxn = sqlite3.connect(f'{dbname}.db')
    dataframes = {}

    for table_name in table_names:
        sql_handler = SqlHandler(dbname=dbname, table_name=table_name)
        df = sql_handler.from_sql_to_pandas()
        dataframes[table_name] = df

    df_comb = pd.concat(dataframes.values(), axis=1)
    return df_comb


#pre_last
def preprocess_data(df_comb):
    # Exclude 'CustomerID' and any column with 'ID' in its name, as well as 'ChurnStatus'
    exclude_columns = [col for col in df_comb.columns if 'ID' in col or col in ['CustomerID', 'ChurnStatus']]

    # Select only the categorical columns, excluding the ones identified above
    categorical_columns = df_comb.select_dtypes(include=['object']).columns.tolist()
    categorical_columns = [col for col in categorical_columns if col not in exclude_columns]
    
    # Perform dummification
    df_comb = pd.get_dummies(df_comb, columns=categorical_columns, drop_first=True, dtype=int)
    
    return df_comb




def split_data(df_comb):
    X = df_comb.drop("ChurnStatus", axis=1)
    y = df_comb["ChurnStatus"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


def train_decision_tree(X_train, y_train):
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model


def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred)
    return accuracy, conf_matrix, classification_rep


#pre-last
def save_predictions_to_db(X_test, predictions, model_name, dbname, table_name):
    # Ensure 'CustomerID' is part of X_test
    # Create a new DataFrame for the results with the necessary columns
    results_df = pd.DataFrame({
        'CustomerID': X_test['CustomerID'],  # Ensure 'CustomerID' is part of X_test
        'PredictedLabel': predictions,
        'ModelName': [model_name] * len(predictions),
        # Include additional columns as necessary, e.g., Probability, ChurnStatus, ModelVersion
    })

    # Initialize the SqlHandler
    sql_handler = SqlHandler(dbname=dbname, table_name=table_name)
    
    # Use the insert_many method to insert the data
    sql_handler.insert_many(results_df)
    
    # Commit the changes and close the connection
    sql_handler.close_cnxn()

def preprocess_data(df_comb):
    # Exclude 'CustomerID', 'ChurnStatus', and any column with 'ID' in its name
    exclude_columns = [col for col in df_comb.columns if 'ID' in col or col in ['CustomerID', 'ChurnStatus']]
    categorical_columns = df_comb.select_dtypes(include=['object']).columns.tolist()
    categorical_columns = [col for col in categorical_columns if col not in exclude_columns]
    df_comb = pd.get_dummies(df_comb, columns=categorical_columns, drop_first=True, dtype=int)
    return df_comb


