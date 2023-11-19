import sqlite3
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from ..db.sql_interactions import SqlHandler

def data_prep():
    # df_comb = pd.read_csv('updated_telecom_data.csv').drop(['StateID','PlanID','DayUsageID','EveUsageID','NightUsageID','IntlUsageID'])
    dbname = 'temp'
    table_names = ['State', 'PlanDetails', 'DayUsage', 'EveUsage', 'NightUsage', 'IntlUsage','CustomerMetrics']
    cnxn = sqlite3.connect(f'{dbname}.db')

    dataframes = {}

    for table_name in table_names:
        sql_handler = SqlHandler(dbname=dbname, table_name=table_name)
        df = sql_handler.from_sql_to_pandas()
        dataframes[table_name] = df
    
    df_comb = pd.concat(dataframes.values(), axis=1).drop(['StateID','PlanID','DayUsageID','EveUsageID','NightUsageID','IntlUsageID'],axis=1)
   
    # a = pd.get_dummies(df_comb, dtype=int)
    print(df_comb.head())
    df_comb.to_csv('test.csv',index=False)
    return df_comb



# # # Connect to the SQLite database
# # conn = sqlite3.connect('your_database_name.db')

# # # Example query to retrieve data (replace with your actual query)
# # query = "SELECT column1, column2, target_column FROM your_table_name WHERE conditions;"
# data = data_prep()

# Convert categorical columns to numerical using Label Encoding
# label_encoder = LabelEncoder()
# data["InternationalPlan"] = label_encoder.fit_transform(data["InternationalPlan"])
# data["VoiceMailPlan"] = label_encoder.fit_transform(data["VoiceMailPlan"])

# # Convert StateName column to numerical using one-hot encoding
# data = pd.get_dummies(data, columns=["StateName"], drop_first=True)

# # Example preprocessing steps (replace with your actual preprocessing steps)
# X = data.drop("ChurnStatus", axis=1)
# y = data["ChurnStatus"]

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# # ****************** DECISION TREE MODEL ******************

# model_decisiontree = DecisionTreeClassifier(random_state=42)

# model_decisiontree.fit(X_train, y_train)

# y_pred_decisiontree = model_decisiontree.predict(X_test)

# accuracy_1 = accuracy_score(y_test, y_pred_decisiontree)
# conf_matrix_1 = confusion_matrix(y_test, y_pred_decisiontree)
# classification_rep_1 = classification_report(y_test, y_pred_decisiontree)

# print("Accuracy:", accuracy_1)
# print("\nConfusion Matrix:\n", conf_matrix_1)
# print("\nClassification Report:\n", classification_rep_1)



# # ****************** RANDOM FOREST MODEL ******************

# model_randomforest = RandomForestClassifier(random_state=42)
# model_randomforest.fit(X_train, y_train)

# y_pred_randomforest = model_randomforest.predict(X_test)

# accuracy_2 = accuracy_score(y_test, y_pred_randomforest)
# conf_matrix_2 = confusion_matrix(y_test, y_pred_randomforest)
# classification_rep_2 = classification_report(y_test, y_pred_randomforest)

# print("Accuracy:", accuracy_2)
# print("\nConfusion Matrix:\n", conf_matrix_2)
# print("\nClassification Report:\n", classification_rep_2)


# # ****************** LOGISTIC REGRESSION MODEL ******************

# model_logisticReg = LogisticRegression(random_state=42)

# model_logisticReg.fit(X_train, y_train)

# y_pred_logreg = model_logisticReg.predict(X_test)

# accuracy_3 = accuracy_score(y_test, y_pred_logreg)
# conf_matrix_3 = confusion_matrix(y_test, y_pred_logreg)
# classification_rep_3 = classification_report(y_test, y_pred_logreg)

# print("Accuracy:", accuracy_3)
# print("\nConfusion Matrix:\n", conf_matrix_3)
# print("\nClassification Report:\n", classification_rep_3)




# # *********************** Compare models based on metrics ***********************
# model_metrics = {
#     'Decision Tree': {
#         'Accuracy': accuracy_1,
#         'Classification Report': classification_rep_1
#     },
#     'Random Forest': {
#         'Accuracy': accuracy_2,
#         'Classification Report': classification_rep_2
#     },
#     'Logistic Regression': {
#         'Accuracy': accuracy_3,
#         'Classification Report': classification_rep_3
#     }
# }

# # Choose the best model based on accuracy (you can modify this based on your specific goals)
# best_model_name = max(model_metrics, key=lambda k: model_metrics[k]['Accuracy'])
# best_model_metrics = model_metrics[best_model_name]

# print(f"\nBest Model: {best_model_name}")
# print(f"Accuracy: {best_model_metrics['Accuracy']}")
# print("Classification Report:\n", best_model_metrics['Classification Report'])

# # ************************** Store the model output in the SQLite database ******************************

# best_model_predictions = globals()[f'y_pred_{best_model_name.lower().replace(" ", "")}']

# # Add a new column for predictions in the original dataframe
# data['Predictions'] = best_model_predictions

# # # Update the database with the new column
# # data.to_sql('your_table_name', conn, if_exists='replace', index=False)
