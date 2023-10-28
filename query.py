import sqlite3
dbname = 'temp'
cnxn = sqlite3.connect(f'{dbname}.db')
cursor = cnxn.cursor()
import sqlite3

def run_query(cnxn, table_name, query):
    """
    Run a query on a specific table in the database.

    Parameters:
    - cnxn: The database connection object.
    - table_name: Name of the table to query.
    - query: The SQL query to run. It should contain a placeholder for the table name.

    Returns:
    - result: Result of the query.
    """
    cursor = cnxn.cursor()
    formatted_query = query.format(table_name=table_name)
    cursor.execute(formatted_query)
    result = cursor.fetchall()
    return result


#not yet complete
