import pandas as pd 
from etl.data_preperation.sql_interactions import  SqlHandler

df  = pd.read_csv("telecom_data.csv")

def data_State():
    return pd.DataFrame(df.loc[:,'StateName'])

def data_PlanDetails():
    return pd.DataFrame(df.loc[:,['AreaCode','InternationalPlan','VoiceMailPlan','NumberVMailMessages']])

def data_DayUsage():
    return pd.DataFrame(df.loc[:,["TotalDayMinutes",'TotalDayCalls','TotalDayCharge']])

def data_EveUsage():
    return pd.DataFrame(df.loc[:,["TotalEveMinutes",'TotalEveCalls','TotalEveCharge']])

def data_NightUsage():
    return pd.DataFrame(df.loc[:,["TotalNightMinutes",'TotalNightCalls','TotalNightCharge']])

def data_IntlUsage():
    return pd.DataFrame(df.loc[:,["TotalIntlMinutes",'TotalIntlCalls','TotalIntlCharge']])

def InsertToTables():
    table_names = ['State','PlanDetails','DayUsage','EveUsage','NightUsage','IntlUsage']
    
    sql = SqlHandler('temp','State')
    sql.get_table_columns()
    sql.insert_many(data_State())
    sql.close_cnxn()
    

    sql = SqlHandler('temp','PlanDetails')
    sql.get_table_columns()
    sql.insert_many(data_PlanDetails())
    sql.close_cnxn()
    
    sql = SqlHandler('temp','DayUsage')
    sql.get_table_columns()
    sql.insert_many(data_DayUsage())
    sql.close_cnxn()

    sql = SqlHandler('temp','EveUsage')
    sql.get_table_columns()
    sql.insert_many(data_EveUsage())
    sql.close_cnxn()

    sql = SqlHandler('temp','NightUsage')
    sql.get_table_columns()
    sql.insert_many(data_NightUsage())
    sql.close_cnxn()

    sql = SqlHandler('temp','IntlUsage')
    sql.get_table_columns()
    sql.insert_many(data_IntlUsage())
    sql.close_cnxn()