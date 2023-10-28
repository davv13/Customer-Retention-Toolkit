import pandas as pd 
import os 

os.chdir(".")
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

State = data_State()
PlanDetails = data_PlanDetails()
DayUsage = data_DayUsage()
EveUsage = data_EveUsage()
NightUsage = data_NightUsage()
IntlUsage = data_IntlUsage()

ls = [State,PlanDetails]