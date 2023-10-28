import pandas as pd 

def data_State(df):
    return pd.DataFrame(df.loc[:,'State'])

def data_PlanDetails(df):
    return pd.DataFrame(df.loc[:,['Area Code','International Code']])