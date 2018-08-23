# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 11:43:38 2018

@author: Admin
"""


#############################Librabries####################################################

import os
import pandas as pd
import sys # for passing arguement
from statistics import mode
from statistics import median
from datetime import datetime
from datetime import datetime ,date

import datetime as dt  
import numpy as np
from sklearn.model_selection import train_test_split
from pandasql import sqldf
from pandas.io import sql
from sqlalchemy import create_engine#connection to DB
import re
import math # for find out nan in data
from pandas.io import sql
from sqlalchemy import create_engine
from datetime import datetime


os.chdir("D:/recent_forecasting")

##############################selecting parameters#############################
Time_Range="D"
Pc_ModelDataSize=.80
#Pc_TestDataSize=.20
Pc_ValidationDataSize=.20
userdata_id=114
forecastMessure="NumberofCalls"
FnModelValidPeriod="ModelValidation_DataPoint_Summary" + "_" + str(userdata_id)
FnModelValidPeriod.lower()

################################DB connection#####################################
from pandas.io import sql
from sqlalchemy import create_engine
from datetime import datetime # this for get current datetime
user="root"
pw=''
data_base="forecasting"


engine = create_engine("mysql+pymysql://{user}:{pw}@localhost/{db}"
                     .format(user=user,
                             pw=pw,
                             db=data_base))

conn = engine.connect()


#db = pymysql.connect(host="localhost", user='root', passwd='', db="forecasting")

###################################Model_Validation#####################################

tab_name=pd.read_sql("select user_table table_name from user_data_log where user_data_ID='" + str(userdata_id) + "'",conn)
after_Impdata=pd.read_sql("select * from " + tab_name.table_name[0] + "",conn)

#Imputation_data1=after_Impdata

def Model_Valid(Imputation_data1):
    PrdName = Imputation_data1.Item_Name.unique()
    Model_Report=pd.DataFrame()
    Model_Report_final=pd.DataFrame()
    for i in range(len(PrdName)):
        print(i)
        unPrd = Imputation_data1[Imputation_data1.Item_Name == PrdName[i]].reset_index(drop=True)
#        train_size=round(len(unPrd)*Pc_ModelDataSize)
#        test_size=round(len(unPrd)*Pc_TestDataSize)
#        valid_size=round(len(unPrd)*Pc_ValidationDataSize)
        train,test= train_test_split(unPrd, train_size=Pc_ModelDataSize)
        Model_Report = pd.DataFrame([PrdName[i],len(train),len(test)]).T
        Model_Report_final = pd.concat([Model_Report_final,Model_Report], axis=0, ignore_index=True)

    Model_Report_final.columns = ['Item_Name', 'Train_count',"Test_count"]
    return(Model_Report_final)  

Model_Validation=Model_Valid(after_Impdata)
Model_Validation['SMA']='NULL'
Model_Validation['ARIMA']='NULL'
Model_Validation['NN']='NULL'

Model_valid="ModelValidation_DataPoint_Summary_" + str(userdata_id)
Model_valid=Model_valid.lower()
P=pd.read_sql("SELECT * FROM information_schema.tables WHERE table_name = '" + Model_valid + "'",engine)

if(len(P)==0):
    Model_Validation.to_sql(name=Model_valid,con=conn, if_exists='append')
else:
    print("Data already exiting in Database")
    
    
conn.close()
engine.dispose()   
    
    
#db.close()    
#####################################end#############################################   
 
    
    
    
    
    
    