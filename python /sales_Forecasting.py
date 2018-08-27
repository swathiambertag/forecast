#################################Packages #########################################
import numpy as np
from  sklearn.preprocessing  import StandardScaler
from  sklearn.preprocessing  import MinMaxScaler
import os
import pandas as pd
from sklearn.metrics import mean_squared_error          
from math import sqrt
import matplotlib.pyplot as plt
from pandas.io import sql
from sqlalchemy import create_engine
from datetime import datetime 
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD
from sklearn.preprocessing import StandardScaler
import logging
import pickle#For saving Model with pkl format
import datetime

os.chdir("/home/swathi/Downloads")

###########################DB connection##############################

user="root"
pw='root'
data_base="forecasting"


engine = create_engine("mysql+pymysql://{user}:{pw}@localhost/{db}"
                      .format(user=user,
                              pw=pw,
                              db=data_base))

conn = engine.connect()

#########################Selecting Parameters###############################

forecastPrd="H120"
ModelData_Points=363
ValidationData_Points=91
Time_Range="D"
Model_Type="NN"
Fd=30
userdata_id=60
model_size=80
Valid_size=20
forecastMessure="qty"


#############################Table_Names####################################

Model_Id=forecastPrd + "_" + str(model_size) + "_" + str(Valid_size) + "_" + forecastMessure + "_" + Time_Range + "_" + Model_Type + "_" + str(userdata_id) + ".pkl"#pkl format for saving Model 
Model_Set=forecastPrd + "_" + forecastMessure + "_" + Time_Range + "_" + Model_Type + "_Mp" + str(ModelData_Points) + "_Vp" + str(ValidationData_Points) + "_" + str(userdata_id) + ".csv" 
Fcoutfln="forecast_prediction" + "_" + forecastMessure.lower() + "_" + Time_Range.lower() + "_" + Model_Type.lower() + "_ms" + str(model_size) + "_vs" + str(Valid_size) + "_" + str(userdata_id)
ForecastProf="forecast_profile_" + forecastMessure.lower() + "_" + Time_Range.lower() + "_" + str(userdata_id)
Tab="forecast_prediction" + str(userdata_id)
FS=str(Fd) + Time_Range  + "Forecasted"
##############################Forecasting########################################
# Load from file
#pkl_filename=Model_Id     
      
#with open(pkl_filename, 'rb') as file:  
#    NN_model = pickle.load(file)

if Model_Type =="NN":
   Forecast=Nueral_net_Final.predict_ahead(n_ahead=Fd)

#modelSet=pd.read_csv(Model_Set)
date1=pd.DataFrame([])
modelSet=pd.read_csv(Model_Set)
del modelSet["Unnamed: 0"]
modelSet["Date"]=pd.to_datetime(modelSet["Date"])
#Item_Name=np.repeat(forecastPrd, [Fd], axis=0)
fD=max(modelSet[modelSet.Item_Name==forecastPrd]["Date"])

date1=pd.DataFrame([])
date=pd.DataFrame([])
lD=fD + datetime.timedelta(days=Fd)
next_day = fD
while True:
    if next_day > lD:
        break
    date=pd.DataFrame([next_day])
    date1=pd.concat([date,date1],axis=0)
    next_day += datetime.timedelta(days=1)

########################Forecast_data#################################################3
date1.columns=["Date"]
Date=date1.sort_values(by='Date')
Forecast_data=Date[1:len(Date)]
Forecast_data["Item_Name"]=np.repeat(forecastPrd,Fd)
Forecast_data[forecastMessure]=np.nan
Forecast_data["Forecast"+"_"+forecastMessure]=Forecast
Forecast_data["Sample"]="Forecast"
Forecast_data=Forecast_data.reset_index()
del Forecast_data['index']

Allpredata=modelSet.append(Forecast_data)
Allpredata=Allpredata.reset_index()
del Allpredata['index']
Allpredata["Error%"]=Allpredata.Residuals/Allpredata[forecastMessure]
Allpredata=Allpredata[['julian','Date','Item_Name',forecastMessure,'pred_'+forecastMessure,'Forecast_'+forecastMessure, 'Residuals','Error%','Sample']]

#############################arranging data############################3
Index1=Allpredata["pred_"+forecastMessure].index[Allpredata["pred_"+forecastMessure].isnull()]
#Allpredata["pred_"+forecastMessure].Index1[Allpredata["pred_"+forecastMessure]]
Allpredata["pred_"+forecastMessure][Index1.values]=0
df=Allpredata[Allpredata["pred_"+forecastMessure]!=0]
df1=df.append(Forecast_data)
Allpredata1=df1[['julian','Date','Item_Name','qty','pred_qty','Forecast_qty','Residuals','Error%','Sample']]
Allpredata1=Allpredata1.reset_index()
del Allpredata1['index']
#Allpredata["pred_"+forecastMessure].isnull()
############################plot#####################################
plt.plot(Allpredata1[forecastMessure])
plt.plot(Allpredata1["pred_" + forecastMessure])
plt.plot(Allpredata1["Forecast_" + forecastMessure])
plt.show()

#################Final_Raw_data_of All_Products##################################
Raw = pd.DataFrame()
Raw=pd.read_sql("select * from forecast_prediction"+str(userdata_id)+"",conn)
del Raw['index']
#Raw=pd.merge(Raw_data,Allpredata,how='left', left_on=['julian','Item_Name','Date'], right_on = ['julian','Item_Name','Date'])#
if len(Raw.columns)>4 or len(Raw.columns)==4:
    raw=Raw[Raw.Item_Name!=forecastPrd]
    
Raw_data=Allpredata.append(raw)
MainRaw_data=Raw_data[['julian','Date','Item_Name',forecastMessure,'pred_'+forecastMessure,'Forecast_'+forecastMessure,'Residuals','Error%','Sample']]

###################Data send to DB###############################################

try:       
    pd.read_sql("insert into prediction_table (prediction_data_table,prediction_profile_table,user_data_ID) values('"+Fcoutfln+"',"+"'"+ForecastProf+"'"+","+"'"+str(userdata_id)+"'"+")",conn) 
except Exception as ERR:
    print (ERR)
    print('running') 
    
last_rec1=pd.read_sql("SELECT LAST_INSERT_ID() as last_rec",conn)

try:       
    pd.read_sql("insert into prediction_log (prediction_ID,user_data_ID,item_name,status,forecast_type,model_id,model_size,validation_size,forecast_dur,forecast_table,forecast_profile) values('"+str(last_rec1.last_rec[0])+"',"+"'"+str(userdata_id)+"'"+","+"'"+forecastPrd+"'"+","+"'Completed'"+","+"'"+Model_Type+"'"+","+"'"+Model_Id+"'"+","+"'"+str(model_size)+"'"+","+"'"+str(Valid_size)+"'"+","+"'"+str(Fd)+"'"+","+"'"+Fcoutfln+"'"+","+"'"+ForecastProf+"'"+")",conn) 
except Exception as ERR:
    print (ERR)
    print('running') 


#################################Data send to DB######################################
H=pd.read_sql("SELECT * FROM information_schema.tables WHERE table_name = '" + Fcoutfln + "'",conn)

if(len(H)==0):
    Allpredata.to_sql(name=Fcoutfln,con=conn, if_exists='append')
else:
    Allpredata.to_sql(name=Fcoutfln,con=conn, if_exists='replace')
    print("Data already exiting in Database") 
    
H1=pd.read_sql("SELECT * FROM information_schema.tables WHERE table_name = '" + Tab + "'",conn)   

if(len(H1)==1):
    try: 
        pd.read_sql("drop table "+Tab,conn)
    except Exception as ERR:
        print (ERR)
        print('running') 
else:
    print("Data already drop in Database")
 
#FS=str(Fd) + Time_Range  + "Forecasted"
H2=pd.read_sql("SELECT * FROM information_schema.tables WHERE table_name = '" + "forecast_prediction"+str(userdata_id) + "'",conn)   

if(len(H2)==0):
    MainRaw_data.to_sql(name="forecast_prediction"+str(userdata_id),con=conn, if_exists='append')
else:
    print("Data already exiting in Database")    
          
try:
    pd.read_sql("update model_summary set Forecast_Status='" + FS + "' where Model_Id ='" + Model_Id + "'",conn)
except Exception as ERR:
    print (ERR)
    print('running')
    
###################################################end###################################################################################

















































