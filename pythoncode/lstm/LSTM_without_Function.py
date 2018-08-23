
############Library used for LSTM  Model###################################################

import pandas
import os
import numpy
import math
import keras
import tensorflow
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
import pymysql
import matplotlib.pyplot as plt
import pandas as pd

scaler = MinMaxScaler(feature_range=(0, 1))


#################################selecting parameter##################################################
forecastPrd="10442729"
ModelData_Points=68
ValidationData_Points=18
Time_Range="D"
Model_Type="LSTM"
userdata_id=114
model_size=80
Valid_size=20
Pc_ModelDataSize=.80
Pc_ValidationDataSize=.20
forecastMessure="NumberofCalls"
Min_Rec_Model=12000
Min_Rec_Val=3000



####################################DataBase connection########################################
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

#############################################Fetch data from DataBase ##############################################


table_name=pd.read_sql("select user_table table_name from user_data_log where user_data_ID='" + str(userdata_id) + "'",conn)
Used_data=pd.read_sql("select * from " + table_name.table_name[0] + "",conn)
pro_data=pd.read_sql("select * from " + "pagr" + "_" + str(userdata_id) + "",conn)

Model_Id=forecastPrd + "_" + str(model_size) + "_" + str(Valid_size) + "_" + forecastMessure + "_" + Time_Range + "_" + Model_Type + "_" + str(userdata_id) + ".mod"

#################################LSTM Modeling#########################################






Prod_wise=Used_data[Used_data.Item_Name==forecastPrd]
del Prod_wise['index']
min_MP_Th=ModelData_Points   ## No of data points for modelling
min_VP_Th=ValidationData_Points  
Prod_wise1=Prod_wise.sort_values('julian')  
running=str("Running")


try:
    pd.read_sql("INSERT INTO model_summary (user_data_id,Item_Name,MS,VS,MP,VP,Model_Type,Model_Id) values("+str(userdata_id)+","+"'"+str(forecastPrd)+"'"+","+str(model_size)+","+str(Valid_size)+","+str(ModelData_Points)+","+str(ValidationData_Points)+","+"'"+str(Model_Type)+"'"+","+"'"+str(Model_Id)+"'"+"'"+str(15)+"'"+")",conn)
except:
    #pd.read_sql("update " + FnModelValidPeriod + " set NN='Running'  where Item_Name= '"  + forecastPrd + "' and  Train_count= "+str(ModelData_Points)+" and Test_count = "+str(ValidationData_Points)+"",conn)
    print('running')
  
    
    
    
last_insert_id=pd.read_sql("SELECT LAST_INSERT_ID() as last_rec",conn)

if Min_Rec_Model > min_VP_Th:
    try:
        pd.read_sql("update model_summary set Model_Status = 'Insufficient Data points for Modeling' where MS_id= " + str(last_insert_id.last_rec[0]) + " and  Model_Type = '" + Model_Type + "'",conn)
        print("No of data points are less or insufficient")
    except:
        print('running')
        
if Min_Rec_Val > min_VP_Th:
    try:
        pd.read_sql("update model_summary set Model_Status = 'Insufficient Data points for Modeling' where MS_id= " + str(last_insert_id.last_rec[0]) + " and  Model_Type = '" + Model_Type + "'",conn)
        print("No of data points are less or insufficient")
    except:
        print('running')
        
if Min_Rec_Model > min_MP_Th and Min_Rec_Val > min_VP_Th:
    
     try:
        pd.read_sql("update model_summary set Model_Status = 'Insufficient Data points for Modeling' where MS_id= " + str(last_insert_id.last_rec[0]) + " and  Model_Type = '" + Model_Type + "'",conn)
        print("No of data points are less or insufficient") 
     except:
        print('running')   
 
    
    
    
Prod_wise2=Prod_wise.iloc[(len(Prod_wise)-(min_MP_Th+min_VP_Th)):len(Prod_wise)]
train1=Prod_wise2.iloc[0:min_MP_Th]    
test1 = Prod_wise2.iloc[min_MP_Th:len(Prod_wise2)]  
######################### LSTM Modeling ################################################
dataset=Prod_wise2[forecastMessure]
dataset1=pd.DataFrame(dataset)
scaling_dataset=scaler.fit_transform(dataset1)
train, test = scaling_dataset[0:ModelData_Points,:],scaling_dataset[ModelData_Points:len(scaling_dataset),:]

##################### convert an array of values into a dataset matrix###########################

def create_dataset(dataset1, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset1)-look_back-1):#
		a = dataset1[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset1[i + look_back, 0])
	return numpy.array(dataX), numpy.array(dataY)

look_back = 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

trainX = numpy.reshape(trainX, (trainX.shape[0],1,trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

# #######################create and fit the LSTM network######################################
model = Sequential()
model.add(LSTM(2, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)

###########################Model Prediction on test data ###############################
model_predict=model.predict(testX)

########################Model prediction descaling###############################
model_predict_descale=scaler.inverse_transform(model_predict)
testY_descale=scaler.inverse_transform([testY])

#############################Calculation of Residuals #######################################
Final_test=pd.DataFrame([testY_descale[0],model_predict_descale[:,0]]).T
Final_test['Residuals']=testY_descale[0]-model_predict_descale[:,0]
Final_test.columns=["Actual","Prediction","Residuals"]

##########################Final_result#######################################
df = test1.reset_index(drop=True)
df1=df.drop([0,(len(test1)-1)])
df1=df1.reset_index(drop=True)
df2=pd.DataFrame([])
df2=pd.concat([df1,Final_test],axis=1,ignore_index=True)

df2.columns=["Julian","Date",'Item_Name', 'NumberofCalls',"Actual","Predicted","Residuals"]

df2=df2.drop(["Julian","NumberofCalls"],axis=1)

##########################Calculating Error##########################################
RMSE_score=math.sqrt(mean_squared_error(Final_test['Actual'], Final_test['Prediction']))
print('RMSE_score: %.2f RMSE' % RMSE_score)

#############################Ploting###############################################
plt.plot(df2.Actual,color='red',label='act_data')
plt.plot(df2.Predicted,color='blue',label='pred_data')
plt.title('pred_graph')
plt.xlabel('Smarts')
plt.ylabel('Probability')
plt.legend()
plt.show()


############################close Db connection####################################
conn.close()
engine.dispose()   
    























































































