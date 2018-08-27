
#######################Packages with parameter############################################################################

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

os.chdir("/home/swathi/Downloads")

#################################selecting parameter##################################################
forecastPrd="H120"
ModelData_Points=363
ValidationData_Points=91
Time_Range="D"
Model_Type="NN"
userdata_id=60
model_size=80
Valid_size=20
Pc_ModelDataSize=.80
Pc_ValidationDataSize=.20
forecastMessure="qty"
Min_Rec_Model=30
Min_Rec_Val=7

####################################DataBase connection########################################

user="root"
pw='root'
data_base="forecasting"


engine = create_engine("mysql+pymysql://{user}:{pw}@localhost/{db}"
                     .format(user=user,
                             pw=pw,
                             db=data_base))
conn = engine.connect()

#############################################Fetch data from DataBase ##############################################


table_name=pd.read_sql("select user_table table_name from user_data_log where user_data_ID='" + str(userdata_id) + "'",conn)
Used_data=pd.read_sql("select * from " + table_name.table_name[0] + "",conn)
pro_data=pd.read_sql("select * from " + "pagr" + "_" + str(userdata_id) + "",conn)

Model_Id=forecastPrd + "_" + str(model_size) + "_" + str(Valid_size) + "_" + forecastMessure + "_" + Time_Range + "_" + Model_Type + "_" + str(userdata_id) + ".pkl"#pkl format for saving Model 
Model_Set=forecastPrd + "_" + forecastMessure + "_" + Time_Range + "_" + Model_Type + "_Mp" + str(ModelData_Points) + "_Vp" + str(ValidationData_Points) + "_" + str(userdata_id) + ".csv" 
OptTab="optimization_pdq_" + forecastMessure.lower() + "_" + Time_Range + "_" + str(userdata_id)
#Fcoutfln="forecast_prediction" + "_" + forecastMessure.lower() + "_" + Time_Range.lower() + "_" + Model_Type.lower() + "_ms" + str(model_size) + "_vs" + str(Valid_size) + "_" + str(userdata_id)

#################################Nueral Network Modeling#########################################

def Model_optmise(Used_data1,ModelData_Points1,ValidationData_Points1,forecastMessure1,forecastPrd1,userdata_id1,model_size1,Valid_size1,Model_Type1,Model_Id1,Min_Rec_Model1,Min_Rec_Val1):
    
    Prod_wise1=Used_data1[Used_data1.Item_Name==forecastPrd1]
    del Prod_wise1['index']
    min_MP_Th=ModelData_Points1   ## No of data points for modelling
    min_VP_Th=ValidationData_Points1  
    Prod_wise1=Prod_wise1.sort_values('julian')  
    Target_var=Prod_wise1[forecastMessure1].astype(int)
    running=str("Running")
    
    
    try:
        pd.read_sql("INSERT INTO model_summary (user_data_id,Item_Name,MS,VS,MP,VP,Model_Type,Model_Id,Model_Status) values("+str(userdata_id1)+","+"'"+str(forecastPrd1)+"'"+","+str(model_size1)+","+str(Valid_size1)+","+str(ModelData_Points1)+","+str(ValidationData_Points1)+","+"'"+str(Model_Type1)+"'"+","+"'"+str(Model_Id1)+"'"+","+"'Running'"+")",conn)
    except Exception as ERR:
            print (ERR)
            print('running')
           
    last_insert_id=pd.read_sql("SELECT LAST_INSERT_ID() as last_rec",conn)
    
    if Min_Rec_Model1 > min_VP_Th:
        try:
            pd.read_sql("update model_summary set Model_Status = 'Insufficient Data points for Modeling' where MS_id= " + str(last_insert_id.last_rec[0]) + " and  Model_Type = '" + Model_Type1 + "'",conn)
            print("No of data points are less or insufficient")
        except Exception as ERR:
            print (ERR)
            print('running')
            
    if Min_Rec_Val1 > min_VP_Th:
        try:
            pd.read_sql("update model_summary set Model_Status = 'Insufficient Data points for Modeling' where MS_id= " + str(last_insert_id.last_rec[0]) + " and  Model_Type = '" + Model_Type1+ "'",conn)
            print("No of data points are less or insufficient")
        except Exception as ERR:
            print (ERR)
            print('running')
            
    if Min_Rec_Model1 > min_MP_Th and Min_Rec_Val1 > min_VP_Th:
        
         try:
            pd.read_sql("update model_summary set Model_Status = 'Insufficient Data points for Modeling' where MS_id= " + str(last_insert_id.last_rec[0]) + " and  Model_Type = '" + Model_Type1 + "'",conn)
            print("No of data points are less or insufficient") 
         except Exception as ERR:
            print (ERR)
            print('running') 

    return Prod_wise1,Target_var,min_MP_Th,min_VP_Th,last_insert_id


Model_optimisation=Model_optmise(Used_data,ModelData_Points,ValidationData_Points,forecastMessure,forecastPrd,userdata_id,model_size,Valid_size,Model_Type,Model_Id,Min_Rec_Model,Min_Rec_Val)


#Used_data1=Used_data
#ModelData_Points1=ModelData_Points
#ValidationData_Points1=ValidationData_Points
#forecastMessure1=forecastMessure
#forecastPrd1=forecastPrd
#userdata_id1=userdata_id
#model_size1=model_size
#Valid_size1=Valid_size
#Model_Type1=Model_Type
#Model_Id1=Model_Id
#Min_Rec_Model1=Min_Rec_Model
#Min_Rec_Val1=Min_Rec_Val



Pro_data=Model_optimisation[0] 
Target_variable=Model_optimisation[1]
min_MP_Th=Model_optimisation[2] 
min_VP_Th=Model_optimisation[3] 
last_insert_id=Model_optimisation[4]

#####################Train and Test data###############################################

Train_data=Pro_data[0:min_MP_Th]
Test_data=Pro_data[min_MP_Th:len(Pro_data)]
#######################Converting data to array format #############################
Train_arr=np.array(Train_data[forecastMessure])
####################################NN_Model#########################################

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)


class TimeSeriesNnet(object):
	def __init__(self, hidden_layers = [20, 15, 5], activation_functions = ['relu', 'relu', 'relu'], 
              optimizer = SGD(), loss = 'mean_absolute_error'):
		self.hidden_layers = hidden_layers
		self.activation_functions = activation_functions
		self.optimizer = optimizer
		self.loss = loss

		if len(self.hidden_layers) != len(self.activation_functions):
			raise Exception("hidden_layers size must match activation_functions size")

	def fit(self, timeseries, lag = 7, epochs = 10000, verbose = 0):
		self.timeseries = np.array(timeseries, dtype = "float64") # Apply log transformation por variance stationarity
		self.lag = lag
		self.n = len(timeseries)
		if self.lag >= self.n:
			raise ValueError("Lag is higher than length of the timeseries")
		self.X = np.zeros((self.n - self.lag, self.lag), dtype = "float64")
		self.y = np.log(self.timeseries[self.lag:])
		self.epochs = epochs
		self.scaler = StandardScaler()
		self.verbose = verbose

		logging.info("Building regressor matrix")
		# Building X matrix
		for i in range(0, self.n - lag):
			self.X[i, :] = self.timeseries[range(i, i + lag)]

		logging.info("Scaling data")
		self.scaler.fit(self.X)
		self.X = self.scaler.transform(self.X)

		logging.info("Checking network consistency")
		# Neural net architecture
		self.nn = Sequential()
		self.nn.add(Dense(self.hidden_layers[0], input_shape = (self.X.shape[1],)))
		self.nn.add(Activation(self.activation_functions[0]))

		for layer_size, activation_function in zip(self.hidden_layers[1:],self.activation_functions[1:]):
			self.nn.add(Dense(layer_size))
			self.nn.add(Activation(activation_function))

		# Add final node
		self.nn.add(Dense(1))
		self.nn.add(Activation('linear'))
		self.nn.compile(loss = self.loss, optimizer = self.optimizer)
		logging.info("Training neural net")
		# Train neural net
		self.nn.fit(self.X, self.y, nb_epoch = self.epochs, verbose = self.verbose)

	def predict_ahead(self, n_ahead = 1):
    		# Store predictions and predict iteratively
        	self.predictions = np.zeros(n_ahead)
    
        	for i in range(n_ahead):
        		self.current_x = self.timeseries[-self.lag:]
        		self.current_x = self.current_x.reshape((1, self.lag))
        		self.current_x = self.scaler.transform(self.current_x)
        		self.next_pred = self.nn.predict(self.current_x)
        		self.predictions[i] = np.exp(self.next_pred[0, 0])
        		self.timeseries = np.concatenate((self.timeseries, np.exp(self.next_pred[0,:])), axis = 0)
        
        	return self.predictions
        
        
###############################Model_build########################################
node=[5,10,15,20,25,30,35,40,45,50]
error=100000000

for i in range(0,7):
            
    #import random
    #random.seed(9001)                                
    Nueral_net=TimeSeriesNnet(hidden_layers = [node[i],node[i+1],node[i+2]],activation_functions = ['softmax','linear','sigmoid'])
    Nueral_net.fit(Train_arr,lag = 100, epochs = 1000, verbose = 0)
    
    ##################################Model_Testing#################################
    #import random
    #random.seed(9001)    
    Pred=Nueral_net.predict_ahead(n_ahead=min_VP_Th)
    Test_data["pre_" + forecastMessure]=Pred
    Test_data["Residuals"]=Test_data[forecastMessure]-Test_data["pre_" + forecastMessure]
    
    #########################Error calculation################################3
    Valid_RMSE = sqrt(mean_squared_error(Test_data[forecastMessure],Test_data["pre_" + forecastMessure]))
    Valid_MAPE=abs((Test_data[forecastMessure]-Test_data["pre_" + forecastMessure])/Test_data[forecastMessure]).mean()
   
    plt.plot(Test_data[forecastMessure])
    plt.plot(Test_data["pre_" + forecastMessure])
    plt.show()
    
    if error>Valid_MAPE:
       error= Valid_MAPE
       hl=[node[i],node[i+1],node[i+2]]
       n = i
       RMSE=round(Valid_RMSE,3)
       Test_data_set=Test_data
       Test_data_set.to_csv("Test_data" + str(Valid_MAPE) + ".csv") 
       #print(error)
    else:
       print("Model_is_running")

    
    print("MAPE" +" " + str(Valid_MAPE))
#################################Forecast###################################
try:
    pd.read_sql("update model_summary set Model_Status='Running' where MS_id='" + str(last_insert_id.last_rec[0]) + "' and Model_Type ='" + Model_Type + "'",conn)
except Exception as ERR:
    print (ERR)
    print('running')         
        
Test_data_set=pd.read_csv("Test_data" + str(error) + ".csv")
del Test_data_set["Unnamed: 0"]
Test_data_set["Date"]=pd.to_datetime(Test_data_set["Date"])
        
plt.plot(Test_data_set[forecastMessure])
plt.plot(Test_data_set["pre_" + forecastMessure])
plt.show()

################################Final_Model###############################
Final=Pro_data[ValidationData_Points:len(Pro_data)]
Finaltrain_arr=np.array(Final[forecastMessure])

Nueral_net_Final=TimeSeriesNnet(hidden_layers = [hl[0],hl[1],hl[2]],activation_functions = ['softmax','linear','sigmoid'])
Nueral_net_Final.fit(Finaltrain_arr,lag = 100, epochs = 1000, verbose = 0)
 
########################Final_dataset#######################################
   
Pro_data=Pro_data.reset_index()
del Pro_data['index']

Final_Model=pd.merge(Pro_data,Test_data_set,  how='left', left_on=['julian','Date','Item_Name'] ,right_on = ['julian','Date','Item_Name'])
Final_Model=Final_Model.drop([forecastMessure + "_y"], axis=1)
Final_Model.columns=["julian","Date","Item_Name",forecastMessure,"pred_" + forecastMessure,"Residuals"]
Final_Model["Sample"]="Model"

#############################saving Model ###########################
#import pickle
#
## Save to file in the current working directory
#pkl_filename = Model_Id 
#with open(pkl_filename, 'wb') as file:  
#    pickle.dump(TimeSeriesNnet, file)
#
# #Load from file
#with open(pkl_filename, 'rb') as file:  
#    pickle_model = pickle.load(file)
#    
##pickle_model.    
# pickle_model.predict_ahead(n_ahead=min_VP_Th)   
####################Error details of Modeling##############################################3

Final_Result=pd.DataFrame([])
Valid_RMSE_MAPE=pd.DataFrame([])
pro=pd.DataFrame(Final_Model.Item_Name.unique())
RMSE_error = pd.DataFrame([round(RMSE,2)])
MAPE_error = pd.DataFrame([round(error,2)])
Valid_RMSE_MAPE=pd.concat([RMSE_error,MAPE_error],axis=1)
Final_Result=pd.concat([pro,Valid_RMSE_MAPE],axis=1,ignore_index=True)

Final_Result.columns=["Item_Name","RMSE_error","MAPE_error"]
##########################Data save as CSv Format############################
Final_Model.to_csv(Model_Set) 
 
###########################Data send to DB #############################################3

try:
    pd.read_sql("update modelvalidation_datapoint_summary_"+str(userdata_id) +" set NN ='Completed' where Item_Name='" + forecastPrd + "' and  Train_count='" + str(ModelData_Points) + "' and  Test_count ='" + str(ValidationData_Points) + "'",conn)
except Exception as ERR:
    print (ERR)
    print('running') 

try:       
    pd.read_sql("insert into model_log_table (user_data_id,model_type,model_profile_quantity) values("+str(userdata_id)+","+"'"+str(Model_Type)+"'"+","+"'"+OptTab+"'"+")",conn) 
except Exception as ERR:
    print (ERR)
    print('running') 

###################################Send to DB#################################
    
P=pd.read_sql("SELECT * FROM information_schema.tables WHERE table_name = '" + OptTab + "'",conn)

if(len(P)==0):
    Final_Result.to_sql(name=OptTab,con=conn, if_exists='append')
else:
    Final_Result.to_sql(name=OptTab,con=conn, if_exists='replace')
    print("Data already exiting in Database") 
    
#######################Kill connection########################################

conn.close()
engine.dispose()   
    
##################End##########################################################    

   













 





















