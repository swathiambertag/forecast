
######################################Libraries#############################################
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
import pymysql
import sys

#os.chdir("/media/swathi/New Volume/Downloads/Data")

os.chdir("/home/swathi/Downloads")
#df=pd.read_csv("CCD_new_data.csv")'
Fn = "CCD_new_data.csv"
Fn1 = "Date_Format.csv"
Fnw="CCD_new_data"

#user_id=sys.argv[1]

user_id=8
token=''
Random_no=''
#######################Read csv file#############################################################
x=pd.read_csv(Fn,skipinitialspace=True)
a1=pd.read_csv(Fn1,skipinitialspace=True)

#######################convert table name to proper format #############################################
Fn=re.sub(" ", "_", Fn)
Fn=Fn[:-4]
Fn=re.sub("[^a-zA-Z0-9 ' ']|^[\t]","_",Fn)
Fn=Fn + ".csv"
Fnw=re.sub(" ", "_", Fnw)
Fnw=re.sub("[^a-zA-Z0-9 ' ']|^[\t]","_",Fnw)

######################################Changing colum names########################################

ndata=x.columns
forecastMessure=x.columns[2]
x.columns = ['DateTime','Item_Name',forecastMessure]

######################################Data clean (Item_Name)################################################
if len(x[x.isnull().any(1)])>0:  
        Miss_len = len(x[x.isnull().any(1)])
        Miss_per=Miss_len/len(x)*100
      
def Data_clean(x2):
    #x2 = x2.dropna(axis = 0, how = 'any').reset_index(drop=True)
    x2=x.dropna(axis=0, subset=('DateTime','Item_Name',)) 
    x2['Item_Name']=x2['Item_Name'].astype('str')   
    x2['Item_Name']=x2['Item_Name'].str.replace(r'[^a-zA-Z0-9 ' ']|^[\t]',r'').astype('str')
    return(x2)

x3=Data_clean(x)


######################mutidate function ###############################
samples=x3['DateTime']
format=a1['Formats']
def multidate(string):
    for fmt in format:
        try:
            return datetime.strptime(string, fmt).date()
        except ValueError:
            pass
b=[]
for sample in samples:
     b.append(multidate(sample))
print(b)
x3['Date1']=b

x3=x3.reset_index()#this for if we remove missing values in Data we need to rearrange index
x3=x3.drop('index', axis=1)
#############Date format checking m/d/y and d/m/y#########################
if(len(x[x.isnull().any(1)])>0):
    
    def Date_Format(x2):
        nans = lambda x2: x2[x2.isnull().any(axis=1)]#they are checking X is having NA or not
        p=nans(x2).index.values
        q=pd.DataFrame(pd.to_datetime(x2['DateTime'][p])).astype(object)
        d1 = pd.DataFrame(x2['Date1']).rename(columns={'Date1':'DateTime'})
        d1 = d1.drop(d1.index[p])#Removing NA value 
        x0 = d1.append(q)
        x0 = x0.sort_index()
        x2['Date1'] = x0
        return(x2)

    x3=Date_Format(x3)


#for i in range(len(x4.Date1)):
#    x4.Date1[i] = str(x4.Date1[i]).split(" ")[0]
#    x4.Date1[i] = datetime.strptime(x4.Date1[i], '%Y-%m-%d').date()



#x4.Date1[0].split()
#x['Date1'] = x1
#########################Time################################

x3["DateTime"]=pd.to_datetime(x3["DateTime"])
x3["Time"]=x3["DateTime"].dt.time
Time=x3["Time"]


x4=x3.sort_values(by='Date1')
#x4=x4.sort_values(by='DateTime')

######################Date frequency#############################################
#if Time.isnull().values.any() |  len(Time.unique()) == 1 :
#    unqe_data=pd.to_datetime(x4["Date1"]).unique()
#    unqe_data1=(unqe_data[1:len(unqe_data)] - unqe_data[0:len(unqe_data)-1])
#    unqe_data1=unqe_data1/86400000000000
#    
#    Data_Freq=mode(unqe_data1.astype(int))
#    data_freq_tab=["Day","Week","Month","Quarter","Year"]
#    if Data_Freq == 1:
#        data_freq_tab1 = data_freq_tab[0]
#    elif Data_Freq==7 :
#         data_freq_tab1 = data_freq_tab[1]
#    elif (Data_Freq== 28 | Data_Freq == 29 |Data_Freq==30 | Data_Freq==31):
#        data_freq_tab1 = data_freq_tab[2]
#    elif (Data_Freq > 89 & Data_Freq < 93) :
#        data_freq_tab1 = data_freq_tab[3]
#    elif (Data_Freq==365 | Data_Freq==366):
#        data_freq_tab1 = data_freq_tab[4]
#    Time_Range = data_freq_tab1
#    
    
if Time.isnull().values.any() |  len(Time.unique()) == 1 :
   unqe_data=pd.to_datetime(x4["Date1"]).unique()
   unqe_data1=(unqe_data[1:len(unqe_data)] - unqe_data[0:len(unqe_data)-1])
   unqe_data1=unqe_data1/86400000000000
   Data_Freq=mode(unqe_data1.astype(int))
   data_freq_tab=["Day","Week","Month","Quarter","Year"]
   if Data_Freq == 1:
       data_freq_tab1 = 'D'
   elif Data_Freq==7 :
        data_freq_tab1 = '7D'
   elif (Data_Freq== 28 | Data_Freq == 29 |Data_Freq==30 | Data_Freq==31):
       data_freq_tab1 = pd.DateOffset(months=1)
   elif (Data_Freq > 89 & Data_Freq < 93) :
       data_freq_tab1 = (pd.DateOffset(months=1)*4)
   elif (Data_Freq==365 | Data_Freq==366):
       data_freq_tab1 = pd.DateOffset(years=1)
   Time_Range = data_freq_tab1    
   #x4['DateTime']=x4['Date1']
#####################################Time intervals#########################################
#if((! (Time.isnull().values.any()))  &&  len(Time.unique()) == 1):
if(Time.isnull().any()==False and len(np.unique(Time.values))>1):
    x4['Hour']=x4['DateTime'].dt.hour.astype("str").apply(lambda x: x.zfill(2)) #leading 0 to Hour
    x4['Minute']=x4['DateTime'].dt.minute.astype("str").apply(lambda x: x.zfill(2))
    x4['Second']=x4['DateTime'].dt.second.astype("str").apply(lambda x: x.zfill(2))
    x4['Time']=x4['Hour'].astype(str)+":" +x4['Minute'].astype(str)+":" +x4['Second'].astype(str) 
    x4['DateTime'] = x4['Date1'].astype(str)+" " + x4['Time'].astype(str)
    uniqe_datetime=pd.to_datetime(x4['DateTime'].unique())
    uniqe_datetime1=(uniqe_datetime[1:len(uniqe_datetime)]-uniqe_datetime[0:len(uniqe_datetime)-1]).astype(int)
    uniqe_datetime1=uniqe_datetime1/3600000000000
    Datetime_Fre=mode(uniqe_datetime1.astype(int))
    Time_Range=Datetime_Fre.astype('str')+"H" #Time Range Time intervals'1H'
 ##############################Ordering_columns###########################################
cols = x4.columns.tolist()
 # taking columnsnames
if len(x4.columns) == 8:
    x4=x4[cols[:-5]] # deleting column
elif len(x4.columns) == 5:
    x4 = x4[cols[:-2]]   
##############################Creat Julian###################################################################   
    #x4["DateTime"]=pd.to_datetime(x4["DateTime"])
x4=x4.sort_values('DateTime')  
#df=x4 
def Create_Julian(df):
    da = df.sort_values('DateTime')
    min_value=min(da.DateTime)
    max_value=max(da.DateTime)
    #Julian = pd.DataFrame()
    Julian = pd.DataFrame(pd.date_range(min_value, max_value, freq = Time_Range))
    Julian.columns = ['Date']
    Julian['seq']=list(range(0,len(Julian)))
    #Unique_Date = pd.DataFrame()
    Unique_Date = pd.DataFrame(pd.to_datetime(x["DateTime"]))
    Unique_Date.columns = ['DateTime']
    s=pd.merge(Unique_Date, Julian, how='left', left_on = ['DateTime'], right_on = ['Date'])
    df['julian']=s['seq']
    df['DateTime']=pd.to_datetime(df["DateTime"])
    #Imputation=sqldf("""SELECT Item_Name,count(*) as Imputation FROM Imputation_data where Original is NULL group by Item_Name""", locals())
    #s=sqldf("""select df.*,Julian.seq from df left join Julian on df.Date1= Julian.Date""",locals())
#    naT = s[s.DateTime.isnull()]
#    Missing_Dates_Pattern = pd.DataFrame(naT.index).values.tolist()
    return df

#Skip_Dates_Pattern = pd.DataFrame(Missing_Date_Pattern(x4)).diff()

x4=Create_Julian(x4)
   
#######Data Profile data###########################################################
#def Profile(x2):
#    a1=x2.groupby(x2['Item_Name'])[forecastMessure].agg([len,'std', 'mean', 'max','median','sum']).reset_index()
#    a12=x2.groupby(x2['Item_Name'])['DateTime'].agg(['min','max']).reset_index()
#    PAgr=pd.merge(a1,a12,  how='left', left_on='Item_Name', right_on = 'Item_Name')
#    PAgr.columns = ["Item_Name","Total_"+forecastMessure.title(),"Std_"+forecastMessure.title(),"Average_"+forecastMessure.title(),"Max_"+forecastMessure.title(),"Median_"+forecastMessure.title(),"sum_"+forecastMessure.title(),
#                "StartDateTime_"+forecastMessure.title(),"EndDateTime_"+forecastMessure.title()]
#    return(PAgr)
#
#Profile_data=Profile(x4)


def Profile(x2):
    a1=x2.groupby(x2['Item_Name'])[forecastMessure].agg([len,'std', 'mean', 'max','median','sum']).reset_index()
    a12=x2.groupby(x2['Item_Name'])['DateTime'].agg(['min','max']).reset_index()
    a13=x2.groupby(x2['Item_Name'])['julian'].agg(['min','max']).reset_index()
    PAgr1=pd.merge(a1,a12,  how='left', left_on='Item_Name', right_on = 'Item_Name')
    PAgr=pd.merge(PAgr1,a13, how='left', left_on='Item_Name', right_on = 'Item_Name')
    PAgr.columns = ["Item_Name","Total_"+forecastMessure,"Std_"+forecastMessure.title(),"Average_"+forecastMessure.title(),"Max_"+forecastMessure.title(),"Median_"+forecastMessure.title(),"sum_"+forecastMessure.title(),
                "StartDateTime_"+forecastMessure.title(),"EndDateTime_"+forecastMessure.title(),"Min_Julian","Max_Julian"]
    return(PAgr)

Profile_data=Profile(x4)

###################################Median_Impute_for_Missing-date########################################
#PAgr1 = Profile_data
#x1 = x4



def Median_Imp(x1,PAgr1):

    Imp_median1=pd.DataFrame([])
    u = pd.unique(PAgr1.Item_Name)

    for n in range(len(u)):
        u1 = x1[x1.Item_Name == u[n]]
        u2 = u1[forecastMessure]
        qnt1 = u2.quantile(.25)
        qnt2 = u2.quantile(.75)
        qnt3 = u2.quantile(.5)
        iqr = np.subtract(*np.percentile(u2, [75, 25]))
        H1 = 1.5 * iqr
        HS1 = round((qnt2 + H1),2)
        LS1 = round((qnt1 - H1),2)
        qnt_25 = u1[(u1[forecastMessure]>LS1) & (u1[forecastMessure]<qnt3)]
        qnt_25 = qnt_25[qnt_25[forecastMessure]!=0]
        qnt_75 = u1[(u1[forecastMessure]>qnt3) & (u1[forecastMessure]<HS1)]
        qnt_75 = qnt_75[qnt_75[forecastMessure]!=0]
        qnt_25_size = len(u1[(u1[forecastMessure]>LS1) & (u1[forecastMessure]<qnt3)])
        qnt_75_size = len(u1[(u1[forecastMessure]>qnt3) & (u1[forecastMessure]<HS1)])

        if qnt_25_size > qnt_75_size:
            qp = median(qnt_25[forecastMessure])  
        if qnt_75_size > qnt_25_size: 
            qp = median(qnt_75[forecastMessure])   
        if qnt_75_size == qnt_25_size:  
            qp = median(u1[forecastMessure])  
        Imp_median = pd.DataFrame([u[n],round(qp)]).T  
        Imp_median1=pd.concat([Imp_median1,Imp_median], axis=0, ignore_index=True)
        #print(n)
  
    Imp_median1.columns=["Item_Name","M_Imp"]
    return Imp_median1

Imp_Mdata=Median_Imp(x4,Profile_data)

####################Find Out_Missing_date##################################################################

def Missing_date(x2):
    #x2=x4
    un=x2.Item_Name.unique()
    xmisD=pd.DataFrame([])

    for i in range(len(un)):
        u=x2[x2.Item_Name==un[i]]
        seq=pd.DataFrame([list(range(min(u.julian),max(u.julian)+1))]).T
        seq['Date']=pd.date_range(min(u.DateTime), max(u.DateTime), freq = Time_Range)
        seq['Item_Name']=np.repeat(u.Item_Name.unique(), [len(seq)], axis=0)#Repeating of Item_Name
        seq.columns=["seq","Date","Item_Name"]
        mer=pd.merge(seq,u, how='left', left_on='seq', right_on = 'julian')
        mer=mer.drop(['DateTime', 'Item_Name_y','julian'], axis=1)
        mer.columns=["julian","Date","Item_Name",forecastMessure]
        xmisD=pd.concat([xmisD,mer],axis=0,ignore_index=True)
    return xmisD
    
x5=Missing_date(x4)

##########################################Impute of Median#################################################

#x2=x5
#Imp_Mdata1=Imp_Mdata

def Imputaion_Fn(x2,Imp_Mdata1):
    Prd = x2.Item_Name.unique()
    unprd = pd.DataFrame()
    for i in range(len(Prd)):
        #print(i)
        unPrd_3 = x2[x2.Item_Name == Prd[i]]
        if unPrd_3.isnull().values.any() :
            Imp=Imp_Mdata1[Imp_Mdata1.Item_Name==Prd[i]]
            index1 = unPrd_3[forecastMessure].index[unPrd_3[forecastMessure].apply(np.isnan)]
            unPrd_3[forecastMessure][index1.values] = float(Imp.M_Imp)#float for rounding value
        unprd = pd.concat([unprd,unPrd_3], axis=0, ignore_index=True)
    unprd["Original"] = x2[forecastMessure]
    unprd.rename(columns={forecastMessure:"Imputed"+"_" +forecastMessure}, inplace=True)
    #print(i)
    #unprd_4.to_csv("Imputaion_data.csv").reset_index(drop=True)
    return(unprd)

x5=Imputaion_Fn(x5,Imp_Mdata)
xdata=x5
x5.rename(columns={"Imputed"+"_" +forecastMessure:forecastMessure}, inplace=True)

####################Boxplot_with_Data_capping####################################################
#x2=x5
#def Outlier(x2):
#def Outliers_capping(x2):
#    PrdName = x2.Item_Name.unique()
#    #x11=pd.DataFrame()
#    OutReportALL=pd.DataFrame()
#    unPrd1 = pd.DataFrame()
#    #OutReport=pd.DataFrame[]
#    
#    
#    for i in range(len(PrdName)):
#        #print(i)
#        unPrd=x2[x2.Item_Name == PrdName[i]]
#    #unPrd.index=range(len(unPrd))
#        unPrdVal=unPrd[forecastMessure]
#        q=unPrdVal.quantile([0.25,0.75])
#        if(q.iloc[0]==q.iloc[1]):
#            outReport=pd.DataFrame([PrdName[i],np.nan,np.nan,0,0]).T
#        else:
#            if(q.iloc[1]>q.iloc[0]):
#                H=1.5*(q.iloc[1]-q.iloc[0])
#                HS1=float(q.iloc[1]+H)# Removing unwanted (float)
#                LS1=float(q.iloc[0]-H)
#                gre_HS=unPrd[(unPrd[forecastMessure]>HS1)]
#                gre_Index=unPrd[forecastMessure].index[unPrd[forecastMessure]>HS1]
#                less_LS=unPrd[(unPrd[forecastMessure]<LS1)]
#                less_Index=unPrd[forecastMessure].index[unPrd[forecastMessure]<LS1]
#                nout=len(gre_Index)+len(less_Index)
#            if(nout>=1):
#                unPrd[forecastMessure][gre_Index.values] = HS1#Data capping
#                unPrd[forecastMessure][less_Index.values] = LS1 #Data capping
#                outReport=pd.DataFrame([PrdName[i],HS1,LS1,len(gre_Index),len(less_Index),nout,round(nout/len(unPrd)*100,2)]).T
#            else:
#                 outReport=pd.DataFrame([PrdName[i],HS1,LS1,0,0,0,0]).T
#                
#        unPrd1=pd.concat([unPrd1,unPrd], axis=0, ignore_index=True)
#        OutReportALL=pd.concat([OutReportALL,outReport],axis=0,ignore_index=True)  
#        
#    OutReportALL.columns=["Item_Name","HS","LS","Gre_Max","Les_Min","nout","pcOut"]
#    return(unPrd1,OutReportALL)
#
#x_twod=Outliers_capping(x5) #xtwo_data
#x6=x_twod[0]
#Out_Reports=x_twod[1] 
# 
#
#PAgr=pd.merge(Profile_data,Out_Reports,how='left', left_on=['Item_Name'], right_on = ['Item_Name'])#
##########################################Normal_Distribution_for Outliers#######################################
#x2=x5
def Norm_dist(x2):
    summ_norm=pd.DataFrame([])
    Norm_data=pd.DataFrame([])
    Unq_Prd=Profile_data.Item_Name.unique()
    for i in range(len(Unq_Prd)):
        Prd = x2[x2.Item_Name == Unq_Prd[i]]
        Prd_Name=Unq_Prd[i]
        m=np.mean(Prd[forecastMessure])
        mean=round(m,2)
        std=Prd[forecastMessure].std()
        std_dev=round(std,2)
        sig = [mean+3*std_dev,mean+2*std_dev,mean+1*std_dev,mean+0*std_dev,mean-1*std_dev,mean-2*std_dev,mean-3*std_dev]
        Max = mean+ 2 * std_dev
        Min = mean- 2* std_dev
    
        if(Max==Min):
            norm=pd.DataFrame([Unq_Prd[i],Max,Min,0,0,0,0]).T
        if(Max>Min):
            max_above = Prd[Prd[forecastMessure] > Max]
            Index=Prd[forecastMessure].index[Prd[forecastMessure]>=Max]
            Prd[forecastMessure][Index.values]=round(Max)
            min_above = Prd[Prd[forecastMessure] < Min]
            Index1=Prd[forecastMessure].index[Prd[forecastMessure]<=Min]
            Prd[forecastMessure][Index1.values]=round(Min)
            Total_nabove_min_max=len(Prd[forecastMessure].index[Prd[forecastMessure]>=Max]|Prd[forecastMessure].index[Prd[forecastMessure]<=Min])
            norm=pd.DataFrame([Unq_Prd[i],Max,Min,len(Index),len(Index1),Total_nabove_min_max,round(Total_nabove_min_max/len(Prd)*100,2)]).T
        
        summ_norm=pd.concat([summ_norm,norm], axis=0, ignore_index=True)
        Norm_data=pd.concat([Norm_data,Prd], axis=0,ignore_index=True)
    
    summ_norm.columns=["Item_Name","sd_HS","sd_LS","sd_Gre_Max","sd_Les_Min","nout","pcout"]
    return summ_norm,Norm_data

    
x_normd=Norm_dist(x5)
Out_Reports_norm=x_normd[0]    
x6=x_normd[1]       

PAgr=pd.merge(Profile_data,Out_Reports_norm,how='left', left_on=['Item_Name'], right_on = ['Item_Name'])
           
###########################################Pagr table updation###################################### 
x6['updated']=0
Imputation=sqldf("""SELECT Item_Name,count(*) as Miss_days FROM x6  where Original is NULL group by Item_Name""", locals())
PAgr=pd.merge(PAgr,Imputation,how='left', left_on=['Item_Name'], right_on = ['Item_Name'])#

if PAgr.isnull().values.any(): # if any value having na in Pagr we are conver that value to 0
    I=PAgr['Miss_days'].index[PAgr['Miss_days'].apply(np.isnan)]  
    PAgr['Miss_days'][I.values]=0
    
PAgr['Act_AfterImput']=PAgr["Total_"+forecastMessure]+PAgr['Miss_days'].astype(int)
PAgr['PcImputation']=round(PAgr.Miss_days/PAgr.Act_AfterImput,2)

###############################check data points is suffiecient or not ######################################
#PAgr1=PAgr

def Check_data_point(PAgr1):

    u=PAgr1.Item_Name.unique()
    un2=pd.DataFrame([])
    un3=pd.DataFrame([])
    prI=pd.DataFrame([])
    pr=pd.DataFrame([])
    
    for i in range(len(u)):
        un=PAgr1[PAgr1.Item_Name==u[i]]
        if (un["Total_"+forecastMessure]<30)[i] and (un.Act_AfterImput<40)[i]:
            un1=un[un.Item_Name==u[i]]
            un2=pd.concat([un2,un1],axis=0,ignore_index=True)
            prI=pd.DataFrame([un2.Item_Name]).T
            prI['Status']="Insufficient"
        else:
           un1=un[un.Item_Name==u[i]] 
           un3=pd.concat([un3,un1],axis=0,ignore_index=True)
           prs=pd.DataFrame([un3.Item_Name]).T
           prs['Status']="Sufficient"
           
    if len(prI)>0 and len(prs)>0:
        pr=pd.concat([prs,prI],axis=0,ignore_index=True)
    if len(prI)==0:
        pr=prs
    if len(prs)==0:
        pr=prI
    return pr

pro_point=Check_data_point(PAgr)

#############################Randomly_passing_pd#########################################################
#PAgr1=PAgr 
 
def Random_Fd(PAgr1):
     
     u=PAgr1.Item_Name.unique()
     Fd2=pd.DataFrame([])
     
     for i in range(len(u)):
         un=PAgr1[PAgr1.Item_Name==u[i]]
         Fd=float(round(un["Total_"+forecastMessure]*30/365))
         Fd1=pd.DataFrame([u[i],Fd]).T
         Fd2=pd.concat([Fd2,Fd1],axis=0,ignore_index=True)
     Fd2.columns=["Item_Name","FD"]
     return(Fd2)
 
RanFd=Random_Fd(PAgr) 

###########################################Send data to DB #######################################################
from pandas.io import sql
from sqlalchemy import create_engine
from datetime import datetime # this for get current datetime
user="root"
pw='root'
data_base="forecasting"


engine = create_engine("mysql+pymysql://{user}:{pw}@localhost/{db}".format(user=user,pw=pw,db=data_base))
conn = engine.connect()

#db = create_engine('mysql://root@localhost/forecasting')
#for i in range(1,2000):
#    conn = engine.connect()
#    #some simple data operations
#    conn.close()
#    engine.dispose()






#db = pymysql.connect(host="localhost", user='root', passwd='', db="forecasting")

pd.read_sql("desc user_data_log",conn)

f=pd.read_sql("select * from user_data_log",conn)

#pd.read_sql("INSERT INTO user_data_log1 (userID,ActualTableName,DataFrequency,user_table,user_data_name,Date,ForecastProduct,forecastMessure, token) values("+str(user_id)+","+"'"+table+"'"+","+"'"+Time_Range+"'"+","+"'"+user_table+"'"+","+"'"+Fnw+"'"+","+"'"+ndata[0]+"'"+","+"'"+ndata[1]+"'"+","+"'"+ndata[2]+"'"+","+"'"+token+"'"+")",engine)
try:
    pd.read_sql("INSERT INTO user_data_log (userID,ActualTableName,DataFrequency,user_table,user_data_name,Date,ForecastProduct,forecastMessure, token) values("+str(user_id)+","+"'" "'"+","+"'"+Time_Range+"'"+","+"'" "'"+","+"'"+Fnw+"'"+","+"'"+ndata[0]+"'"+","+"'"+ndata[1]+"'"+","+"'"+ndata[2]+"'"+","+"'"+token+"'"+")",conn)
    print('running')
except Exception as ERR:
            print (ERR)
            print('running')

userdata_id=pd.read_sql("SELECT LAST_INSERT_ID() as userdata_id",conn)
userdata_id=userdata_id.iloc[0]#took only user_data_id value
userdata_id=int(userdata_id)

fn2=Fnw + "_" + str(userdata_id)
fn3=Fnw + "_" + "withoutimp" + "_" + str(userdata_id)
Profile="PAgr" + "_" + str(userdata_id)
Profile=Profile.lower()#convert to lower case

fn2=fn2.lower()
fn3=fn3.lower()
try:
    pd.read_sql("update user_data_log set user_table='" + fn2 + "',ActualTableName='" + fn3 + "' where user_data_ID ='" + str(userdata_id) + "'",conn)
    print('running')
except Exception as ERR:
            print (ERR)
            print('running')
x_data=x6.iloc[:,0:4]
x_withoutImp=x6.iloc[:,0:3]
x_withoutImp[forecastMessure]=x6['Original']
fname="forecast_prediction"+ str(userdata_id)

M=pd.read_sql("SELECT * FROM information_schema.tables WHERE table_name = '" + fn2 + "'",conn)
#M1=pd.read_sql("SELECT * FROM table_schema.tables WHERE table_name = '" + fn2 + "'",engine)
N=pd.read_sql("SELECT * FROM information_schema.tables WHERE table_name = '" + fn3 + "'",conn)
O=pd.read_sql("SELECT * FROM information_schema.tables WHERE table_name = '" + Profile + "'",conn)

if(len(M)==0):
    x_data.to_sql(name=fn2,con=conn, if_exists='append')
else:
    print("Data already exiting in Database") 
    
if(len(N)==0):
    x_withoutImp.to_sql(name=fn3,con=conn, if_exists='append')
else:
    print("Data already exiting in Database") 
    
if(len(O)==0):
    PAgr.to_sql(name=Profile,con=conn, if_exists='append')
else:
     print("Data already exiting in Database") 
     
if(len(O)==0):
    x_data.to_sql(name=fname,con=conn, if_exists='append')
else:
     print("Data already exiting in Database") 
     

print(userdata_id) 


####################################Kill connection#########################3
   
conn.close()
engine.dispose()



































