DATASET_LOCATION = "../CSV/1900_2021_DISASTERS.csv"

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

def importCleanDataSet():
    
    ## Read DataFrame ##
    ds = pd.read_csv(DATASET_LOCATION,skipinitialspace=True,usecols=[0,4,5,6,8,10,11,12,28,29,30,31,32,33,34,35,37]) # [0,4,5,6,8,10,11,12,28,29,30,31,32,33,34,35,37]

    ds.drop(['ISO', 'Region'], axis=1, inplace=True)  
    #ds.drop(['Disaster Type','Disaster Subgroup'], axis=1, inplace=True)  



    dsAvg = ds['Total Deaths'].mean()
    print(dsAvg)
    ds.fillna({'Start Month': 1, 'Start Day': 1,'End Month':1,'End Day':1,'Total Deaths':dsAvg,'No Injured':0,'No Affected':0,'Total Affected':0}, inplace=True)
    
    ds['Start Day'] = ds['Start Day'].astype(int)
    ds['End Day'] = ds['End Day'].astype(int)
    ds['Start Month'] = ds['Start Month'].astype(int)
    ds['End Month'] = ds['End Month'].astype(int)
    ds['Year'] = ds['Year'].astype(int)
    ds['End Year'] = ds['End Year'].astype(int)

    dsStartTmp = pd.DataFrame()
    dsEndTmp = pd.DataFrame()

    dsStartTmp['year'] = ds['Year']
    dsStartTmp['month'] = ds['Start Month']
    dsStartTmp['day'] = ds['Start Day']
    
    dsEndTmp['year'] = ds['End Year']
    dsEndTmp['month'] = ds['End Month']
    dsEndTmp['day'] = ds['End Day']

    ds['Start Date'] = pd.to_datetime(dsStartTmp[['year', 'month', 'day']], errors='coerce')
    ds['End Date'] = pd.to_datetime(dsEndTmp[['year', 'month', 'day']], errors='coerce')

    # Fill NaT values with a default date (you can change this to fit your needs)
    default_date = pd.to_datetime('1900-01-01')
    ds['Start Date'].fillna(default_date, inplace=True)
    ds['End Date'].fillna(ds['Start Date'], inplace=True)

    ds['Duration'] = (ds['End Date'] - ds['Start Date']).dt.days
    ds.drop(['Start Month', 'Start Day','End Year','End Month','End Day','Start Date','End Date','Event Name'],axis=1,inplace=True)

    ## Mapping String values ## 

    type_mapping = {type_str: idx + 1 for idx, type_str in enumerate(ds['Disaster Type'].unique())}
    ds['Disaster Type'] = ds['Disaster Type'].replace(type_mapping)

    type_mapping = {type_str: idx + 1 for idx, type_str in enumerate(ds['Disaster Subgroup'].unique())}
    ds['Disaster Subgroup'] = ds['Disaster Subgroup'].replace(type_mapping)

    type_mapping = {type_str: idx + 1 for idx, type_str in enumerate(ds['Disaster Subtype'].unique())}
    ds['Disaster Subtype'] = ds['Disaster Subtype'].replace(type_mapping)

    # type_mapping = {type_str: idx + 1 for idx, type_str in enumerate(ds['ISO'].unique())}
    # ds['ISO'] = ds['ISO'].replace(type_mapping)

    # type_mapping = {type_str: idx + 1 for idx, type_str in enumerate(ds['Region'].unique())}
    # ds['Region'] = ds['Region'].replace(type_mapping)

    type_mapping = {type_str: idx + 1 for idx, type_str in enumerate(ds['Continent'].unique())}
    ds['Continent'] = ds['Continent'].replace(type_mapping)

    ## Changing Types ##

    ds['Total Deaths'] = ds['Total Deaths'].astype(int)
    ds['No Injured'] = ds['No Injured'].astype(int)
    ds['No Affected'] = ds['No Affected'].astype(int)
    ds['Total Affected'] = ds['Total Affected'].astype(int)


    

    return ds

def getInfoDataSet(ds):
    ds.info()

def separateValuesTD(ds):
    x = ds.drop(['Total Deaths','Total Affected','No Affected','No Injured'],axis=1)
    y = ds['Total Deaths'].values
    return x,y

def initTraining(x,y):
    Xtrain, Xtest, ytrain, ytest = train_test_split(x, y,test_size=0.25, random_state=0)
    Xtrain = Xtrain.values
    Xtest = Xtest.values
    if len(Xtrain.shape) < 2:
        Xtrain = Xtrain.reshape(-1, 1)
    if len(Xtest.shape) < 2:
        Xtest = Xtest.reshape(-1, 1)

    return Xtrain,Xtest,ytrain,ytest

def showHist(X):
    index = np.arange(len(X))
    bar_width = 0.9
    
    plt.bar(index, X[1], bar_width,  color="green")
    plt.xticks(index, X[0]) # labels get centered
    plt.show()