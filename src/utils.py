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
    ds = pd.read_csv(DATASET_LOCATION,skipinitialspace=True,usecols=[0,4,5,6,8,10,11,12,28,29,30,31,32,33,34,35,37]) 
    
    return ds

def getInfoDataSet(ds):
    ds.info()

def separateValues(ds):
    x = ds.drop(['Total Deaths'],axis=1)
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