"""
@file
@brief This file contains functions to import, clean, and preprocess the dataset for analysis.

@author: RICHARD Corentin & HODIN Dorian
"""

# Import necessary libraries 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Import specific classes from scikit-learn
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Location of the CSV file containing the dataset
DATASET_LOCATION = "CSV/1900_2021_DISASTERS.csv"

def importCleanDataSet():
    """
    Function to import and clean the dataset.

    Reads the dataset from the specified location, performs data cleaning operations such as handling missing values,
    converting data types, and preprocessing date information. Maps string values to integers for categorical variables
    and transforms the dataset for analysis.

    Returns:
    pandas.DataFrame: The cleaned and preprocessed dataset.
    """

    ## Read DataFrame ##
    ds = pd.read_csv(DATASET_LOCATION,skipinitialspace=True,usecols=[0,4,5,6,8,10,11,12,28,29,30,31,32,33,34,35,37]) # [0,4,5,6,8,10,11,12,28,29,30,31,32,33,34,35,37]

    
    # Handling missing values for 'Total Deaths'
    ds['Total Deaths'].fillna(0, inplace=True)
    #ds['Total Deaths'] = ds['Total Deaths'].fillna(ds.groupby(['Disaster Subtype', 'Region'])['Total Deaths'].transform('median'))
    
    # Filtering out extreme values for 'Total Deaths'
    ds = ds[ds['Total Deaths'] <= ds['Total Deaths'].quantile(0.80)]
    
    #ds.drop(['ISO', 'Region'], axis=1, inplace=True)  
    #ds.drop(['Disaster Type','Disaster Subgroup'], axis=1, inplace=True)  

    ##### Defining Duration Data #####

    # Handling missing values and type conversion for date columns
    ds.fillna({'Start Month': 1, 'Start Day': 1,'End Month':1,'End Day':1,'No Injured':0,'No Affected':0,'Total Affected':0}, inplace=True)    
    
    ds['Start Day'] = ds['Start Day'].astype(int)
    ds['End Day'] = ds['End Day'].astype(int)
    ds['Start Month'] = ds['Start Month'].astype(int)
    ds['End Month'] = ds['End Month'].astype(int)
    ds['Year'] = ds['Year'].astype(int)
    ds['End Year'] = ds['End Year'].astype(int)

    # Creating new columns for start and end dates
    dsStartTmp = pd.DataFrame()
    dsEndTmp = pd.DataFrame()
    
    dsStartTmp['year'] = ds['Year']
    dsStartTmp['month'] = ds['Start Month']
    dsStartTmp['day'] = ds['Start Day']
    
    dsEndTmp['year'] = ds['End Year']
    dsEndTmp['month'] = ds['End Month']
    dsEndTmp['day'] = ds['End Day']

    # Converting dates to datetime objects
    ds['Start Date'] = pd.to_datetime(dsStartTmp[['year', 'month', 'day']], errors='coerce')
    ds['End Date'] = pd.to_datetime(dsEndTmp[['year', 'month', 'day']], errors='coerce')

    # Fill NaT values with a default date
    default_date = pd.to_datetime('1900-01-01')
    ds['Start Date'].fillna(default_date,inplace=True)
    ds['End Date'].fillna(ds['Start Date'], inplace=True)

    # Calculating duration for each event
    ds['Duration'] = (ds['End Date'] - ds['Start Date']).dt.days
    
    #### Handling Lethality Column #### 
    # Creating binary variable 'Lethality' indicating if deaths were recorded
    ds['Lethality'] = (ds['Total Deaths'] > 0).astype(int)

    #### Handling Other Columns #### 

    ## Mapping string values to integers for categorical variables ##
    type_mapping = {type_str: idx + 1 for idx, type_str in enumerate(ds['Disaster Type'].unique())}
    ds['Disaster Type'] = ds['Disaster Type'].replace(type_mapping)

    type_mapping = {type_str: idx + 1 for idx, type_str in enumerate(ds['Disaster Subgroup'].unique())}
    ds['Disaster Subgroup'] = ds['Disaster Subgroup'].replace(type_mapping)

    type_mapping = {type_str: idx + 1 for idx, type_str in enumerate(ds['Disaster Subtype'].unique())}
    ds['Disaster Subtype'] = ds['Disaster Subtype'].replace(type_mapping)

    type_mapping = {type_str: idx + 1 for idx, type_str in enumerate(ds['ISO'].unique())}
    ds['ISO'] = ds['ISO'].replace(type_mapping)

    type_mapping = {type_str: idx + 1 for idx, type_str in enumerate(ds['Region'].unique())}
    ds['Region'] = ds['Region'].replace(type_mapping)

    type_mapping = {type_str: idx + 1 for idx, type_str in enumerate(ds['Continent'].unique())}
    ds['Continent'] = ds['Continent'].replace(type_mapping)

    # Changing Types 
    ds['Total Deaths'] = ds['Total Deaths'].astype(int)
    ds['No Injured'] = ds['No Injured'].astype(int)
    ds['No Affected'] = ds['No Affected'].astype(int)
    ds['Total Affected'] = ds['Total Affected'].astype(int)

    # Dropping unnecessary columns
    ds.drop(['Start Month', 'Start Day','End Year','End Month','End Day','Start Date','End Date','Event Name'],axis=1,inplace=True)

    return ds

def getInfoDataSet(ds):
    """
    Function to display information about the dataset.

    Prints out information such as data types, memory usage, and the number of non-null values for each column.

    Args:
    ds (pandas.DataFrame): The dataset to display information for.
    """
    ds.info()

def separateValuesRegression(ds):
    """
    Function to separate features and target variable for regression.

    Separates the features (independent variables) and the target variable (Total Deaths) for regression analysis.

    Args:
    ds (pandas.DataFrame): The dataset containing the features and target variable.

    Returns:
    tuple: A tuple containing the features (X) and target variable (y) for regression analysis.
    """
    # Make a copy of the dataset to avoid modification of the original dataset
    dsTmp = ds.copy()
    # Extract target variable (Total Deaths) where Total Deaths is not equal to 0
    y = dsTmp[dsTmp['Total Deaths'] != 0]['Total Deaths'].values
    # Extract features (independent variables) excluding 'Total Deaths', 'Total Affected', 'No Affected', and 'No Injured' columns
    x = dsTmp[dsTmp['Total Deaths'] != 0].drop(['Total Deaths','Total Affected','No Affected','No Injured'], axis=1)
    return x,y

def separateValuesClassification(ds):
    """
    Function to separate features and target variable for classification.

    Separates the features (independent variables) and the binary target variable (Lethality) for classification analysis.

    Args:
    ds (pandas.DataFrame): The dataset containing the features and target variable.

    Returns:
    tuple: A tuple containing the features (X) and binary target variable (y) for classification analysis.
    """
    # Extract features (independent variables) excluding 'Total Deaths', 'Total Affected', 'No Affected', 'No Injured', and 'Lethality' columns
    x = ds.drop(['Total Deaths','Total Affected','No Affected','No Injured','Lethality'],axis=1)
    # Extract binary target variable 'Lethality'
    y = ds['Lethality'].values
    return x,y

def initTraining(x,y):
    """
    Function to initialize training and testing data.

    Splits the dataset into training and testing sets for model evaluation.

    Args:
    x (numpy.ndarray): The features (independent variables) for training and testing.
    y (numpy.ndarray): The target variable for training and testing.

    Returns:
    tuple: A tuple containing the training and testing features and target variable.
    """
    # Split dataset into training and testing sets with 75% for training and 25% for testing
    Xtrain, Xtest, ytrain, ytest = train_test_split(x, y,test_size=0.25, random_state=0)
    
    # Convert training and testing features to numpy arrays
    Xtrain = Xtrain.values
    Xtest = Xtest.values

    # Ensure feature arrays have 2 dimensions
    if len(Xtrain.shape) < 2:
        Xtrain = Xtrain.reshape(-1, 1)
    if len(Xtest.shape) < 2:
        Xtest = Xtest.reshape(-1, 1)

    return Xtrain,Xtest,ytrain,ytest