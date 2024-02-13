"""
@file
@brief This file contains the main function to train regression and classification models on the cleaned dataset.

@author: RICHARD Corentin & HODIN Dorian
"""

# Import necessary libraries
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC 
from utils import * 
from sklearn.metrics import mean_squared_error,r2_score


def main(): 
    """
    Main function to train regression and classification models on the cleaned dataset.
    """

    # Import and clean the dataset
    fullDataSet = importCleanDataSet()

    # Display information about the dataset
    getInfoDataSet(fullDataSet)

    # User choice for regression or classification
    choice = 0
    while choice!=1 and choice!=2 and choice!=9:
        choice = int(input("\nChoisissez le type de modèle à utiliser:\n\n- 1. Régression\n- 2. Classification\n\n- 9. Sortir\n\nEntrez votre choix (1 ou 2): "))

    if choice == 1:
        ### Regression Menu ###
        print (" ----------- Régréssion des totaux de morts -----------")
        
        # Separate features and target variable for regression
        X,Y = separateValuesRegression(fullDataSet)
        
        # Initialize training and testing data
        Xtrain,Xtest,ytrain,ytest = initTraining(X,Y)

        # Initialize regression models
        modelLinearRegression = LinearRegression()
        modelLasso = Lasso(alpha=1.0,max_iter=10000)
        modelElasticNet = ElasticNet(alpha=1.0,max_iter=10000)

        # Fit regression models
        modelLinearRegression.fit(Xtrain,ytrain)
        modelLasso.fit(Xtrain,ytrain)
        modelElasticNet.fit(Xtrain,ytrain)

        # Predictions
        ypreditLineatRegression = modelLinearRegression.predict(Xtest)
        ypreditLasso = modelLasso.predict(Xtest)
        ypreditElasticNet = modelElasticNet.predict(Xtest)

        # Calculate Mean Squared Error and R2 Score
        mse_linear_regression = mean_squared_error(ytest, ypreditLineatRegression)
        mse_lasso = mean_squared_error(ytest, ypreditLasso)
        mse_elastic_net = mean_squared_error(ytest, ypreditElasticNet)
        
        r2_linear_regression = r2_score(ytest, ypreditLineatRegression)
        r2_lasso = r2_score(ytest, ypreditLasso)
        r2_elastic_net = r2_score(ytest, ypreditElasticNet)


        # Print results
        print( "-- Linear Regression : ")
        print("Mean Squared Error - Linear Regression:", mse_linear_regression)
        print("R2 Score - Linear Regression:", r2_linear_regression)

        print( "-- Lasso : ")
        print("Mean Squared Error - Lasso:", mse_lasso)
        print("R2 Score - Lasso:", r2_lasso)

        print( "-- ElasticNet : ")
        print("Mean Squared Error - ElasticNet:", mse_elastic_net)
        print("R2 Score - ElasticNet:", r2_elastic_net)

        # Display histogram of 'Total Deaths' values
        plt.figure(figsize=(10, 6))
        plt.hist(fullDataSet['Total Deaths'], bins=100, color='skyblue', edgecolor='black')
        plt.title('Histogramme des valeurs de Total Deaths')
        plt.xlabel('Total Deaths')
        plt.ylabel('Fréquence')
        plt.grid(True)
        plt.show()
    
    if choice == 2:
        print (" ----------- Classification de la Léthalité -----------")

        # Separate features and target variable for classification
        X,Y = separateValuesClassification(fullDataSet)

        # Initialize training and testing data
        Xtrain,Xtest,ytrain,ytest = initTraining(X,Y)

        # Initialize classification models
        modelSVC = SVC(C=5.0, kernel='rbf', gamma='scale')
        modelLogisticRegression = LogisticRegression(C=5.0, solver='lbfgs', max_iter=10000)

        
        # Fit classification models
        modelSVC.fit(Xtrain,ytrain)
        modelLogisticRegression.fit(Xtrain,ytrain)

        # Predict using SVC
        yPreditSVC = modelSVC.predict(Xtest)
        yPreditLogistic= modelLogisticRegression.predict(Xtest)

        # Calculate accuracy score for SVC
        print('SVC Accuracy Score : ',accuracy_score(yPreditSVC,ytest))
        print('Logistic Regression Accuracy Score : ',accuracy_score(yPreditLogistic,ytest))

  
# Entry point of the script
if __name__=="__main__": 
    main() 

