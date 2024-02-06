from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from utils import * 
from sklearn.metrics import mean_squared_error




def main(): 
    fullDataSet = importCleanDataSet()

    getInfoDataSet(fullDataSet)


    X,Y = separateValuesTD(fullDataSet)
    
    Xtrain,Xtest,ytrain,ytest = initTraining(X,Y)

    modelLinearRegression = LinearRegression()
    modelLasso = Lasso(alpha=5.0,max_iter=100000)
    modelElasticNet = ElasticNet(alpha=5.0,max_iter=100000)

    modelLinearRegression.fit(Xtrain,ytrain)
    modelLasso.fit(Xtrain,ytrain)
    modelElasticNet.fit(Xtrain,ytrain)

    ypreditLineatRegression = modelLinearRegression.predict(Xtest)
    ypreditLasso = modelLasso.predict(Xtest)
    ypreditElasticNet = modelElasticNet.predict(Xtest)
    # print(ypredit)
    mse_linear_regression = mean_squared_error(ytest, ypreditLineatRegression)
    mse_lasso = mean_squared_error(ytest, ypreditLasso)
    mse_elastic_net = mean_squared_error(ytest, ypreditElasticNet)

    print("Mean Squared Error - Linear Regression:", mse_linear_regression)
    print("Mean Squared Error - Lasso:", mse_lasso)
    print("Mean Squared Error - ElasticNet:", mse_elastic_net)
    


  
# Using the special variable  
# __name__ 
if __name__=="__main__": 
    main() 

