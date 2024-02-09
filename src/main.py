from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from utils import * 
from sklearn.metrics import mean_squared_error,r2_score




def main(): 
    fullDataSet = importCleanDataSet()

    #showHist(fullDataSet['Total Deaths'])

    getInfoDataSet(fullDataSet)

    X,Y = separateValuesRegression(fullDataSet)
    
    Xtrain,Xtest,ytrain,ytest = initTraining(X,Y)

    modelLinearRegression = LinearRegression()
    modelLasso = Lasso(alpha=1.0,max_iter=10000)
    modelElasticNet = ElasticNet(alpha=1.0,max_iter=10000)

    modelLinearRegression.fit(Xtrain,ytrain)
    modelLasso.fit(Xtrain,ytrain)
    modelElasticNet.fit(Xtrain,ytrain)

    ypreditLineatRegression = modelLinearRegression.predict(Xtest)
    ypreditLasso = modelLasso.predict(Xtest)
    ypreditElasticNet = modelElasticNet.predict(Xtest)

    
    
    mse_linear_regression = mean_squared_error(ytest, ypreditLineatRegression)
    mse_lasso = mean_squared_error(ytest, ypreditLasso)
    mse_elastic_net = mean_squared_error(ytest, ypreditElasticNet)
    
    r2_linear_regression = r2_score(ytest, ypreditLineatRegression)
    #r2_lasso = r2_score(ytest, ypreditLasso)
    r2_elastic_net = r2_score(ytest, ypreditElasticNet)



    print( "-- Linear Regression : ")
    print("Mean Squared Error - Linear Regression:", mse_linear_regression)
    print("R2 Score - Linear Regression:", r2_linear_regression)
    print(np.mean((ypreditLineatRegression/ytest)*100) , "% d'erreur moyenne ")

    # print( "-- Lasso : ")
    # print("Mean Squared Error - Lasso:", mse_lasso)
    # print("R2 Score - Lasso:", r2_lasso)
    # print(np.mean((ypreditLasso/ytest)*100) , "% d'erreur moyenne ")

    print( "-- ElasticNet : ")
    print("Mean Squared Error - ElasticNet:", mse_elastic_net)
    print("R2 Score - ElasticNet:", r2_elastic_net)
    print(np.mean((ypreditElasticNet/ytest)*100) , "% d'erreur moyenne ")
    #r2

# standart scaler
    plt.figure(figsize=(10, 6))
    plt.hist(fullDataSet['Total Deaths'], bins=100, color='skyblue', edgecolor='black')
    plt.title('Histogramme des valeurs de Total Deaths')
    plt.xlabel('Total Deaths')
    plt.ylabel('Fréquence')
    plt.grid(True)
    plt.show()


  
# Using the special variable  
# __name__ 
if __name__=="__main__": 
    main() 

