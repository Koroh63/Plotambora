from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from utils import * 



def main(): 
    fullDataSet = importCleanDataSet()

    getInfoDataSet(fullDataSet)


    X,Y = separateValuesTD(fullDataSet)
    
    Xtrain,Xtest,ytrain,ytest = initTraining(X,Y)

    modelLinearRegression = LinearRegression()
    modelLasso = Lasso(alpha=5.0,max_iter=100000)
    modelElasticNet = ElasticNet(alpha=5.0,max_iter=100000)

    #modelLinearRegression.fit(Xtrain,ytrain)
    modelLasso.fit(Xtrain,ytrain)
    modelElasticNet.fit(Xtrain,ytrain)

    #ypreditLineatRegression = modelLinearRegression.predict(Xtest)
    ypreditLasso = modelLasso.predict(Xtest)
    ypreditElasticNet = modelElasticNet.predict(Xtest)
    # print(ypredit)
    #print(accuracy_score(ytest, ypreditLineatRegression))
    print(accuracy_score(ytest, ypreditLasso))
    print(accuracy_score(ytest, ypreditElasticNet))

    
    


  
# Using the special variable  
# __name__ 
if __name__=="__main__": 
    main() 

