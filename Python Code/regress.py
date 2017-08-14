import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.linear_model import BayesianRidge, LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score

from sklearn.neighbors import KNeighborsClassifier


##A Simple linear Regression
def regress_simple(X, y):
    # Split the data into training/testing sets
    X_train = X[:-20]
    X_test = X[-20:]

    # Split the targets into training/testing sets
    y_train = y[:-20]
    y_test = y[-20:]  
    
    # Create linear regression object
    regr = linear_model.LinearRegression()

    # Train the model using the training sets
    regr.fit(X, y)

#    # The coefficients
#    print('Coefficients: \n', regr.coef_)
#    # The mean squared error
#    print("Mean squared error: %.2f"
#          % np.mean((regr.predict(X_test) - y_test) ** 2))
#    # Explained variance score: 1 is perfect prediction
#    print('Variance score: %.2f' % regr.score(X_test, y_test))
#    print(regr.score(X_test, y_test))
#
#    # Plot outputs
#    plt.scatter(1, y_test,  color='black')
#    plt.plot(X_test, regr.predict(X_test), color='blue', linewidth=3)
#
#    plt.xticks(())
#    plt.yticks(())
#
#    plt.show()
    

##A Bayesian Ridge Linear Regression
def regress_Bayesian_ridge(X, y):
    # Split the data into training/testing sets
    X_train = X[:-10]
    X_test = X[-10:]

    # Split the targets into training/testing sets
    y_train = y[:-10]
    y_test = y[-10:]  
    
    clf = linear_model.BayesianRidge()
    clf.fit(X_train, y_train)
    print(clf.score(X_test, y_test))
    
    ols = LinearRegression()
    ols.fit(X_train, y_train)
    print(ols.score(X_test, y_test))
    
    lw = 2
    plt.figure(figsize=(6, 5))
    plt.title("Weights of the model")
#    plt.plot(y, color='gold', linewidth=lw, label="Ground truth")
    plt.plot(clf.coef_, color='lightgreen', linewidth=lw,
             label="Bayesian Ridge estimate")
    plt.plot(ols.coef_, color='navy', linestyle='--', label="OLS estimate")
    plt.xlabel("Molecules")
    plt.ylabel("Adsorption Energy")
    plt.legend(loc="best", prop=dict(size=12))

    plt.show()

##A knn Regression
def regress_knn(X, y):
    SPLIT_FACTOR = 10
    
    # Split the data into training/testing sets
    X_train = X[:-SPLIT_FACTOR]
    X_test = X[-SPLIT_FACTOR:]

    # Split the targets into training/testing sets
    y_train = y[:-SPLIT_FACTOR]
    y_test = y[-SPLIT_FACTOR:]  
    
    # Create knn regression object
    regr = KNeighborsRegressor(61, "distance")
    # Train the model using the training sets
    regr.fit(X_train, y_train)

    print("Mean squared error: %.2f" % np.mean((regr.predict(X_test) - y_test) ** 2))
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % regr.score(X_test, y_test))
    print(regr.score(X_test, y_test))
    
    return regr.score(X_test, y_test)
    
def classify_knn(X, y):
    neigh = KNeighborsClassifier(n_neighbors=61)
    neigh.fit(X, y) 
    print(neigh.score(X, y))
    
    