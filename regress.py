import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from utils import *
from sklearn import datasets, linear_model
from sklearn.linear_model import BayesianRidge, LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn import cross_validation
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.grid_search import GridSearchCV
from sklearn.grid_search import RandomizedSearchCV

##A Simple linear Regression
def regress_simple(X, y):
    # Split the data into training/testing sets
    X_train = X[:-100]
    X_test = X[-100:]

    # Split the targets into training/testing sets
    y_train = y[:-100]
    y_test = y[-100:]  
    
    # Create linear regression object
    regr = linear_model.LinearRegression()

    # Train the model using the training sets
    #regr.fit(X, y)

    scores = cross_validation.cross_val_score(regr, X, y, scoring='neg_mean_absolute_error')
    print(scores)
    
##A Bayesian Ridge Linear Regression
def regress_Bayesian_ridge(X, y):
    # Split the data into training/testing sets
    X_train = X[:-100]
    X_test = X[-100:]

    # Split the targets into training/testing sets
    y_train = y[:-100]
    y_test = y[-100:]  
    
    clf = linear_model.BayesianRidge(n_iter=300, tol=0.00001, alpha_1=200, alpha_2=200, fit_intercept=False, normalize=True, copy_X=True, verbose=True)
    scores = cross_validation.cross_val_score(clf, X, y, scoring='neg_mean_absolute_error', cv=5)
    print(scores)
#    print ("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() / 2))
    
def regress_SVR(X, y, k):
    svr = SVR(kernel=k, degree=5)
    scores = cross_validation.cross_val_score(svr, X, y, scoring='neg_mean_absolute_error', cv=3)
    print(scores)
    print ("Accuracy: %0.2f \(+/- %0.2f\)" % (scores.mean(), scores.std() / 2))
        
##A knn Regression
def regress_knn(X, y):
    SPLIT_FACTOR = 100
    
    # Split the data into training/testing sets
    X_train = X[:-SPLIT_FACTOR]
    X_test = X[-SPLIT_FACTOR:]
    y_train = y[:-SPLIT_FACTOR]
    y_test = y[-SPLIT_FACTOR:]  
    
    # Create knn regression object
    regr = KNeighborsRegressor(899, "distance")

    regr.fit(X_train, y_train)

    # check score
    regress_score(regr, X_train, y_train, 'training')
    regress_score(regr, X_test,  y_test,  'testing')
    

def kernel_ridge_regress(X, y, kernel="laplacian", alpha=0.0554063818372217965, gamma=0.01234923715677394):
    # initiate kernel ridge
    regr = KernelRidge(kernel=kernel, alpha=alpha, gamma=gamma)
    
    scores = cross_validation.cross_val_score(regr, X, y, scoring='neg_mean_absolute_error', cv=10)
    print(scores)
    print ("Accuracy: %0.01f (+/- %0.01f)" % (scores.mean() * -1, scores.std() / 2))
    
    

def regress_ridge_RandomSearchCV(X, y, param_dist, n_iter_search=100):
    clf = KernelRidge()
    
    r_search = RandomizedSearchCV(clf,param_distributions=param_dist, n_iter=n_iter_search, cv=10, scoring='neg_mean_absolute_error')
    r_search.fit(X, y)
    
    print("Best MAE: %0.01f" % r_search.best_score_)
    print("Best Parameters: %s" %r_search.best_params_)
    
    
""" 
report regression r2_score and mean_absolute_error score
"""    
def regress_score(regr, X, y, header):
        
    # predict
    y_pred = regr.predict(X)

    r2Score = r2_score(y, y_pred)
    meanAbsScore = mean_absolute_error(y, y_pred)
    print("%s r2_score:%10.6f mean_absolute_error:%10.6f" % (header, r2Score, meanAbsScore))


    
    
