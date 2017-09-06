import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.linear_model import BayesianRidge, LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score
from sklearn import cross_validation
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.grid_search import GridSearchCV
from sklearn.grid_search import RandomizedSearchCV

parameter_candidates = [
  {'gamma': [0.001, 0.0001], 'kernel': ['rbf', 'laplacian'], 'alpha': [0.0001,0.001,0.01,0.1,1]},
]


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
    
##A Bayesian Ridge Linear Regression
def regress_Bayesian_ridge(X, y):
    # Split the data into training/testing sets
    X_train = X[:-10]
    X_test = X[-10:]

    # Split the targets into training/testing sets
    y_train = y[:-10]
    y_test = y[-10:]  
    
    clf = linear_model.BayesianRidge(n_iter=300, tol=0.00001, alpha_1=200, alpha_2=200, fit_intercept=False, normalize=True, copy_X=True, verbose=True)
    scores = cross_validation.cross_val_score(clf, X, y, cv=5)
    print(scores)
    print ("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() / 2))
    
def regress_SVR(X, y, k):
    svr = SVR(kernel=k, degree=5)
    scores = cross_validation.cross_val_score(svr, X, y, cv=3)
    print(scores)
    print ("Accuracy: %0.2f \(+/- %0.2f\)" % (scores.mean(), scores.std() / 2))
    
def kernel_ridge_regress(X, y):
    clf = GridSearchCV(estimator=KernelRidge(),param_grid=parameter_candidates)
    clf.fit(X, y)
    print(clf.score(X, y))

"""
    print('Best score:', clf.best_score_)
    print('Best Kernel:', clf.best_estimator_.kernel)
    print('Best Gamma:', clf.best_estimator_.gamma)
    print('Best Alpha:', clf.best_estimator_.alpha)
"""
    
#    scores = cross_validation.cross_val_score(clf, X, y, cv=2)
#    print(scores)
#    print "Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() / 2)

##A knn Regression
def regress_knn(X, y):
    SPLIT_FACTOR = 50
    
    # Split the data into training/testing sets
    X_train = X[:-SPLIT_FACTOR]
    X_test = X[-SPLIT_FACTOR:]

    # Split the targets into training/testing sets
    y_train = y[:-SPLIT_FACTOR]
    y_test = y[-SPLIT_FACTOR:]  
    
    # Create knn regression object
    regr = KNeighborsRegressor(61, "distance")

    scores = cross_validation.cross_val_score(regr, X, y, cv=5)
    print(scores)
    print ("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() / 2))
    
    
    
