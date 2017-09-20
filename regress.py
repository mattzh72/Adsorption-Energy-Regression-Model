import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from utils import *
from sklearn.dummy import DummyRegressor
from sklearn import linear_model
from sklearn.linear_model import BayesianRidge, LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import RandomizedSearchCV
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern, ConstantKernel as C

def dummy_regressor(X, y):
    regr = DummyRegressor()
    
    scores = cross_val_score(regr, X, y, scoring='neg_mean_absolute_error', cv=10)
    print(scores)
    print ("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() / 2))
    

##A Simple linear Regression
def regress_simple(X, y):
    # Create linear regression object
    regr = linear_model.LinearRegression()

    scores = cross_val_score(regr, X, y, scoring='neg_mean_absolute_error', cv=10)
    print(scores)
    print ("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() / 2))

    
##A Bayesian Ridge Linear Regression
def regress_Bayesian_ridge(X, y):
    clf = linear_model.BayesianRidge(n_iter=300, tol=0.00001, alpha_1=200, alpha_2=200, fit_intercept=False, normalize=True, copy_X=True, verbose=True)
    scores = cross_val_score(clf, X, y, scoring='neg_mean_absolute_error', cv=10)
    print(scores)
    print ("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() / 2))
        
##A knn Regression
def regress_knn(X, y):
    # Create knn regression object
    regr = KNeighborsRegressor(869, "distance")

    scores = cross_val_score(regr, X, y, scoring='neg_mean_absolute_error', cv=10)
    print(scores)
    print ("Accuracy: %s (+/- %s)" % (scores.mean() * -1, scores.std() / 2))
    

def kernel_ridge_regress(X, y, kernel="laplacian", alpha=5e-4, gamma=0.008):
    # initiate kernel ridge
    regr = KernelRidge(kernel=kernel, alpha=alpha, gamma=gamma)
    
    scores = cross_val_score(regr, X, y, scoring='neg_mean_absolute_error', cv=10)
    print(scores)
    print ("Accuracy: %s (+/- %s)" % (scores.mean() * -1, -scores.std()))
    
    

def regress_ridge_RandomSearchCV(X, y, param_dist, n_iter_search=100):
    clf = KernelRidge(kernel='laplacian')
    
    r_search = RandomizedSearchCV(clf,param_distributions=param_dist, n_iter=n_iter_search, cv=10, scoring='neg_mean_absolute_error')
    r_search.fit(X, y)
    
    print("Best MAE: %s" % r_search.best_score_)
    print("Best Parameters: %s" %r_search.best_params_)
    

def gp_regress(X, y, kernel=None): 
    gp = GaussianProcessRegressor(alpha = 0.00073, normalize_y=True)
    
    if (kernel != None):
        gp = GaussianProcessRegressor(alpha = 0.00073, kernel=kernel, normalize_y=True, optimizer=None)
    
    scores = cross_val_score(gp, X, y,scoring='neg_mean_absolute_error', cv=10)
    print(scores)
    print ("Accuracy: %s (+/- %s)" % (scores.mean() * -1, -scores.std()))
    
def SVR_regress(X, y, kernel='rbf', C=1e1, gamma=0.1, tol=1e-5, epsilon=0.1):
    svr = SVR(kernel='rbf', C=C, gamma=gamma, epsilon=epsilon, tol=tol)

    scores = cross_val_score(svr, X, y,scoring='neg_mean_absolute_error', cv=10)
    print(scores)
    print ("Accuracy: %s (+/- %s)" % (scores.mean() * -1, -scores.std()))
    
def SVR_RandomSearchCV(X, y, param_dist, n_iter_search=100):
    svr = SVR(kernel='rbf', tol=1e-5)
    
    r_search = RandomizedSearchCV(svr,param_distributions=param_dist, n_iter=n_iter_search, cv=10, scoring='neg_mean_absolute_error')
    r_search.fit(X, y)
    
    print("Best MAE: %s" % r_search.best_score_)
    print("Best Parameters: %s" %r_search.best_params_)

    
    
