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
    print ("Accuracy: %s (+/- %s)" % (scores.mean() * -1, -scores.std()))
    
    return regr
    

##A Simple linear Regression
def regress_simple(X, y):
    # Create linear regression object
    regr = linear_model.LinearRegression()

    scores = cross_val_score(regr, X, y, scoring='neg_mean_absolute_error', cv=10)
    print(scores)
    print ("Accuracy: %s (+/- %s)" % (scores.mean() * -1, -scores.std()))
    
    return regr

    
##A Bayesian Ridge Linear Regression
def regress_Bayesian_ridge(X, y, alpha_1=1e-6,alpha_2=1e-6,lambda_1=1e-6,lambda_2=1e-6,):
    regr = linear_model.BayesianRidge(n_iter=300, tol=0.00001, alpha_1=alpha_1, alpha_2=alpha_2, lambda_1=lambda_1, lambda_2=lambda_2, fit_intercept=False, normalize=True, copy_X=True, verbose=True)
    
    scores = cross_val_score(regr, X, y, scoring='neg_mean_absolute_error', cv=10)
    print(scores)
    print ("Accuracy: %s (+/- %s)" % (scores.mean() * -1, -scores.std()))
    
    return regr
    
def regress_Bayesian_ridge_RandomSearchCV(X, y, param_dist, n_iter_search=100):
    regr = linear_model.BayesianRidge(n_iter=300, tol=1e-10, fit_intercept=False)
    
    r_search = RandomizedSearchCV(regr,param_distributions=param_dist, n_iter=n_iter_search, cv=10, scoring='neg_mean_absolute_error')
    r_search.fit(X, y)
    
    print("Best MAE: %s" % r_search.best_score_)
    print("Best Parameters: %s" %r_search.best_params_)
    
        
##A knn Regression
def regress_knn(X, y):
    # Create knn regression object
    regr = KNeighborsRegressor(869, "distance")

    scores = cross_val_score(regr, X, y, scoring='neg_mean_absolute_error', cv=10)
    print(scores)
    print ("Accuracy: %s (+/- %s)" % (scores.mean() * -1, scores.std() / 2))
    
    return regr
    

def kernel_ridge_regress(X, y, kernel="laplacian", alpha=5e-4, gamma=0.008):
    # initiate kernel ridge
    regr = KernelRidge(kernel=kernel, alpha=alpha, gamma=gamma)
    
    scores = cross_val_score(regr, X, y, scoring='neg_mean_absolute_error', cv=10)
    print(scores)
    print ("Accuracy: %s (+/- %s)" % (scores.mean() * -1, -scores.std()))
    
    return regr
    
    

def regress_ridge_RandomSearchCV(X, y, param_dist, n_iter_search=100):
    regr = KernelRidge(kernel='laplacian')
    
    r_search = RandomizedSearchCV(regr,param_distributions=param_dist, n_iter=n_iter_search, cv=10, scoring='neg_mean_absolute_error')
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
    
    return gp
    
def SVR_regress(X, y, kernel='rbf', C=1e1, gamma=0.1, tol=1e-5, epsilon=0.1):
    svr = SVR(kernel='rbf', C=C, gamma=gamma, epsilon=epsilon, tol=tol)

    scores = cross_val_score(svr, X, y,scoring='neg_mean_absolute_error', cv=10)
    print(scores)
    print ("Accuracy: %s (+/- %s)" % (scores.mean() * -1, -scores.std()))
    
    return svr
    
def SVR_RandomSearchCV(X, y, param_dist, n_iter_search=100):
    svr = SVR(kernel='rbf', tol=1e-5)
    
    r_search = RandomizedSearchCV(svr,param_distributions=param_dist, n_iter=n_iter_search, cv=10, scoring='neg_mean_absolute_error')
    r_search.fit(X, y)
    
    print("Best MAE: %s" % r_search.best_score_)
    print("Best Parameters: %s" %r_search.best_params_)

    
    
