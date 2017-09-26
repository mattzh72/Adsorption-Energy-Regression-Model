from extract import extract_molecular_data
from preprocess import encode_data
from preprocess import label_data

from regress import *

from utils import *

from feat import featurize
from feat import extractTarget

import numpy as np
import pickle    

import scipy.stats as spst

""" save/load data
if savedAlready, load from saved pickle file
else extract from ase DB
"""
savedAlready = True
if savedAlready == False:
    data = extract_molecular_data('dE_H_1k.db', dx=1, useAseDistance=True, filterSigma=0)
    with open('data.pickle', 'wb') as fp:
        pickle.dump(data, fp)
else:
    with open('data.pickle', 'rb') as fp:
        data = pickle.load(fp)

# featurize the data        
X = featurize(data, coulomb_eigen=False, coulomb_random_samples=5)
y = extractTarget(data)

#param_dist_kernel = {'gamma': spst.expon(scale=0.0001), 'alpha': spst.expon(scale=0.000001)}
#param_dist_SVR = {'C': spst.expon(scale=1), 'gamma': spst.expon(scale=0.000001), 'epsilon': spst.expon(0.001)}
#param_dist_Bayesian_ridge = {'alpha_1':spst.expon(scale=1e-10), 'alpha_2':spst.expon(scale=1e-10), 'lambda_1': spst.expon(scale=1e-4), 'lambda_2': spst.expon(scale=1e-10)}

# run regression model
#dummy_regressor(X, y)
regress_simple(X, y)
#regress_Bayesian_ridge(X, y, alpha_2=3.218290550214049e-05, lambda_1=6.633380980280872e-05, lambda_2=6.312410500509716e-05, alpha_1=4.494947658665962e-05)
#regress_Bayesian_ridge_RandomSearchCV(X, y, param_dist_Bayesian_ridge)
#regress_knn(X,y)

##0.212755658269
#kernel_ridge_regress(X, y, alpha=0.006087531857835262, gamma=0.0007325085059569133)
##0.211291183164
#krr = kernel_ridge_regress(X, y, alpha=6.81901586710227e-06, gamma=0.000575920215580223)
#getPredictionsAndActual(krr, X, y)

#regress_ridge_RandomSearchCV(X, y, param_dist, n_iter_search=100)

#kernel_RBF = C(constant_value=300) + WhiteKernel(noise_level=0.025) + RBF(length_scale=1050) 
#kernel_Matern = C(constant_value=1e2) + Matern(length_scale=2500, nu=0.2)
#kernel_Matern_RBF = C(constant_value=1e2) + Matern(length_scale=2500, nu=0.2) + RBF(length_scale=3.3e3)
#gp_regress(X, y, kernel_Matern)

"""

Best MAE: -0.238425368081
Best Parameters: {'epsilon': 0.10871593459397691, 'C': 1.524529776443795, 'gamma': 8.887647378544308e-07}

"""

#SVR_regress(X, y, C=1.524529776443795, gamma=8.887647378544308e-07, epsilon=0.10871593459397691)
#SVR_RandomSearchCV(X, y, param_dist_SVR)
