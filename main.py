from extract import extract_molecular_data
from preprocess import encode_data
from preprocess import label_data

from regress import *

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
X = featurize(data, coulomb_eigen=True, coulomb_random_samples=0)
y = extractTarget(data)

#param_dist_kernel = {'gamma': spst.expon(scale=0.0001), 'alpha': spst.expon(scale=0.000001)}
param_dist_SVR = {'C': spst.expon(scale=1), 'gamma': spst.expon(scale=0.000001), 'epsilon': spst.expon(0.001)}


#print(X[0])

# run regression model
#dummy_regressor(X, y)
#regress_simple(X, y)
#regress_Bayesian_ridge(X, y)
#regress_knn(X,y)

##0.212755658269
#kernel_ridge_regress(X, y, alpha=0.006087531857835262, gamma=0.0007325085059569133)
##0.211291183164
#kernel_ridge_regress(X, y, alpha=6.81901586710227e-06, gamma=0.000575920215580223)

#regress_ridge_RandomSearchCV(X, y, param_dist, n_iter_search=100)

#kernel_RBF = C(constant_value=300) + WhiteKernel(noise_level=0.025) + RBF(length_scale=1050) 
#kernel_Matern = C(constant_value=1e2) + Matern(length_scale=2500, nu=0.2)
#kernel_Matern_RBF = C(constant_value=1e2) + Matern(length_scale=2500, nu=0.2) + RBF(length_scale=3.3e3)
#gp_regress(X, y, kernel_Matern_RBF)

"""

Best MAE: -0.238425368081
Best Parameters: {'epsilon': 0.10871593459397691, 'C': 1.524529776443795, 'gamma': 8.887647378544308e-07}

"""

#SVR_regress(X, y, C=1.524529776443795, gamma=8.887647378544308e-07, epsilon=0.10871593459397691)
#SVR_RandomSearchCV(X, y, param_dist_SVR)
