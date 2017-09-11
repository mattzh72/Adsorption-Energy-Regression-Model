from extract import extract_molecular_data, extract_molecular_distances
from preprocess import encode_data
from preprocess import label_data

from regress import *

from feat import featurize
from feat import extractTarget

import numpy as np
import pickle    

""" save/load data
if savedAlready, load from saved pickle file
else extract from ase DB
"""
savedAlready = True 
if savedAlready == False:
    data = extract_molecular_data('dE_H_1k.db', dx=1, useAseDistance=True)
    with open('data.pickle', 'wb') as fp:
        pickle.dump(data, fp)
else:
    with open('data.pickle', 'rb') as fp:
        data = pickle.load(fp)

# featurize the data        
X = featurize(data, coulomb_eigen=False)
y = extractTarget(data)

# run regression model
#regress_simple(X, y)
#regress_Bayesian_ridge(X, y)
#regress_knn(X,y)
#kernel_ridge_regress(X, y)
regress_ridge(X, y)
