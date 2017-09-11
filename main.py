from extract import extract_molecular_data, extract_molecular_distances
from preprocess import encode_data
from preprocess import label_data

from regress import *

from feat import featurize
from feat import extractTarget

import numpy as np
import pickle    

## save/load data
savedAlready = False 
if savedAlready == False:
    data = extract_molecular_data('dE_H_1k.db', dx=1, useAseDistance=True)
    with open('data.pickle', 'wb') as fp:
        pickle.dump(data, fp)
else:
    with open('data.pickle', 'rb') as fp:
        data = pickle.load(fp)

# featurize the data        
X = featurize(data, coulomb_eigen=True)
y = extractTarget(data)

# run regression model
#kernel_ridge_regress(X, y)
#regress_Bayesian_ridge(X, y)
#regress_knn(X,y)
#regress_simple(X, y)
regress_ridge(X, y)



