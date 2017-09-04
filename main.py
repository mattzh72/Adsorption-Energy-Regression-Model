from extract import extract_molecular_data, extract_molecular_distances
from preprocess import encode_data
from preprocess import label_data

from regress import regress_knn
from regress import regress_simple
from regress import regress_Bayesian_ridge
from regress import regress_SVR
from regress import kernel_ridge_regress

from feat import coulomb_matrix
from feat import calculate_eigenvalues
from feat import featurize
from feat import extractTarget

import numpy as np

##Get data
data = extract_molecular_data('dE_H_1k.db')
X = featurize(data)
y = extractTarget(data)

kernel_ridge_regress(X, y)



