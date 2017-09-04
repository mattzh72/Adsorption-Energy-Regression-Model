from extract import extract_molecular_data, extract_molecular_distances
from preprocess import encode_data
from preprocess import label_data

from regress import regress_knn
from regress import classify_knn
from regress import regress_simple
from regress import regress_Bayesian_ridge
from regress import regress_SVR
from regress import kernel_ridge_regress
import numpy as np


##Get data
data = extract_molecular_distances('dE_H_1k.db')

##Encode molecular formulas
formulas = []
for molecule in data:
    formulas.append(molecule[0])
encodedMolecules = encode_data(formulas)

#index 0 contains formula, index 2 contains array of distances between hydrogen and other atoms, index 3 contains energy value


## Set input and output
X = []
y = []
for i in range(len(data)):
    molecule = data[i]
    distances = molecule[1]
    
    totalDistance = 0
    for j in range(len(distances)):
        totalDistance = totalDistance + distances[j]
    
    
    averageDistance = totalDistance/len(distances)
    X.append(np.append(encodedMolecules[i], averageDistance))
    y.append(molecule[2])
    
#print(label_data(formulas))
#TODO: label_data produces array of floats, may not be compatible... 
#classify_knn(label_data(formulas), y)
kernel_ridge_regress(X, y)
    
#bestScore = 0
#bestBucket = 1
#for i in range(978):
#    print(i+1)
#    score = regress_knn(X, y, i + 1)
#    if (score > bestScore):
#        bestScore = score
#        bestBucket = i + 1
#        
#print(bestBucket)
#print(bestScore)


