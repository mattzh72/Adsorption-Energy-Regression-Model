from extract import extract_molecular_data, extract_molecular_distances
from preprocess import encode_data
from regress import regress_simple
import numpy as np


##Get data
data = extract_molecular_distances('dE_H_1k.db')

##Encode molecular formulas
formulas = []
for molecule in data:
    formulas.append(molecule[0])
encodedMolecules = encode_data(formulas)

## Set input and output
##TODO: Find 2D matrix representation of structural data
X = []
y = []
#for i in range(len(data)):
#    molecule = data[i]
#    distances = molecule[1]
#    
#    while len(distances) < 40:
#        distances.append(0)
#        
#    X.append(distances)
#    y.append(molecule[2])
                      

regress_simple(X, y)


