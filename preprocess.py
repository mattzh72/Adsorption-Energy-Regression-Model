from sklearn import preprocessing
import numpy as np

##One hot encodes the molecular formulas
def encode_data(data):
    labeledData = label_data(data) 
    labeledData = labeledData.reshape(-1, 1)

    ##Create one hot encoder
    encoder = preprocessing.OneHotEncoder()
    encodedData = encoder.fit_transform(labeledData).toarray()
    
    return encodedData

def label_data(data):
    le = preprocessing.LabelEncoder()
    labeledData = le.fit_transform(data)
    labeledData = labeledData.reshape(-1, 1) #reshape because passing 1d arrays to hot encoding is   deprecated
    
    return labeledData
    
    

