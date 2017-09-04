import numpy as np
import math

def removeOuts(molecules):
    srtdenergies = []
    for mol in molecules:
        srtdenergies.append(mol["energy"])
    qthree = np.percentile(srtdenergies,75,'midpoint')
    qone = xnp.percentile(srtdenergies,25,'midpoint')
    iqr = qthree-qone 
    determinant = 1.5*iqr
    
    newMolecules = []
    for mol in molecules (0,999):
        if( mol["energy"] > qone-determinant or mol["energy"] < qthree + determinant ):
            newMolecules.append[mol]
    
    return newMolecules
        

def removeOut(nums):
    noOut = []
    qthree = np.percentile(nums,75,'midpoint')
    qone = xnp.percentile(nums,25,'midpoint')
    iqr = qthree-qone 
    determinant = 1.5*iqr
    
    newNums = []
    for num in nums:
        if( num > qone-determinant or num < qthree + determinant ):
            newNums.append(num)
            
    return newNums

        


def removeOutliers(x, outlierConstant):
    a = np.array(x)
    upper_quartile = np.percentile(a, 75)
    lower_quartile = np.percentile(a, 25)
    IQR = (upper_quartile - lower_quartile) * outlierConstant
    quartileSet = (lower_quartile - IQR, upper_quartile + IQR)
    resultList = []
    for y in a.tolist():
        if y >= quartileSet[0] and y <= quartileSet[1]:
            resultList.append(y)
    return resultList


nums = [1,2.2,3,4.5,5,6.4,7,8,9,1000.99]
print (removeOutliers(nums, 1.5)) 