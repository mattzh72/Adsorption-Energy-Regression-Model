from ase.db import connect
import numpy as np

#Extracts molecular data
#Returns an array of dictionaries containing molecular formula, positions, and energies
#Indices
def extract_molecular_data(dbName):
    db = connect(dbName)

    data = []
    
    for row in db.select(relaxed = True):
        atoms = []
        for i in range(len(row.numbers)):
            atom = {'num': row.numbers[i], 'position':row.positions[i]}
            atoms.append(atom)
            
        molecule = {'atoms': atoms, 'energy': row.energy}
        data.append(molecule)

    return data


#Extracts molecule data
#Returns as an array of arrays 
#Inner arrays are organized as such:
#index 0 contains formula, index 2 contains array of distances between hydrogen and other atoms, index 3 contains energy value
def extract_molecular_distances(dbName):
    db = connect(dbName)

    molecules = []
    
    for row in db.select(relaxed = True):
        if (row.energy < 100):
            molecule = []
            molecule.append(row.formula)
            molecule.append(calculateDistances(row.positions[-1], row.positions))
            molecule.append(row.energy)  
            molecules.append(molecule)
        
    return molecules


#Calculates distance between origin and all the points in array points
def calculateDistances(origin, points):
    distances = []
    
    for point in points:
        distances.append(calculateDistance(origin, point))
    
    return distances[:-1]

#Calculates the distance between 2 3D points
def calculateDistance(p1, p2):
    distance = None
    squared_deltas = []
    
    for i in range(len(p1)):
        delta = math.fabs(np.subtract(p1[i], p2[i]))
        delta_squared = delta * delta
        squared_deltas.append(delta_squared)
               
    delta_sums = 0
    
    for delta in squared_deltas:
        delta_sums += delta
        
    return math.sqrt(delta_sums)
        
    
        
        


    
