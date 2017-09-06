from ase.db import connect
import numpy as np
import math

#Extracts molecular data
#Returns an array of dictionaries containing molecular formula, positions, and energies
#Indices
def extract_molecular_data(dbName, dx=0):
    db = connect(dbName)

    data = []
    
    for row in db.select(relaxed = True):
        atoms = []
        sz = len(row.numbers)
        hydrogenIdx = sz - 1 # hydrogen is always the last one
        for i in range(sz):
            if dx > 0 and calculate_distance(row, i, hydrogenIdx) > dx:
                continue
            atom = {'num': row.numbers[i], 'position':row.positions[i]}
            atoms.append(atom)
        print("orig size: %d  after filtering %d" % (sz, len(atoms)))
#        print(atoms)
        molecule = {'atoms': atoms, 'energy': row.energy}
        data.append(molecule)

    return data

def calculate_distance(row, atomA, atomB):
    if atomA == atomB:
        return 0
    p1 = row.positions[atomA]
    p2 = row.positions[atomB]
    d = 0.0;
    for i in range(3):
        d = d + (p1[i]-p2[i])**2
#    print ("dist %5.2f" % (math.sqrt(d)))
    return math.sqrt(d) 

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
        
    
        
        


    
