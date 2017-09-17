from ase.db import connect
from ase import Atoms
import ase.db
from ase.data import covalent_radii
from ase.utils import basestring
from ase.constraints import FixAtoms
import itertools
import re
import numpy as np
import math

#Extracts molecular data
#Returns an array of dictionaries containing molecular formula, positions, and energies
#Indices
def extract_molecular_data(dbName, dx=0, useAseDistance=True, filterSigma=0):
    db = connect(dbName)

    # use ase get_mic_distance() to get neighboring atoms
    if useAseDistance == True:
        return get_molecular_aseDist(db, dx, filterSigma)
    
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
#        print("orig size: %d  after my filtering %d" % (sz, len(atoms)))
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
        
def get_mic_distance(p1, p2, cell, pbc):
    """ This method calculates the shortest distance between p1 and p2
         through the cell boundaries defined by cell and pbc.
         This method works for reasonable unit cells, but not for extremely
         elongated ones.
    """
    ct = cell.T
    pos = np.mat((p1, p2))
    scaled = np.linalg.solve(ct, pos.T).T
    for i in range(3):
        if pbc[i]:
            scaled[:, i] %= 1.0
            scaled[:, i] %= 1.0
    P = np.dot(scaled, cell)

    pbc_directions = [[-1, 1] * int(direction) + [0] for direction in pbc]
    translations = np.mat(list(itertools.product(*pbc_directions))).T
    p0r = np.tile(np.reshape(P[0, :], (3, 1)), (1, translations.shape[1]))
    p1r = np.tile(np.reshape(P[1, :], (3, 1)), (1, translations.shape[1]))
    dp_vec = p0r + ct * translations
    d = np.min(np.power(p1r - dp_vec, 2).sum(axis=0))**0.5
    return d

def get_atom_neighborlist(atoms, centerAtom, dx=0.2, no_count_types=None):
    """
    Method to get the a dict with list of neighboring
    atoms defined as the two covalent radii + fixed distance.
    Option added to remove neighbors between defined atom types.
    """
    cell = atoms.get_cell()
    pbc = atoms.get_pbc()

    if no_count_types is None:
        no_count_types = []

    conn = []
    atomi = centerAtom;
    for atomj in atoms:
        if atomi.index != atomj.index:
            if atomi.number not in no_count_types:
                if atomj.number not in no_count_types:
                    d = get_mic_distance(atomi.position,
                                         atomj.position,
                                         cell,
                                         pbc)
                    cri = covalent_radii[atomi.number]
                    crj = covalent_radii[atomj.number]
                    d_max = crj + cri + dx
                    if d < d_max:
                        conn.append(atomj)
    conn.append(centerAtom)
    return conn

# use ase get_mic_distance() to get neighboring atoms
def get_molecular_aseDist(db, dx, filterSigma=0):

    data = []

    for row in db.select(relaxed = True):
        atoms = row.toatoms()
        # H atom is always the last
        nbAtoms = get_atom_neighborlist(atoms, atoms[len(atoms)-1], dx=dx)

        atomData = []
        for a in nbAtoms:
            atomData.append( {'num': a.number, 'position':a.position} )
#        print("orig size: %d  after ase filtering %d" % (len(atoms), len(atomData)))
        molecule = {'atoms': atomData, 'energy': row.energy}
    
        data.append(molecule)

    # filter out bad data if needed
    if filterSigma > 0:
        data = filter_by_sigma(data, filterSigma)

    print('total number of data: %d' % (len(data)))
    
    return data

# check if need to filter out bad data points out of number sigma 
def filter_by_sigma(db, nSigma):
    # save starting iterator
    # calculate mean and sigma of atomization_energy
    energyArr = []
    for row in db:
        energyArr.append(row["energy"])
    mean = np.mean(energyArr)
    sigma = np.std(energyArr)
    # cut off at nSigma
    leftCutOff  = mean - nSigma*sigma
    rightCutOff = mean + nSigma*sigma
    # filter and create new db
    newDb = []
    for row in db:
        en = row["energy"]
        if ( en >= leftCutOff and en <= rightCutOff):
            newDb.append(row)
        else:
            print('filter out row with engergy %f with sigma %f' % (en, nSigma))
    return newDb
    
