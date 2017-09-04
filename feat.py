import numpy as np
import math

def pad_array(x, shape, fill=0, both=False):
    x = np.asarray(x)
    if not isinstance(shape, tuple):
        shape = tuple(shape for _ in range(x.ndim))
    pad = []
    for i in range(x.ndim):
        diff = shape[i] - x.shape[i]
        assert diff >= 0
        if both:
            a, b = divmod(diff, 2)
            b += a
            pad.append((a, b))
        else:
            pad.append((0, diff))
    pad = tuple(pad)
    x = np.pad(x, pad, mode='constant', constant_values=fill)
    return x

def coulomb_matrix(molecule):
    numAtoms = len(molecule["formula"])
    matrix = np.zeros((numAtoms, numAtoms))
    
    for i in range(numAtoms):
        for j in range(numAtoms):
            if i == j:
                matrix[i, j] = 0.5 * molecule["formula"][i]**2.4
            elif i < j:
                matrix[i, j] = (molecule["formula"][i] * molecule["formula"][j]) / calculate_atom_distance(molecule["positions"][i], molecule["positions"][j])
                matrix[j, i] = matrix[i, j]
            else:
                continue

    return matrix

def calculate_atom_distance(p1, p2):
    return calculateDistance(p1, p2)
    
    
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

def calculate_eigenvalues(molecule, max_num_atoms):
    cmat = coulomb_matrix(molecule)
    w, v = np.linalg.eig(cmat)
    w_abs = np.abs(w)
    sortidx = np.argsort(w_abs)
    sortidx = sortidx[::-1]
    w = w[sortidx]
    f = pad_array(w, max_num_atoms)
    
    return f

def featurize(molecules):
    max_atoms = -1
    for i in range(len(molecules)):
        if (len(molecules[i]["formula"]) > max_atoms):
            max_atoms = len(molecules[i]["formula"])
    
    features = []
    for mol in molecules:
        f = calculate_eigenvalues(mol, max_atoms)
        features.append(f)
        
    return features

def extractTarget(molecules):
    energies = []
    
    for mol in molecules:
        e = mol["energy"]
        energies.append(e)
        
    return energies
        
        
