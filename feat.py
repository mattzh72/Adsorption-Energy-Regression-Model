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
    numAtoms = len(molecule["atoms"])
    matrix = np.zeros((numAtoms, numAtoms))
    
    for i in range(numAtoms):
        for j in range(numAtoms):
            atomNumI = molecule["atoms"][i]["num"]
            atomNumJ = molecule["atoms"][j]["num"]
            if i == j:
                matrix[i, j] = 0.5 * atomNumI**2.4
            elif i < j:
                d = calculate_atom_distance(molecule["atoms"][i]["position"], molecule["atoms"][j]["position"])
                #  Convert AtomPositions from Angstrom to bohr (atomic units)
                d = d / (0.52917721092)
                matrix[i, j] = (atomNumI * atomNumJ) / d
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

def feat_coulombMatrix(molecule, max_atoms):
    m = coulomb_matrix(molecule)
    row_norms = np.asarray([np.linalg.norm(row) for row in m], dtype=float)
    p = np.argsort(row_norms)
    m = m[p][:, p]  # permute rows first, then columns
    m = pad_array(m, max_atoms)
    rval = m[np.triu_indices_from(m)]
    # flatten into one list
    rval = np.asarray(np.ravel(rval))
    return rval

def featurize(molecules, coulomb_eigen=True): 
    max_atoms = -1
    for mol in molecules:
        if (len(mol["atoms"]) > max_atoms):
            max_atoms = len(mol["atoms"])
    
    features = []
    for mol in molecules:
        if coulomb_eigen == True:
            f = calculate_eigenvalues(mol, max_atoms)
        else:
            f = feat_coulombMatrix(mol, max_atoms)
        features.append(f)
        
    return features

def extractTarget(molecules):
    energies = []
    
    for mol in molecules:
        e = mol["energy"]
        energies.append(e)
        
    return energies
        
        
