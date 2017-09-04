import numpy as np

def pad_array(x, shape, fill=0, both=False):
  """
  Pad an array with a fill value.
  Parameters
  ----------
  x : ndarray
      Matrix.
  shape : tuple or int
      Desired shape. If int, all dimensions are padded to that size.
  fill : object, optional (default 0)
      Fill value.
  both : bool, optional (default False)
      If True, split the padding on both sides of each axis. If False,
      padding is applied to the end of each axis.
  """
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

def coulomb_matrix(atoms):
    numAtoms = len(atoms)
    matrix = np.zeros((numAtoms, numAtoms))
    
    for i in range(numAtoms):
        for j in range(numAtoms):
            if i == j:
                matrix[i, j] = 0.5 * z[i]**2.4
            elif i < j:
                matrix[i, j] = (z[i] * z[j]) / d[i, j]
                matrix[j, i] = m[i, j]
            else:
                continue

    rval = np.asarray(rval)
    return rval

