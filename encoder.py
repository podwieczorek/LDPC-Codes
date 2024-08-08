import warnings
import numpy as np


def gauss_jordan_elimination(h, k):
    m, n = np.shape(h)
    # initialize pivot
    i = 0
    j = k
    while i < m and j < n:
        # if pivot is 0, we search for 1 below and swap the rows
        if h[i][j] == 0:
            # find non-zero element below
            i1 = i
            while h[i1][j] == 0 and i1 < m:
                i1 += 1
            if h[i1][j] == 0:
                warnings.warn("Invalid format, returning")
                return
            # swap the rows
            h[[i, i1]] = h[[i1, i]]
        # all elements above and below pivot should be 0
        for ri in range(m):
            if ri == i:
                continue
            if h[ri][j] == 1:
                h[ri] = (h[ri] + h[i]) % 2
        i += 1
        j += 1


def create_generator_matrix(h, k):
    g = np.identity(k, dtype=int)
    p = h[:, 0:k]
    return np.concatenate((g, np.transpose(p)), axis=1)


def encode(h, k, message):
    gauss_jordan_elimination(h, k)
    print(h)
    g = create_generator_matrix(h, k)
    codeword = (np.transpose(message) @ g) % 2
    return codeword
