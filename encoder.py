"""
Basic (but not very efficient) encoding method for linear codes, performed in 3 steps:
1. Convert parity check matrix H into reduced row echelon form by Gauss Jordan elimination
2. Create generator matrix G=[I P^T] from  H=[P I]
3. Multiply message by generator matrix G to obtain codeword

Note:
    Not every parity check matrix can be converted into reduced row echelon form just by row operations.
    In that case column permutations are needed. Column permutation will result in different code.
    Resulting code will be different, but (by some definitions) equivalent or "permutation equivalent"
Note 2:
    This encoding method involves row addition and may not preserve the matrix sparseness
"""


import warnings
import numpy as np


# after column swaps, the code changes. We need to keep track of the swaps, in order to properly decode messages
# todo refactor to get rid of the global variable
column_swaps = []


def find_non_zero_element_below(h, pivot_position, column_index):
    for element_index, element in enumerate(h.T[column_index][pivot_position:]):
        if element == 1:
            return pivot_position + element_index
    return None


def swap_columns(h, column_to_swap_index, pivot_position):
    for column_index, column in enumerate(h.T):
        for element in column[pivot_position:]:
            if element == 1:
                # column permutations change one code to different one (!)
                # but this two codes are 'equivalent' by some definitions
                warnings.warn("Column permutation, code changes!")
                h[:, [column_index, column_to_swap_index]] = h[:, [column_to_swap_index, column_index]]
                column_swaps.append([column_index, column_to_swap_index])
                return


def gauss_jordan_elimination(h, k):
    m, n = np.shape(h)
    # initialize pivot
    i = 0
    j = k
    while i < m and j < n:
        # if pivot is 0 and there are only 0s beneath it, we have to swap columns
        if h[i][j] == 0 and find_non_zero_element_below(h, i, j) is None:
            swap_columns(h, j, i)

        # if pivot is zero we need to swap the rows
        if h[i][j] == 0:
            i1 = find_non_zero_element_below(h, i, j)
            if i1 is None:
                return
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
    p = h[:, :k]
    return np.concatenate((g, np.transpose(p)), axis=1)


def swap_columns_alist(h_alist, k):
    alist_offset = 4
    for swap in column_swaps:
        index1, index2 = swap

        # step 1: "swapping" columns in variable nodes
        temp = h_alist[index1 + alist_offset]
        h_alist[index1 + alist_offset] = h_alist[index2 + alist_offset]
        h_alist[index2 + alist_offset] = temp

        # step 2: "swapping" columns in check nodes
        index1 += 1  # alist format uses 1-based indexing
        index2 += 1
        for check_nodes_indices in h_alist[alist_offset+k:]:
            for i in range(len(check_nodes_indices)):
                if check_nodes_indices[i] == index1:
                    check_nodes_indices[i] = index2
                elif check_nodes_indices[i] == index2:
                    check_nodes_indices[i] = index1


def encode(h, h_alist, k, message):
    column_swaps.clear()
    gauss_jordan_elimination(h, k)
    g = create_generator_matrix(h, k)
    swap_columns_alist(h_alist, k)
    codeword = (message @ g) % 2
    return codeword
