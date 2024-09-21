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

import utils.helper_functions


class Encoder:
    def __init__(self, h, h_alist):
        self.h = h
        self.h_alist = h_alist
        self.m, self.n = np.shape(h)
        self.k = self.n - self.m
        self.column_swaps = []
        self.g = None

    def preprocess(self):
        self.column_swaps.clear()
        self._gauss_jordan_elimination()
        self.g = self._create_generator_matrix()
        utils.helper_functions.swap_columns_h_alist(self.h_alist, self.column_swaps)

    def encode(self, message):
        codeword = (message @ self.g) % 2
        return codeword

    def _find_non_zero_element_below(self, pivot_position, column_index):
        for element_index, element in enumerate(self.h.T[column_index][pivot_position:]):
            if element == 1:
                return pivot_position + element_index
        return None

    def _swap_columns(self, column_to_swap_index, pivot_position):
        for column_index, column in enumerate(self.h.T):
            for element in column[pivot_position:]:
                if element == 1:
                    # column permutations change one code to different one (!)
                    # but this two codes are 'equivalent' by some definitions
                    warnings.warn("Column permutation, code changes!")
                    self.h[:, [column_index, column_to_swap_index]] = self.h[:, [column_to_swap_index, column_index]]
                    self.column_swaps.append([column_index, column_to_swap_index])
                    return

    def _gauss_jordan_elimination(self):
        # initialize pivot
        i = 0
        j = self.k
        while i < self.m and j < self.n:
            # if pivot is 0 and there are only 0s beneath it, we have to swap columns
            if self.h[i][j] == 0 and self._find_non_zero_element_below(i, j) is None:
                self._swap_columns(j, i)

            # if pivot is zero we need to swap the rows
            if self.h[i][j] == 0:
                i1 = self._find_non_zero_element_below(i, j)
                if i1 is None:
                    return
                self.h[[i, i1]] = self.h[[i1, i]]

            # all elements above and below pivot should be 0
            for ri in range(self.m):
                if ri != i and self.h[ri][j] == 1:
                    self.h[ri] = (self.h[ri] + self.h[i]) % 2
            i += 1
            j += 1

    def _create_generator_matrix(self):
        g = np.identity(self.k, dtype=int)
        p = self.h[:, :self.k]
        return np.concatenate((g, np.transpose(p)), axis=1)
