"""
Back substitution encoder, "probably the most straightforward way of constructing an encoder"
Before encoding, we need to bring H into upper triangular form by Gauss-Jordan elimination.
Note that this procedure involves row addition and may not preserve the matrix sparseness

Algorithm described in "Efficient Encoding of Low-Density Parity-Check Codes"
Thomas J. Richardson and Rüdiger L. Urbanke
"""


import numpy as np
import warnings

import utils.helper_functions


class BsEncoder:
    def __init__(self, h, h_alist):
        self.h = h
        self.h_alist = h_alist
        self.m, self.n = np.shape(h)
        self.k = self.n - self.m
        self.swaps = []
        self.removes = []

    # in preprocess step we transform matrix h into upper triangular form
    # note that after preprocessing, matrix is no longer sparse
    def preprocess(self):
        self._gauss_jordan_elimination()
        if len(self.swaps) > 0:
            utils.helper_functions.swap_columns_h_alist(self.h_alist, self.swaps)
        if len(self.removes) > 0:
            utils.helper_functions.remove_columns_h_alist(self.h_alist, self.h, self.removes)
            print(f'To bring H matrix into upper triangular form, {len(self.removes)} column(s) was/were removed!')

    def encode(self, message):
        # back substitution
        p = np.zeros(self.m)
        for i in range(self.m-1, -1, -1):
            for j in range(i, self.m-1):
                p[i] += self.h[i][j] * p[j]
            for j in range(self.k):
                p[i] += self.h[i][j+self.m] * message[j]
        p = p % 2
        return np.concatenate((p, message), axis=None)

    def _find_non_zero_element_below(self, pivot_position, column_index):
        for element_index, element in enumerate(self.h.T[column_index][pivot_position:]):
            if element == 1:
                return pivot_position + element_index
        return None

    def _too_many_columns_removed(self):
        num_of_removed_columns = len(self.removes)
        return num_of_removed_columns >= self.k

    # todo check - looks suspicious
    def _swap_columns(self, column_to_swap_index, pivot_position):
        for column_index, column in enumerate(self.h.T):
            for element in column[pivot_position:]:
                if element == 1:
                    # note: column permutations change one code to different one
                    # but this two codes are 'equivalent' by some definitions
                    warnings.warn("Column permutation, code changes!")
                    self.h[:, [column_index, column_to_swap_index]] = self.h[:, [column_to_swap_index, column_index]]
                    self.swaps.append([column_index, column_to_swap_index])
                    return True
        return False

    def _gauss_jordan_elimination(self):
        # initialize pivot
        i = 0
        j = 0
        while i < self.m and j < self.m:
            if self._too_many_columns_removed():
                raise ValueError('Too many columns were removed, n should be greater than m!')

            # if pivot is 0 we need to swap rows (or columns)
            if self.h[i][j] == 0:
                non_zero_element_below = self._find_non_zero_element_below(i, j)
                if non_zero_element_below is None:
                    # swap columns
                    was_column_swap_possible = self._swap_columns(j, i)
                    # check if column swap was successful, if not, we need to remove the column
                    if not was_column_swap_possible:
                        self.removes.append(j)
                        j += 1
                        continue
                else:
                    # swap rows
                    self.h[[i, non_zero_element_below]] = self.h[[non_zero_element_below, i]]

            # all elements above and below pivot should be 0
            for ri in range(i+1, self.m):
                if self.h[ri][j] == 1:
                    self.h[ri] = (self.h[ri] + self.h[i]) % 2
            i += 1
            j += 1
