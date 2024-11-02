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

    # in preprocess step we transform matrix h into upper triangular form
    # note that after preprocessing, matrix is no longer sparse
    def preprocess(self):
        self._gauss_jordan_elimination()
        # If initial matrix does not have full row rank, in order to bring it into upper triangular form,
        # linear dependant rows should be removed. Consequently, code parameters may change.
        self.create_new_h_alist()

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

    def _remove_row(self, row_index):
        warnings.warn("Removing linearly dependant row!")
        self.h = np.delete(self.h, row_index, axis=0)
        self.m -= 1
        self.k += 1

    def _swap_columns(self, column_to_swap_index, pivot_position):
        was_column_swapped = True
        for column_index, column in enumerate(self.h.T[column_to_swap_index:]):
            for element in column[pivot_position:]:
                if element == 1:
                    self.h[:, [column_index, column_to_swap_index]] = self.h[:, [column_to_swap_index, column_index]]
                    return was_column_swapped
        return not was_column_swapped

    def _gauss_jordan_elimination(self):
        # initialize pivot
        i = 0
        j = 0
        while i < self.m and j < self.m:
            # if pivot is 0 we need to swap rows (or columns)
            if self.h[i][j] == 0:
                non_zero_element_below = self._find_non_zero_element_below(i, j)
                if non_zero_element_below is None:
                    # swap columns
                    was_column_swap_possible = self._swap_columns(j, i)
                    if not was_column_swap_possible:
                        self._remove_row(i)
                        continue
                else:
                    # swap rows
                    self.h[[i, non_zero_element_below]] = self.h[[non_zero_element_below, i]]

            # all elements below pivot should be 0
            for ri in range(i+1, self.m):
                if self.h[ri][j] == 1:
                    self.h[ri] = (self.h[ri] + self.h[i]) % 2
            i += 1
            j += 1

    def create_new_h_alist(self):
        pass
