"""
Back substitution encoder, "probably the most straightforward way of constructing an encoder"
Before encoding, we need to bring H into upper triangular form by Gauss-Jordan elimination.
Note that this procedure involves row addition and may not preserve the matrix sparseness

Algorithm described in "Efficient Encoding of Low-Density Parity-Check Codes"
Thomas J. Richardson and Rüdiger L. Urbanke
"""


import numpy as np
import warnings


class BsEncoder:
    def __init__(self, h, h_alist):
        self.h = h
        self.h_alist = h_alist
        self.m, self.n = np.shape(h)
        self.k = self.n - self.m
        self.swaps = []

    # in preprocess step we transform matrix h into upper triangular form
    # note that after preprocessing, matrix is no longer sparse
    def preprocess(self):
        self._gauss_jordan_elimination()
        self._handle_column_swaps()  # "swap" columns in h_alist

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

    def _swap_columns(self, column_to_swap_index, pivot_position):
        for column_index, column in enumerate(self.h.T):
            for element in column[pivot_position:]:
                if element == 1:
                    # note: column permutations change one code to different one
                    # but this two codes are 'equivalent' by some definitions
                    warnings.warn("Column permutation, code changes!")
                    self.h[:, [column_index, column_to_swap_index]] = self.h[:, [column_to_swap_index, column_index]]
                    self.swaps.append([column_index, column_to_swap_index])
                    return

    def _gauss_jordan_elimination(self):
        # initialize pivot
        i = 0
        j = 0
        while i < self.m and j < self.m:
            # if pivot is 0 and there are only 0s beneath it, we have to swap columns
            if self.h[i][j] == 0 and self._find_non_zero_element_below(i, j) is None:
                self._swap_columns(j, i)

            # if pivot is zero we need to swap the rows
            if self.h[i][j] == 0:
                i1 = self._find_non_zero_element_below(i, j)
                if i1 is None:
                    warnings.warn('Cannot bring matrix into upper triangular form!')
                self.h[[i, i1]] = self.h[[i1, i]]

            # all elements above and below pivot should be 0
            for ri in range(i+1, self.m):
                if self.h[ri][j] == 1:
                    self.h[ri] = (self.h[ri] + self.h[i]) % 2
            i += 1
            j += 1

    # in order to decode properly, we also need to "swap" columns in h_alist
    def _handle_column_swaps(self):
        alist_offset = 4
        for swap in self.swaps:
            index1, index2 = swap

            # step 1: "swapping" columns in variable nodes
            temp = self.h_alist[index1 + alist_offset]
            self.h_alist[index1 + alist_offset] = self.h_alist[index2 + alist_offset]
            self.h_alist[index2 + alist_offset] = temp

            # step 2: "swapping" columns in check nodes
            index1 += 1  # alist format uses 1-based indexing
            index2 += 1
            for check_nodes_indices in self.h_alist[alist_offset + self.n:]:
                for i in range(len(check_nodes_indices)):
                    if check_nodes_indices[i] == index1:
                        check_nodes_indices[i] = index2
                    elif check_nodes_indices[i] == index2:
                        check_nodes_indices[i] = index1
