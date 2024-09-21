import warnings
import numpy as np

import utils.helper_functions


# todo use back substitution
# todo GJ elimination almost the same as in encoder, to refactor
def invert_matrix(matrix):
    n = matrix.shape[0]
    matrix = matrix % 2
    augmented_matrix = np.concatenate((matrix, np.identity(n, dtype=int)), axis=1)

    for i in range(n):
        if augmented_matrix[i, i] == 0:
            for j in range(i + 1, n):
                if augmented_matrix[j, i] == 1:
                    augmented_matrix[[i, j]] = augmented_matrix[[j, i]]
                    break
        if augmented_matrix[i, i] == 0:
            warnings.warn('Matrix is singular!')
            return None

        augmented_matrix[i] = augmented_matrix[i] % 2
        for j in range(n):
            if j != i and augmented_matrix[j, i] == 1:
                augmented_matrix[j] = (augmented_matrix[j] + augmented_matrix[i]) % 2
    return augmented_matrix[:, n:]


class RuEncoder:
    def __init__(self, h, h_alist):
        self.h = h
        self.h_alist = h_alist
        self.m, self.n = np.shape(h)
        self.k = self.n = self.m
        self.phi = None
        self.phi_inv = None
        self.g = None
        self.swaps = []
        self.t_inv = None

    def preprocess(self):
        # function transforms h into approximate upper triangular form and returns gap g
        self.g = self._approximate_upper_triangulation()
        self._invert_t()  # initialize t_inv
        utils.helper_functions.swap_columns_h_alist(self.h_alist, self.swaps)

        # if gap is 0, calculating phi is not needed
        if self.g != 0:
            self.phi = self._calculate_phi()
            self.phi_inv = invert_matrix(self.phi)
            if self.phi_inv is None:
                # todo handle by column permutations
                raise ValueError("Phi is singular!")

    def _invert_t(self):
        t = self.h[:self.m - self.g, :self.m - self.g]
        self.t_inv = invert_matrix(t)

    def _approximate_upper_triangulation(self):
        t = 0
        g = 0
        while True:
            if t == self.m - g:
                return g
            # find minimum residual degree and columns with that degree
            min_res_degree, columns = self._minimum_residual_degree(t, g)
            chosen_column_index = np.random.randint(0, columns.size, dtype=int)
            random_column = columns[chosen_column_index]
            if min_res_degree == 1:
                self._extend(random_column, t, g)
            else:
                g += self._choose(random_column, t, g)
            t += 1

    def _extend(self, column_to_swap, t, g):
        # swap columns
        self.h[:, [t, column_to_swap]] = self.h[:, [column_to_swap, t]]
        self.swaps.append((t, column_to_swap))
        # find row with 1 in residual parity check matrix
        sub_array = self.h[t:self.m - g, t]
        row_to_swap = np.where(sub_array == 1)[0][0] + t
        # swap rows
        self.h[[t, row_to_swap], :] = self.h[[row_to_swap, t], :]

    def _choose(self, column_to_swap, t, g):
        # swap columns
        self.h[:, [t, column_to_swap]] = self.h[:, [column_to_swap, t]]
        self.swaps.append((t, column_to_swap))
        # find rows with 1 in residual parity check matrix
        sub_array = self.h[t:self.m - g, t]
        row_indices = np.where(sub_array == 1)[0] + t
        # swap first row
        first_row_to_swap = int(row_indices[0])
        self.h[[t, first_row_to_swap], :] = self.h[[first_row_to_swap, t], :]

        # move other rows with 1 at the end of the h matrix
        rows_to_move = []
        for row_index in row_indices[-1:0:-1]:
            if rows_to_move == []:
                rows_to_move = self.h[row_index]
            else:
                rows_to_move = np.vstack((rows_to_move, self.h[row_index]))
            self.h = np.delete(self.h, row_index, axis=0)

        rows_to_move = np.reshape(rows_to_move, (row_indices.size - 1, self.n))
        np.concatenate((self.h, rows_to_move), axis=0)

        return row_indices.size - 1

    def _minimum_residual_degree(self, t, g):
        residual_h = self.h[t:self.m - g, t:self.n]
        column_sums = np.sum(residual_h, axis=0)
        column_sums[column_sums == 0] = np.iinfo(np.int32).max
        min_nonzero_weight = np.min(column_sums)
        columns_with_min_weight = np.where(column_sums == min_nonzero_weight)[0] + t
        return min_nonzero_weight, columns_with_min_weight

    def _calculate_phi(self):
        a = self.h[:self.m - self.g, self.m - self.g:self.m]
        c = self.h[self.m - self.g:, self.m - self.g:self.m]
        e = self.h[self.m - self.g:, :self.m - self.g]
        eta = (e @ (self.t_inv @ a) % 2) % 2
        return (c + eta) % 2

    def encode(self, message):
        if self.g == 0:
            p1 = self._calculate_p1(message, None)
            return np.concatenate((p1, message), axis=None)
        p2 = self._calculate_p2(message)
        p1 = self._calculate_p1(message, p2)
        return np.concatenate((p1, p2, message), axis=None)

    def _calculate_p1(self, s, p2):
        b = self.h[:self.m - self.g, self.m:]
        bst = b @ np.transpose(s) % 2

        if p2 is None:
            return (self.t_inv @ bst) % 2

        a = self.h[:self.m - self.g, self.m - self.g:self.m]
        ap2t = a @ np.transpose(p2) % 2
        return (self.t_inv @ ((ap2t + bst) % 2)) % 2

    def _calculate_p2(self, s):
        d = self.h[self.m - self.g:, self.m:]
        e = self.h[self.m - self.g:, :self.m - self.g]
        b = self.h[:self.m - self.g, self.m:]
        dst = d @ np.transpose(s) % 2
        etbst = (e @ (self.t_inv @ (b @ np.transpose(s))))
        return (self.phi_inv @ ((dst - etbst) % 2)) % 2
