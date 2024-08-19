import numpy as np


class RuEncoder:
    def __init__(self, h, h_alist):
        self.h = h
        self.h_alist = h_alist
        self.m, self.n = np.shape(h)
        self.k = self.n = self.m
        self.phi = None
        self.g = None
        self.swaps = None

    def preprocess(self):
        # function transforms h into approximate upper triangular form and returns gap g
        self.g = self._approximate_upper_triangulation()
        # if gap is 0, calculating phi is not needed
        if self.g != 0:
            self.phi = self._calculate_phi()

    def _approximate_upper_triangulation(self):
        t = 0
        g = 0
        while True:
            if t == self.m - g:
                return g
            # find minimum residual degree and columns with that degree
            min_res_degree, columns = self._minimum_residual_degree(t)
            chosen_column_index = np.random.randint(0, columns.size, dtype=int)
            random_column = columns[chosen_column_index]
            if min_res_degree == 1:
                self._extend(random_column, t)
            else:
                g += self._choose(random_column, t)
            t += 1

    # todo, extend() and choose() are similar
    def _extend(self, column_to_swap, t):
        # swap columns
        self.h[:, [t, column_to_swap]] = self.h[:, [column_to_swap, t]]
        # todo add to swaps
        # find row with 1 in residual parity check matrix
        sub_array = self.h[t:self.m - self.g, t]
        row_to_swap = np.where(sub_array == 1)[0][0] + t
        # swap rows
        self.h[[t, row_to_swap], :] = self.h[[row_to_swap, t], :]

    def _choose(self, column_to_swap, t):
        # swap columns
        self.h[:, [t, column_to_swap]] = self.h[:, [column_to_swap, t]]
        # todo add to swaps
        # find rows with 1 in residual parity check matrix
        sub_array = self.h[t:self.m - self.g, t]
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

    def _minimum_residual_degree(self, t):
        residual_h = self.h[t:self.m - self.g, t:self.n]
        column_sums = np.sum(residual_h, axis=0)
        column_sums[column_sums == 0] = np.iinfo(np.int32).max
        min_nonzero_weight = np.min(column_sums)
        columns_with_min_weight = np.where(column_sums == min_nonzero_weight)[0] + t
        return min_nonzero_weight, columns_with_min_weight

    # todo check if phi is singular
    def _calculate_phi(self):
        a = self.h[:self.m - self.g, self.m - self.g:self.m]
        c = self.h[self.m - self.g:, self.m - self.g:self.m]
        e = self.h[self.m - self.g:, :self.m - self.g]
        t_inv = np.linalg.inv(self.h[:self.m - self.g, :self.m - self.g])  # todo in gf2
        eta = e @ (t_inv @ a)
        return c - eta

    # todo check if gap is 0
    def encode(self, message):
        p2 = self._calculate_p2(message)
        p1 = self._calculate_p1(message, p2)
        # todo swap columns h and h_alist
        return np.concatenate((p1, p2, message), axis=None)

    def _calculate_p1(self, s, p2):
        a = self.h[:self.m - self.g, self.m - self.g:self.m]
        b = self.h[:self.m - self.g, self.m:]
        t = self.h[:self.m - self.g, :self.m - self.g]
        t_inv = np.linalg.inv(t)
        ap2t = a @ np.transpose(p2)
        bst = b @ np.transpose(s)

        return - t_inv @ (ap2t + bst)

    def _calculate_p2(self, s):
        d = self.h[self.m - self.g:, self.m:]
        e = self.h[self.m - self.g:, :self.m - self.g]
        b = self.h[:self.m - self.g, self.m:]
        t = self.h[:self.m - self.g, :self.m - self.g]
        phi_inv = np.linalg.inv(self.phi)  # todo in gf2
        t_inv = np.linalg.inv(t)  # todo in gf2
        dst = d @ np.transpose(s)
        etbst = (e @ (t_inv @ (b @ np.transpose(s))))
        return - phi_inv @ (dst - etbst)
