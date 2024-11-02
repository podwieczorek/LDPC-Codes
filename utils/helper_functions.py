import warnings
import os

import numpy as np


def get_h_alist(file_path):
    if not os.path.isfile(file_path):
        warnings.warn('File does not exist, check file path')

    with open(file_path, 'r') as file:
        h_alist_int = []
        for line in file:
            row = [int(i) for i in line.split()]
            h_alist_int.append(row)

        return h_alist_int


def swap_columns_h_alist(h_alist, swaps):
    alist_offset = 4
    n = h_alist[0][0]
    for swap in swaps:
        index1, index2 = swap

        # step 1: "swapping" columns in variable nodes
        temp = h_alist[index1 + alist_offset]
        h_alist[index1 + alist_offset] = h_alist[index2 + alist_offset]
        h_alist[index2 + alist_offset] = temp

        # step 2: "swapping" columns in check nodes
        index1 += 1  # alist format uses 1-based indexing
        index2 += 1
        for check_nodes_indices in h_alist[alist_offset + n:]:
            for i in range(len(check_nodes_indices)):
                if check_nodes_indices[i] == index1:
                    check_nodes_indices[i] = index2
                elif check_nodes_indices[i] == index2:
                    check_nodes_indices[i] = index1


def remove_rows(a, y):
    return [row for i, row in enumerate(a) if i not in y]


# todo check if it does what its supposed to do
def remove_columns_h_alist(h_alist, h, removes):
    alist_offset = 4

    # remove columns from h
    np.delete(h, removes, axis=1)

    # change n value in h alist
    new_n_value = np.shape(h)[1]
    h_alist[0][0] = new_n_value

    # remove columns in variable nodes
    removes += alist_offset
    h_alist = remove_rows(h_alist, removes)
    removes -= alist_offset

    # remove columns in check nodes
    for row_index in range(alist_offset + new_n_value):
        # todo remove
        pass


# todo test
def create_h_alist(h_matrix, file_path):
    new_file_path = file_path[:-4] + '.alist'
    with open(new_file_path, 'w') as new_file:

        # number of variable nodes and number of parity check nodes
        new_file.write(f'{h_matrix.shape[1]} {h_matrix.shape[0]}\n')
        # next three lines are not used in encoder, so they are filled with dummy values
        new_file.write(f'0\n0\n0\n')

        # 1s indices for each column
        for column in h_matrix.T:
            for i, val in enumerate(column):
                if val == 1:
                    new_file.write(f'{i + 1} ')
            new_file.write('\n')

        # 1s indices for each row
        for row in h_matrix:
            for i, val in enumerate(row):
                if val == 1:
                    new_file.write(f'{i + 1} ')
            new_file.write('\n')