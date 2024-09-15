import glob
import os

import numpy as np
from helper_functions import get_h_alist


def save_to_npz(graphs):
    path = '../data/data_npz/wimax_data.npz'
    np.savez(path, graphs)


# every node has only one attribute:
# 1: if the node is a parity check node
# 0: in other cases
def get_attributes_data(h_alist):
    n = h_alist[0][0]
    m = h_alist[0][1]

    variable_node_attributes = np.zeros(n, dtype=int)
    parity_check_node_attributes = np.ones(m, dtype=int)

    attributes_data = np.concatenate((variable_node_attributes, parity_check_node_attributes), axis=None)
    return attributes_data


# get indices (for rows and for columns) from h matrix stored in .alist format
# remove zero-padding and change to zero-based indexing
def get_indices(h_alist):
    alist_offset = 4
    n = h_alist[0][0]

    variable_indices = np.array(h_alist[alist_offset:alist_offset+n])
    parity_check_indices = np.array(h_alist[alist_offset+n:])

    # delete zero padding
    variable_indices = [list(row[row != 0]) for row in variable_indices]
    parity_check_indices = [list(row[row != 0]) for row in parity_check_indices]

    # change to zero-base indexing
    variable_indices = [[x - 1 for x in row] for row in variable_indices]
    parity_check_indices = [[x - 1 for x in row] for row in parity_check_indices]

    return parity_check_indices, variable_indices, n


# get adjacency matrix corresponding to tanner graph of h matrix
# adjacency matrix is stored in format used in scipy.sparse.csr_matrix
# note that "data" is not stored as it would contain only 1's
def get_sparse_indices(parity_check_indices, variable_indices, n):
    indices = []
    indices_pointers = []

    for row in variable_indices:
        indices_pointers.append(len(indices))
        for index in row:
            indices.append(n+index)

    for row in parity_check_indices:
        indices_pointers.append(len(indices))
        for index in row:
            indices.append(index)

    indices_pointers.append(len(indices))

    return indices, indices_pointers


def h_to_sparse(h_alist):
    parity_check_indices, variable_indices, n = get_indices(h_alist)
    indices, indices_pointers = get_sparse_indices(parity_check_indices, variable_indices, n)
    return indices, indices_pointers


def main():
    data = dict()
    path = '../wimax_alist'
    for alist_file in glob.glob(os.path.join(path, '*.alist')):
        h_alist = get_h_alist(alist_file)
        indices, indices_pointers = h_to_sparse(h_alist)
        attributes_data = get_attributes_data(h_alist)

        graph = {'adj_indices': indices,
                 'adj_indptr': indices_pointers,
                 'attr_data': attributes_data}
        graph_name = alist_file.split('/')[-1][:-6]

        data[graph_name] = graph

    save_to_npz(data)


if __name__ == "__main__":
    main()
