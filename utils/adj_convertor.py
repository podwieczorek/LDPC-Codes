#!/usr/bin/env python3

import argparse
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

import helper_functions


def load_adjacency_matrix(file_path):
    return np.loadtxt(file_path, delimiter=' ', dtype=int)


# creates parity check matrix from given adjacency matrix in the same directory
def create_h_txt(matrix, new_matrix_shape, file_path):
    n = new_matrix_shape[1]
    new_matrix = matrix[n:, :n]

    # remove all-zero rows and columns
    new_matrix = new_matrix[~np.all(new_matrix == 0, axis=1)]
    new_matrix = new_matrix[:, ~np.all(new_matrix == 0, axis=0)]
    print(new_matrix.shape)

    new_file_path = file_path[:-4] + '_h.txt'
    np.savetxt(new_file_path, new_matrix, delimiter=' ', fmt='%d')
    return new_matrix


def visualize_graph(adj_matrix):
    g = nx.from_numpy_array(adj_matrix)

    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(g)
    nx.draw(g, pos, with_labels=True, node_color='skyblue', node_size=700, edge_color='gray')
    plt.title('Graph Visualization')
    plt.show()


def get_h_matrix_shape(adj_matrix):
    # todo
    if adj_matrix.shape[0] == 672:
        n = 576
        m = 96
    elif adj_matrix.shape[0] == 192:
        n = 128
        m = 64
    else:
        raise ValueError(f'Shape: {adj_matrix.shape} of the adjacency matrix is not supported')

    return m, n


def main():
    parser = argparse.ArgumentParser(description='Script converts adjacency matrix of tanner graph to corresponding'
                                                 'parity check matrix in both .txt and .alist format')
    parser.add_argument('-p', '--path', help='.txt file to convert', required=True)

    file_path = vars(parser.parse_args())['path']
    adj_matrix = load_adjacency_matrix(file_path)
    shape = get_h_matrix_shape(adj_matrix)

    # show graph before preprocessing
    show_before_preprocessing = False
    if show_before_preprocessing:
        visualize_graph(adj_matrix)

    h_matrix = create_h_txt(adj_matrix, shape, file_path)
    helper_functions.create_h_alist(h_matrix, file_path)


if __name__ == "__main__":
    main()
