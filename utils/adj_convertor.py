import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

import utils.helper_functions

def load_adjacency_matrix(file_path):
    return np.loadtxt(file_path, delimiter=' ', dtype=int)


def make_graph_bipartite(matrix, n):
    print(f'Number of non-zero elements in generated matrix: {np.count_nonzero(matrix[:n, :n])} +', end=' ')
    matrix[:n, :n] = 0
    print(f'{np.count_nonzero(matrix[n:, n:])} ')
    matrix[n:, n:] = 0

    matrix = matrix + matrix.T
    matrix[matrix > 1] = 1


# todo remove empty rows and columns etc
def create_h_txt(matrix, new_matrix_shape, file_path):
    n = new_matrix_shape[1]
    new_matrix = matrix[n:, :n]
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


if __name__ == "__main__":
    file_path = '../generated_data/test2.txt'
    adj_matrix = load_adjacency_matrix(file_path)

    # todo use node attributes to differentiate between bit nodes and parity check nodes
    n = 128
    m = 64
    shape = (m, n)

    # show graph before preprocessing
    # visualize_graph(adj_matrix)

    do_preprocess = True
    if do_preprocess:
        make_graph_bipartite(adj_matrix, n)

    h_matrix = create_h_txt(adj_matrix, shape, file_path)
    utils.helper_functions.create_h_alist(h_matrix, file_path)
