import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def load_adjacency_matrix(file_path):
    return np.loadtxt(file_path, delimiter=' ', dtype=int)


def make_graph_bipartite(matrix, n):
    matrix[:n, :n] = 0
    matrix[n:, n:] = 0

    matrix = matrix + matrix.T
    matrix[matrix > 1] = 1


# todo remove empty rows and columns etc
def create_h_txt(matrix, new_matrix_shape, file_path):
    n = new_matrix_shape[1]
    new_matrix = matrix[n:, :n]
    new_file_path = file_path[:-4] + '_h.txt'
    np.savetxt(new_file_path, new_matrix, delimiter=' ', fmt='%d')
    return new_matrix


# todo test
def create_h_alist(h_matrix, file_path):
    new_file_path = file_path[:-4] + '.alist'
    with open(new_file_path, 'w') as new_file:
        
        # number of variable nodes and number of parity check nodes
        new_file.write(f'{h_matrix.shape[1]} {h_matrix.shape[0]}\n')
        # next three lines are not used in encoder, so there alre filled with dummy values
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


def visualize_graph(adj_matrix):
    g = nx.from_numpy_array(adj_matrix)

    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(g)
    nx.draw(g, pos, with_labels=True, node_color='skyblue', node_size=700, edge_color='gray')
    plt.title('Graph Visualization')
    plt.show()


if __name__ == "__main__":
    file_path = '../generated_data/test1.txt'
    adj_matrix = load_adjacency_matrix(file_path)

    # todo use node attributes to differentiate between bit nodes and parity check nodes
    n = 128
    m = 64
    shape = (m, n)

    make_bipartite = False
    if make_bipartite:
        make_graph_bipartite(adj_matrix, n)

    h_matrix = create_h_txt(adj_matrix, shape, file_path)
    create_h_alist(h_matrix, file_path)
    # visualize_graph(adj_matrix)
