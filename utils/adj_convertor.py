import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def load_adjacency_matrix(file_path):
    return np.loadtxt(file_path, delimiter=' ')


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
    visualize_graph(adj_matrix)
