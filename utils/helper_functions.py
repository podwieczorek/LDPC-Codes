import warnings
import os


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
