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