import argparse
import os
import warnings
import numpy as np


def save_matrices_to_txt_files(created_matrices, dir_name):
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)

    for i, matrix in enumerate(created_matrices):
        file_name = 'gallager_matrix_' + str(i)
        file_path = os.path.join(dir_name, file_name)
        try:
            np.savetxt(file_path, matrix, fmt='%d', delimiter=' ')
        except Exception as e:
            print(f'{e} could not save file: {file_path}!')


def create_base_submatrix(n, num_of_rows_submatrix):
    base_submatrix = np.zeros((num_of_rows_submatrix, n), dtype=int)
    for i in range(n):
        base_submatrix[i % num_of_rows_submatrix][i] = 1
    return base_submatrix


def create_submatrix(base_submatrix):
    submatrix = np.random.permutation(base_submatrix.T).T
    return submatrix


def create_h_matrix(n, c, num_of_rows_submatrix):
    base_submatrix = create_base_submatrix(n, num_of_rows_submatrix)
    out_matrix = base_submatrix
    for _ in range(c-1):
        submatrix = create_submatrix(base_submatrix)
        out_matrix = np.concatenate((out_matrix, submatrix), axis=0)
    return out_matrix


def main():
    parser = argparse.ArgumentParser(description='Program generates parity check matrices. Method proposed by '
                                                 'Robert Gallager ')
    parser.add_argument('-m', help='Number of parity check bits (number of rows in h)', required=True)
    parser.add_argument('-n', help='Number of variable bits (number of columns in h)', required=True)
    parser.add_argument('-c', help='Number of submatrices, note that m/c should be an integer', required=True)
    parser.add_argument('-x', help='How many h matrices will be created, default 1', required=False, default=1)
    parser.add_argument('-d', help='New directory name, default: generated_gallager', required=False,
                        default='generated_gallager')

    m = int(vars(parser.parse_args())['m'])
    n = int(vars(parser.parse_args())['n'])
    c = int(vars(parser.parse_args())['c'])
    h = int(vars(parser.parse_args())['x'])
    dir_name = vars(parser.parse_args())['d']

    num_of_rows_submatrix = m/c
    if not num_of_rows_submatrix.is_integer():
        warnings.warn('m/c should be a whole number!')
        return
    num_of_rows_submatrix = int(num_of_rows_submatrix)

    created_matrices = []
    for _ in range(h):
        matrix = create_h_matrix(n, c, num_of_rows_submatrix)
        print(matrix)
        created_matrices.append(matrix)

    save_matrices_to_txt_files(created_matrices, dir_name)


if __name__ == "__main__":
    main()
