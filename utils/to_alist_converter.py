#!/usr/bin/env python3

import argparse
import os
import glob
import numpy as np

import helper_functions


def main():
    parser = argparse.ArgumentParser(description='Script converts parity check matrices in .txt format to'
                                                 'corresponding matrix in .alist format')
    parser.add_argument('-p', '--path', help='directory with .txt files to convert', required=True)

    directory = vars(parser.parse_args())['path']
    for file in glob.glob(os.path.join(directory, '*.txt')):
        h_matrix = np.loadtxt(file, delimiter=' ', dtype=int)
        helper_functions.create_h_alist(h_matrix, file)


if __name__ == "__main__":
    main()
