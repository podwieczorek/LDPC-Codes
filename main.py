import numpy as np
import warnings
import os

import channels.awgn_bpsk
from encoder import encode
from bp_decoder import decode


def get_h_alist(file_path):
    if not os.path.isfile(file_path):
        warnings.warn('File does not exist, check file path')

    with open(file_path, 'r') as file:
        h_alist_int = []
        for line in file:
            row = [int(i) for i in line.split()]
            h_alist_int.append(row)

        return h_alist_int


if __name__ == "__main__":
    h_alist = get_h_alist('h_matrices/BCH_7_4_1_strip.alist')
    h_txt = 'h_matrices/BCH_7_4_1_strip.txt'
    h = np.loadtxt(h_txt, dtype=int)
    n = np.shape(h)[1]
    k = n - np.shape(h)[0]
    messages = 1

    for _ in range(messages):
        message = np.random.randint(low=0, high=2, size=k)
        encoded_message = encode(h, k, message)
        received_message = channels.awgn_bpsk.transmit(encoded_message, 2)
        decoded_message = decode(h, h_alist, received_message)
