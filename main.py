import numpy as np
import warnings
import os

from encoder import encode
from bf_decoder import decode


def get_h_alist(file_path):
    if not os.path.isfile(file_path):
        warnings.warn("File does not exist, check file path")

    with open(file_path, 'r') as file:
        content = file.readlines()
        return content


if __name__ == "__main__":
    h_alist = get_h_alist('h_matrices/BCH_7_4_1_strip.alist')
    h_txt = 'h_matrices/BCH_7_4_1_strip.txt'
    h = np.loadtxt(h_txt, dtype=int)
    n = np.shape(h)[1]
    k = n - np.shape(h)[0]
    messages = 1

    for _ in range(messages):
        # message = np.random.randint(low=0, high=2, size=k)
        message = [0, 1, 0, 1]

        encoded_message = encode(h, k, message)
        print(encoded_message)
        # signal = awgn_bpsk_channel(encoded_message, K, N, eb_n0)
        decoded_message = decode(h, h_alist, encoded_message)
        print(decoded_message)
