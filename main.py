import matplotlib.pyplot as plt
import numpy as np
import warnings
import os

import channels.awgn_bpsk
from ru_encoder import RuEncoder
from bs_encoder import BsEncoder
from encoder import Encoder
from ms_decoder import decode


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
    h_alist = get_h_alist('h_matrices/BCH_127_78_7_strip.alist')
    h_txt = 'h_matrices/BCH_127_78_7_strip.txt'
    h = np.loadtxt(h_txt, dtype=int)
    n = np.shape(h)[1]
    k = n - np.shape(h)[0]
    eb_n0_range = [i/4 for i in range(25)]
    messages = 10
    result_ber = dict()
    result_fer = dict()

    encoder = BsEncoder(h, h_alist)
    encoder.preprocess()

    for eb_n0 in eb_n0_range:
        ber = 0
        fer = 0

        for _ in range(messages):
            message = np.random.randint(low=0, high=2, size=k)
            encoded_message = encoder.encode(message)
            received_message = channels.awgn_bpsk.transmit(encoded_message, eb_n0)
            decoded_message = decode(h, h_alist, received_message)

            bit_errors = np.count_nonzero(encoded_message != decoded_message)
            frame_error = int(bit_errors > 0)
            ber += bit_errors
            fer += frame_error

        result_ber[eb_n0] = ber / (messages * k)
        result_fer[eb_n0] = fer / messages

        print(f'\t{eb_n0}\t|\t{result_ber[eb_n0]:.10f}\t|\t{result_fer[eb_n0]:.8f}')

    ber_values = np.array(list(result_ber.values()))
    fer_values = np.array(list(result_fer.values()))

    # plotting BER and FER vs Eb/N0 curves
    plt.plot(eb_n0_range, ber_values, 'b', label='BER')
    plt.plot(eb_n0_range, fer_values, 'r', label='FER')
    plt.title('BER and FER vs Eb/N0 in BPSK AWGN channel')
    plt.suptitle(f'messages={messages}, N={n}, K={k}')
    plt.xlabel('Eb/N0[dB]')
    plt.legend()
    plt.yscale('log')
    plt.show()
