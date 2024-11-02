import matplotlib.pyplot as plt
import numpy as np

import channels.awgn_bpsk
from bs_encoder import BsEncoder
from ms_decoder import decode

from utils.helper_functions import get_h_alist


if __name__ == "__main__":
    h_alist = get_h_alist('generated_data/test2.alist')
    h_txt = 'generated_data/test2_h.txt'
    h = np.loadtxt(h_txt, dtype=int)
    n = np.shape(h)[1]
    eb_n0_range = [i/4 for i in range(25)]
    messages = 100
    result_ber = dict()
    result_fer = dict()

    encoder = BsEncoder(h, h_alist)
    h, h_alist, k = encoder.preprocess()

    for eb_n0 in eb_n0_range:
        ber = 0
        fer = 0

        for _ in range(messages):
            message = np.random.randint(low=0, high=2, size=k)
            encoded_message = encoder.encode(message)
            received_message = channels.awgn_bpsk.transmit(encoded_message, eb_n0)
            # todo use correct h to decode!
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
