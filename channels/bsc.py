import numpy as np


def transmit(data, ber):
    # add random error pattern to signal
    errors = np.random.rand(len(data)) < ber
    received_data = np.logical_xor(data, errors).astype(int)

    return received_data
