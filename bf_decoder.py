"""
Gallager's bit-flipping hard-decision decoder
'particularly simple but is applicable only to the BSC at rates far below channel capacity'
Algorithm described in 'Low-Density Parity-Check Codes', R. G. Gallager, 1962
"""

import numpy as np


#  for every message bit calculate in how many failed parity checks the message bit was involved
def calculate_failed_parity_checks(h_alist, n, s):
    f = np.zeros(n, dtype=int)
    alist_offset = 4
    for i in range(n):
        for j in h_alist[i + alist_offset]:
            if j == 0:  # values are right padded with zeros
                break
            else:
                if s[j - 1] == 1:
                    f[i] += 1
    return f


# flipping threshold is half of the parity checks:
# if more than half bit's parity checks fail, the bit should be flipped
def calculate_flipping_thresholds(h):
    flipping_thresholds = h.sum(axis=0)
    flipping_thresholds //= 2
    return flipping_thresholds


def decode(h, h_alist, received_msg):
    m, n = np.shape(h)

    # max_num_of_iteration and flipping_thresholds may be different for every code
    max_num_of_iteration = 1000
    flipping_thresholds = calculate_flipping_thresholds(h)

    iteration = 0
    while iteration < max_num_of_iteration:
        # calculate syndrome, if it is a zero vector, the message was decoded successfully
        s = (received_msg @ np.transpose(h)) % 2
        if not s.any():
            break

        # compute parity checks and flip the bits
        f = calculate_failed_parity_checks(h_alist, n, s)
        for i, fi in enumerate(f):
            if fi > flipping_thresholds[i]:
                received_msg[i] = 1 - received_msg[i]  # flip bit

        iteration += 1

    return received_msg
