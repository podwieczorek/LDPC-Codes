"""
Weighted bit-flipping decoder
It is an improved bit-flipping decoder that includes reliability measure of the received
symbol. For AWGN channel, the simple reliability measure is the magnitude of the symbol.

Algorithm described in:
'Low-Density Parity-Check Codes Based on Finite Geometries: A Rediscovery and New Results',
Y. Kou, S. Lin, M.P.C. Fossorier, 2001
"""

import numpy as np


# for every check bit find its message bit with the smallest magnitude (the weight)
def calculate_weights(h, h_alist, message):
    m, n = np.shape(h)
    alist_offset = 4
    weights = np.full(m, 10000.0)
    for i, weight in enumerate(weights):
        for j in h_alist[alist_offset + (n-m) + i]:
            if j == 0:  # values are right padded with zeros
                break
            elif abs(message[j - 1]) < weights[i]:
                weights[i] = abs(message[j - 1])
    return weights


def calculate_weighted_failed_parity_checks(h, s, weights):
    m, n = np.shape(h)
    weighted_check_sums = np.zeros(n)
    for i in range(n):
        for j in range(m):
            weighted_check_sums[i] += (2 * s[j] - 1) * weights[j] * h[j][i]
    return weighted_check_sums


def decode(h, h_alist, message):
    # max_num_of_iteration may be different for every code
    max_num_of_iterations = 100
    message_hd, message_sd = message
    weights = calculate_weights(h, h_alist, message_sd)

    iterations = 0
    while iterations < max_num_of_iterations:
        # calculate syndrome, if it is a zero vector, the message was decoded successfully
        s = (message_hd @ np.transpose(h)) % 2
        if not s.any():
            break

        # compute weighted parity checks and flip the bit with the biggest weight
        weighted_check_sums = calculate_weighted_failed_parity_checks(h, s, weights)
        max_en_index = weighted_check_sums.argmax()
        message_hd[max_en_index] = 1 - message_hd[max_en_index]  # flip bit

        iterations += 1
    return message_hd
