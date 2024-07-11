import numpy as np

# todo test!


def calculate_failed_parity_checks_per_bit(h_alist, n, s):
    f = np.zeros(n)
    for i in range(n):
        mi = h_alist[i + 4]
        for j in mi:
            if j == 0:
                break
            else:
                if s[j - 1] == 1:
                    f[i] += 1
    return f


def decode(h, h_alist, received_msg):

    max_num_of_iteration = 10  # different for every code
    flipping_threshold = 1  # different for every code
    m, n = np.shape(h)

    iteration = 0
    while iteration < max_num_of_iteration:
        # calculate syndrome
        s = (received_msg @ np.transpose(h)) % 2
        # message decoded successfully
        if not s.any():
            break

        #  for every message bit calculate in how many failed parity checks the message bit was involved
        f = calculate_failed_parity_checks_per_bit(h_alist, n, s)

        # flip bits involved in failed parity checks
        for i, fi in enumerate(f):
            # todo maybe flip the bit for the biggest fi?
            if fi >= flipping_threshold:
                received_msg[i] = 1 - received_msg[i]  # flip bit
        iteration += 1
    return received_msg
