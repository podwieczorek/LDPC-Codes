import numpy as np


def calculate_failed_parity_checks_per_bit(h_alist, n, s):
    # todo make it more efficient
    f = np.zeros(n)
    for i in range(n):
        mi = h_alist[i + 4].split()
        for j in mi:
            if j == '0':
                break
            else:
                if s[int(j) - 1] == 1:
                    f[i] += 1
    return f


def decode(h, h_alist, received_msg):

    max_num_of_iteration = 10  # todo how to choose number of iterations?
    flipping_threshold = 1  # todo how to choose the threshold?
    n = np.shape(h)[1]
    k = n - np.shape(h)[0]
    no_errors_s = np.zeros(k)

    iteration = 0
    while iteration < max_num_of_iteration:
        # calculate syndrome
        s = np.dot(received_msg, np.transpose(h)) % 2

        # message decoded successfully
        if s == no_errors_s:
            break

        #  for every message bit calculate in how many failed parity checks the message bit was involved
        f = calculate_failed_parity_checks_per_bit(h_alist, n, s)

        # flip bits in message that were involved in failed parity checks
        for i, fi in enumerate(f):
            if fi >= flipping_threshold:
                received_msg[i] += 1
                received_msg = received_msg % 2

        iteration += 1

    return received_msg
