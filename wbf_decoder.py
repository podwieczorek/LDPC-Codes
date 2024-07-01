import numpy as np

# todo test


def calculate_weights(h_alist, message, m):
    weights = np.full(m, 10000.0)
    k = h_alist[0][0] - h_alist[0][1]
    # for every check bit find its message bit with the smallest magnitude
    for i, weight in enumerate(weights):
        variable_nodes = h_alist[4 + k + i]
        for j in variable_nodes:
            if j == 0:
                break
            elif abs(message[j - 1]) < weights[i]:
                weights[i] = abs(message[j - 1])
    return weights


def decode(h, h_alist, message):
    max_num_of_iterations = 10
    m, n = np.shape(h)
    message_hd, message_sd = message
    weights = calculate_weights(h_alist, message_sd, m)

    iterations = 0
    while iterations < max_num_of_iterations:
        # calculate syndrome
        s = (message_hd @ np.transpose(h))
        # message decoded successfully
        if not s.any():
            break

        en = np.zeros(n)
        for i in range(n):
            for j in range(m):
                parity_check_index = h_alist[i + 4][j] - 1
                if parity_check_index == -1:
                    break
                else:
                    en[i] -= weights[parity_check_index] * (1 - 2 * s[parity_check_index])

        # find the biggest element in en and flip the bit
        max_en_index = en.argmax()
        message_hd[max_en_index] = 1 - message_hd[max_en_index]  # flip bit
        iterations += 1
    return message_hd