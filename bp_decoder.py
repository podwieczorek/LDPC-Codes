import numpy as np


def init_llr(h):
    pass


def perform_column_operations():
    pass


def perform_row_operations():
    pass


def decode(h, h_alist, message):
    max_num_of_iterations = 10
    m, n = np.shape(h)
    message_hd, message_sd = message
    init_llr(h)

    iterations = 0
    while iterations < max_num_of_iterations:
        # calculate syndrome
        s = (message_hd @ np.transpose(h))
        # message decoded successfully
        if not s.any():
            break

        perform_row_operations()
        perform_column_operations()
        iterations += 1
