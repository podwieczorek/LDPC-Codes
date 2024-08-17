"""
Min-sum soft-decision decoder - simplified belief-propagation (BP) decoder
Instead of using the full sum-product calculation, it computes the minimum of incoming messages
It reduces computational complexity of BP decoder but also slightly reduces decoding accuracy.
"""


import numpy as np


def init_llr(m, n, message_sd, value_nodes_indices):
    # we use one matrix for storing "messages" from variable nodes to check nodes (calculated in horizontal step)
    # and from check nodes to variable nodes (calculated in vertical step), the messages are overwritten in each step
    llr = np.zeros((m, n))
    for i, column in enumerate(value_nodes_indices):
        for value in column:
            if value == 0:  # values are right padded with zeros
                break
            # log likelihood ratios are initialized by soft decision values received from channel
            llr[value-1][i] = message_sd[i]
    return llr


# find minimum value, its index, and next minimum value in the row (checking only given indices)
# todo works only for rows with more than one non-zero elements
def find_min(row, indices_to_check):
    min_val_index = None
    min_val1 = 1000
    min_val2 = None

    for index in indices_to_check:
        if index == 0:  # values are right padded with zeros
            break

        value_at_index = abs(row[index-1])
        if value_at_index < min_val1:
            min_val2 = min_val1
            min_val1 = value_at_index
            min_val_index = index
        elif min_val2 is None or (min_val2 > value_at_index):
            min_val2 = value_at_index

    return min_val1, min_val2, min_val_index


def get_row_sign(row, indices_to_check):
    sign = 1
    for non_zero_value_index in indices_to_check:
        if non_zero_value_index == 0:  # values are right padded with zeros
            break
        if row[non_zero_value_index - 1] < 0:
            sign *= -1
    return sign


def calculate_sum_vector(llr, column_indices_to_check, message_sd):
    sum_vector = np.copy(message_sd)
    for i, column in enumerate(column_indices_to_check):
        for index in column:
            if index == 0:  # values are right padded with zeros
                break
            sum_vector[i] += llr[index-1][i]
    return sum_vector


# column operation, also known as "vertical step", process messages from check nodes
def perform_column_operations(sum_vector, column_indices_to_check, llr):
    for i, column in enumerate(column_indices_to_check):
        for index in column:
            if index == 0:   # values are right padded with zeros
                break
            llr[index-1][i] = sum_vector[i] - llr[index-1][i]


# row operation, also known as "horizontal step", process messages from variable nodes
def perform_row_operations(llr, variable_nodes_indices):
    for row_index, row in enumerate(llr):
        # find minimum value, its index, and next minimum value in the row
        min_val1, min_val2, min_index = find_min(row, variable_nodes_indices[row_index])
        row_sign = get_row_sign(row, variable_nodes_indices[row_index])

        for value in variable_nodes_indices[row_index]:
            if value == 0:   # values are right padded with zeros
                break

            variable_sign = np.sign(llr[row_index][value - 1])
            if value == min_index:
                llr[row_index][value - 1] = min_val2 * row_sign * variable_sign
            else:
                llr[row_index][value - 1] = min_val1 * row_sign * variable_sign


def decode(h, h_alist, message):
    # max_num_of_iteration may be different for every code
    max_num_of_iterations = 100
    m, n = np.shape(h)
    message_hd, message_sd = message
    llr = init_llr(m, n, message_sd, h_alist[4:4+n])

    iterations = 0
    decoded_message = np.copy(message_hd)
    while iterations < max_num_of_iterations:
        # calculate syndrome, if it is a zero vector, the message was decoded successfully
        s = (decoded_message @ np.transpose(h)) % 2
        if not s.any():
            break

        perform_row_operations(llr, h_alist[4+n: 4+n+m])
        # vertical step was divided into two operations in order to simplify calculations
        sum_vector = calculate_sum_vector(llr, h_alist[4:4+n], message_sd)
        perform_column_operations(sum_vector, h_alist[4:4+n], llr)
        decoded_message = (sum_vector > 0).astype(int)

        iterations += 1
    return decoded_message
