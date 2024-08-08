import numpy as np


def init_llr(m, n, message_sd, value_nodes_indices):
    llr = np.zeros((m, n))
    for i, column in enumerate(value_nodes_indices):
        for value in column:
            if value == 0:
                break
            llr[value-1][i] = message_sd[i]
    return llr


# works only for rows with weight >= 2
def find_min(arr, indices_to_check):
    # todo abs
    min_val_index = None
    min_val1 = 1000
    min_val2 = None

    for index in indices_to_check:
        if index == 0:
            break
        if abs(arr[index-1]) < min_val1:
            min_val2 = min_val1
            min_val1 = abs(arr[index-1])
            min_val_index = index
        elif min_val2 is None or (min_val2 > abs(arr[index-1])):
            min_val2 = abs(arr[index-1])

    return min_val1, min_val2, min_val_index


def get_row_sign(row, indices_to_check):
    sign = 1
    for non_zero_value_index in indices_to_check:
        if non_zero_value_index == 0:
            break
        if row[non_zero_value_index - 1] < 0:
            sign *= -1
    return sign


def calculate_sum_vector(llr, column_indices_to_check, message_sd):
    sum_vector = np.copy(message_sd)
    for i, column in enumerate(column_indices_to_check):
        for index in column:
            if index == 0:
                break
            sum_vector[i] += llr[index-1][i]
    return sum_vector


# "vertical step"
def perform_column_operations(sum_vector, column_indices_to_check, llr):
    for i, column in enumerate(column_indices_to_check):
        for index in column:
            if index == 0:
                break
            llr[index-1][i] = sum_vector[i] - llr[index-1][i]


# "horizontal step"
def perform_row_operations(llr, check_nodes_indices):
    for i, row in enumerate(llr):
        min_val1, min_val2, min_index = find_min(row, check_nodes_indices[i])
        row_sign = get_row_sign(row, check_nodes_indices[i])
        for value in check_nodes_indices[i]:
            if value == 0:
                break
            if value == min_index:
                llr[i][value - 1] = min_val2 * row_sign * np.sign(llr[i][value - 1])
            else:
                llr[i][value - 1] = min_val1 * row_sign * np.sign(llr[i][value - 1])


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
        sum_vector = calculate_sum_vector(llr, h_alist[4:4+n], message_sd)
        perform_column_operations(sum_vector, h_alist[4:4+n], llr)
        decoded_message = (sum_vector > 0).astype(int)

        iterations += 1
    return decoded_message
