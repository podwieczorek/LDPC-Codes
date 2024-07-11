import numpy as np

# todo make it a class


def calculate_phi(h, g):
    m = np.shape(h)[0]
    a = h[:m - g, m - g:m]
    c = h[m-g:, m-g:m]
    e = h[m - g:, :m - g]
    t_inv = np.linalg.inv(h[:m-g, :m-g])  # todo in gf2
    eta = e @ (t_inv @ a)
    print(eta)
    return c - eta


def minimum_residual_degree(h, t, g):
    m, n = np.shape(h)
    residual_h = h[t:m - g, t:n]
    column_sums = np.sum(residual_h, axis=0)
    column_sums[column_sums == 0] = np.iinfo(np.int32).max
    min_nonzero_weight = np.min(column_sums)
    columns_with_min_weight = np.where(column_sums == min_nonzero_weight)[0] + t
    return min_nonzero_weight, columns_with_min_weight


# todo, extend() and choose() are almost the same, clean up
def extend(h, column_to_swap, t, g):
    m, n = np.shape(h)
    # swap columns
    h[:, [t, column_to_swap]] = h[:, [column_to_swap, t]]
    # find row with 1 in residual parity check matrix
    sub_array = h[t:m - g, t]
    row_to_swap = np.where(sub_array == 1)[0][0] + t
    # swap rows
    h[[t, row_to_swap], :] = h[[row_to_swap, t], :]


def choose(h, column_to_swap, t, g):
    m, n = np.shape(h)
    # swap columns
    h[:, [t, column_to_swap]] = h[:, [column_to_swap, t]]
    # find rows with 1 in residual parity check matrix
    sub_array = h[t:m-g, t]
    row_indices = np.where(sub_array == 1)[0] + t

    # swap first row
    first_row_to_swap = int(row_indices[0])
    h[[t, first_row_to_swap], :] = h[[first_row_to_swap, t], :]

    # move other rows with 1 at the end of the h matrix
    rows_to_move = []
    for row_index in row_indices[-1:0:-1]:
        if rows_to_move == []:
            rows_to_move = h[row_index]
        else:
            rows_to_move = np.vstack((rows_to_move, h[row_index]))
        h = np.delete(h, row_index, axis=0)

    rows_to_move = np.reshape(rows_to_move, (row_indices.size-1, n))
    np.concatenate((h, rows_to_move), axis=0)

    return row_indices.size - 1


def approximate_upper_triangulation(h):
    m, n = np.shape(h)
    t = 0
    g = 0
    while True:
        if t == m - g:
            return h, g
        # find minimum residual degree and columns with that degree
        min_res_degree, columns = minimum_residual_degree(h, t, g)
        chosen_column_index = np.random.randint(0, columns.size, dtype=int)
        random_column = columns[chosen_column_index]
        if min_res_degree == 1:
            extend(h, random_column, t, g)
        else:
            # todo
            g += choose(h, random_column, t, g)
        t += 1


# input: parity check matrix h
# output: equivalent parity check matrix in approximate upper-triangular form
#         and precalculated phi
def preprocess(h):
    # parity check matrix h in approximate upper-triangular form and gap g
    h_aut, g = approximate_upper_triangulation(h)
    # if gap is 0, calculating phi is not needed
    if g == 0:
        return h, g, []
    else:
        phi = calculate_phi(h_aut, g)
        return h, g, phi


def calculate_p1(h, s, p2, g):
    m, n = np.shape(h)
    a = h[:m-g, m-g:m]
    b = h[:m-g, m:]
    t = h[:m-g, :m-g]
    t_inv = np.linalg.inv(t)
    ap2t = a @ np.transpose(p2)
    bst = b @ np.transpose(s)

    return - t_inv @ (ap2t + bst)


def calculate_p2(h, s, phi, g):
    m, n = np.shape(h)
    d = h[m-g:, m:]
    e = h[m-g:, :m-g]
    b = h[:m - g, m:]
    t = h[:m-g, :m-g]
    phi_inv = np.linalg.inv(phi)  # todo in gf2
    t_inv = np.linalg.inv(t)  # todo in gf2
    dst = d @ np.transpose(s)
    etbst = (e @ (t_inv @  (b @ np.transpose(s))))
    return - phi_inv @ (dst - etbst)


def encode(h, s, phi, g):
    p2 = calculate_p2(h, s, phi, g)
    p1 = calculate_p1(h, s, p2, g)
    return np.concatenate((p1, p2, s), axis=None)
