import torch


def calc_square_dist(a, b, norm=True):
    """
    Calculating square distance between a and b
    a: [bs, n, c]
    b: [bs, m, c]
    """
    n = a.shape[1]
    m = b.shape[1]
    num_channel = a.shape[-1]
    a_square = a.unsqueeze(dim=2)  # [bs, n, 1, c]
    b_square = b.unsqueeze(dim=1)  # [bs, 1, m, c]
    a_square = torch.sum(a_square * a_square, dim=-1)  # [bs, n, 1]
    b_square = torch.sum(b_square * b_square, dim=-1)  # [bs, 1, m]
    a_square = a_square.repeat((1, 1, m))  # [bs, n, m]
    b_square = b_square.repeat((1, n, 1))  # [bs, n, m]

    coor = torch.matmul(a, b.transpose(1, 2))  # [bs, n, m]

    if norm:
        dist = a_square + b_square - 2.0 * coor  # [bs, npoint, ndataset]
        # dist = torch.sqrt(dist)
    else:
        dist = a_square + b_square - 2 * coor
        # dist = torch.sqrt(dist)
    return dist