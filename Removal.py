import numpy as np
from Reversal import reversal

def removal(B, V, n0, n1, n2):
    """
    Removal of vector nodes in Gaussian influence diagram.

    Parameters:
    B (numpy.ndarray): An n x n strictly upper triangular matrix comprised of strictly upper triangular submatrices.
    V (numpy.ndarray): An n x 1 vector with non-negative (including inf) entries.
    n0 (int): The size of vector node x0.
    n1 (int): The size of vector node x1.
    n2 (int): The size of vector node x2.

    Returns:
    B (numpy.ndarray): Updated n x n matrix with removed vector nodes.
    V (numpy.ndarray): Updated n x 1 vector with removed vector nodes.
    """

    # If n2 > 1, reverse arcs from vector node x1 to the first n2-1 elements of vector node x2
    if n2 > 1:
        B, V = reversal(B, V, n0, n1, n2 - 1, 0)

    N = n0 + n1 + n2

    # Iteratively remove elements of the vector nodes
    for i in range(n0 + n1, n0, -1):
        if n0 >= 1:
            B[:n0, N - 1] += B[i - 1, N - 1] * B[:n0, i - 1]

        if i - 1 > n0:
            B[n0:i-1, N - 1] += B[i - 1, N - 1] * B[n0:i-1, i - 1]

        if N - 1 > n0 + n1:
            B[n0 + n1:N - 1, N - 1] += B[i - 1, N - 1] * B[n0 + n1:N - 1, i - 1]

        if V[i - 1] != 0:
            if V[i - 1] != np.inf and V[N - 1] != np.inf:
                V[N - 1] += B[i - 1, N - 1] * B[i - 1, N - 1] * V[i - 1]
            else:
                V[N - 1] = np.inf

    # Set appropriate entries in V and B to 0 after removing node x1
    V[n0:n0 + n1] = 0
    B[n0:n0 + n1, :] = 0
    B[:, n0:n0 + n1] = 0

    return B, V
