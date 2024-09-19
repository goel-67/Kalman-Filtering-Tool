import numpy as np

def inf_to_cov(V, B, domain):
    """
    Converts influence diagram form to covariance form.

    Parameters:
    V (numpy.ndarray): An n x 1 vector with non-negative (including inf) entries.
    B (numpy.ndarray): An n x n matrix that is strictly upper triangular.
    domain (int): The number of rows and columns of B.

    Returns:
    X (numpy.ndarray): The covariance matrix of the multivariate Gaussian distribution.
    """

    # Initialize the covariance matrix X
    X = np.zeros((domain, domain))
    
    # First element in the diagonal
    X[0, 0] = V[0]

    for i in range(1, domain):
        for j in range(i):
            X[i, j] = 0
            for k in range(i):
                if X[j, k] != np.inf:
                    X[i, j] += X[j, k] * B[k, i]
            X[j, i] = X[i, j]  # Since the matrix is symmetric

        # Update diagonal elements
        if V[i] == np.inf:
            X[i, i] = np.inf
        else:
            Y = X[i, :i]
            Z = B[:i, i]
            X[i, i] = V[i] + np.dot(Y, Z)

    return X
