import numpy as np

def cov_to_inf(X, domain):
    """
    Converts covariance form to influence diagram form

    Parameters:
    X (numpy.ndarray): Gaussian distribution (covariance matrix)
    domain (int): the number of rows and columns of X (assuming X is square)

    Returns:
    B (numpy.ndarray): An n x n matrix of Gaussian influence diagram arc coefficients which is strictly upper triangular
    V (numpy.ndarray): An n x 1 matrix of Gaussian influence diagram conditional variances with non-negative entries (including inf)
    P (numpy.ndarray): The (Moore-Penrose generalized) inverse of X
    """
    
    P = np.zeros((domain, domain))
    V = np.zeros(domain)
    B = np.zeros((domain, domain))

    # Initialize first row
    B[0, 0] = 0
    V[0] = X[0, 0]

    if V[0] == 0 or V[0] == np.inf:
        P[0, 0] = 0
    else:
        P[0, 0] = 1 / V[0]

    # Iterate through the domain
    for j in range(1, domain):
        B[j, j] = 0
        B[j, :j] = 0
        B[:j, j] = P[:j, :j].dot(X[:j, j])

        if X[j, j] == np.inf:
            V[j] = np.inf
        else:
            V[j] = X[j, j] - np.dot(X[j, :j], B[:j, j])
            V[j] = max(V[j], 0)

        if V[j] == 0 or V[j] == np.inf:
            P[j, j] = 0
            P[j, :j] = 0
            P[:j, j] = 0
        else:
            P[j, j] = 1 / V[j]
            for k in range(j):
                temp = P[j, j] * B[k, j]
                if k != 0:
                    P[k, :k] += temp * B[:k, j]
                    P[:k, k] = P[k, :k]
                P[k, k] += temp * B[k, j]
            P[j, :j] = -P[j, j] * B[:j, j]
            P[:j, j] = P[j, :j]

    return B, V.reshape(-1, 1), P


domain = 2  # domain can be 6 or 4, meaning 6x6 matrix or 4x4 matrix

# Input matrix sigma Test cases
# Uncomment the test cases you want to use

# Test Case 1
# X = np.array([[16, 8, 12, -4],
#               [8, 5, 11, -4],
#               [12, 11, 70, -31],
#               [-4, -4, -31, 63]])

# Test Case 2
# X = np.array([[16, 8, 12, -4, 16, 12],
#               [8, 5, 11, -4, 10, 4],
#               [12, 11, 70, -31, 40, -19],
#               [-4, -4, -31, 63, 32, 59],
#               [16, 10, 40, 32, 82, 50],
#               [12, 4, -19, 59, 50, 97]])

# Test Case 3
# X = np.array([[16, 8, 16, 12, 12, -4],
#               [8, 5, 10, 4, 11, -4],
#               [16, 10, 82, 50, 40, 32],
#               [12, 4, 50, 97, -19, 59],
#               [12, 11, 40, -19, 70, -31],
#               [-4, -4, 32, 59, -31, 63]])

# Test Case 4
# X = np.array([[np.inf, 0, 0, 0, 0, 0],
#               [0, np.inf, 0, 0, 0, 0],
#               [0, 0, 70, -31, 40, -19],
#               [0, 0, -31, 63, 32, 59],
#               [0, 0, 40, 32, 73.0182661641055, 40.1832415192808],
#               [0, 0, -19, 59, 40.1832415192808, 57.0922006378658]])

# Test Case 5
# X = np.array([[np.inf, 0, 0, 0, 0, 0],
#               [0, np.inf, 0, 0, 0, 0],
#               [0, 0, 73.0182661641055, 40.1832415192808, 40, 32],
#               [0, 0, 40.1832415192808, 57.0922006378658, -19, 59],
#               [0, 0, 40, -19, 70, -31],
#               [0, 0, 32, 59, -31, 63]])

# Test Case 6
X = np.array([[0.00516961, 0.008445032],
              [0.008445032, 2.02692169]])

# Function call
B, V, P = cov_to_inf(X, domain)

# Display the results with high precision
np.set_printoptions(precision=10)
print("B:", B)
print("V:", V)
print("P:", P)