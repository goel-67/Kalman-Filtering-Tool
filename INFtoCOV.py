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


# Test cases
def run_tests():
    domain = 6

    # Input matrix V for Test cases 1 to 5
    # V = np.array([16, 1, 36, 49]).reshape(-1, 1)
    # Uncomment the desired test case
    V = np.array([16, 1, 36, 49, 4, 25]).reshape(-1, 1)
    # V = np.array([16, 1, 62, 55.548, 14.362, 3.5662]).reshape(-1, 1)
    # V = np.array([np.inf, np.inf, 70, 49.271, 0, 0]).reshape(-1, 1)
    # V = np.array([np.inf, np.inf, 73.018, 34.979, 0, 0]).reshape(-1, 1)

    # Input matrix B for Test cases 1 to 5
    '''B = np.array([[0, 0.5, -1.75, -0.125],
                  [0, 0, 5, 0.5],
                  [0, 0, 0, -0.5],
                  [0, 0, 0, 0]]).astype(float)'''
    # Uncomment the desired test case
    B = np.array([[0, 0.5, -1.75, -0.125, 1, 0.5],
                   [0, 0, 5, 0.5, -1, 0.5],
                  [0, 0, 0, -0.5, 1, -0.5],
                  [0, 0, 0, 0, 1, 0.5],
                 [0, 0, 0, 0, 0, 0.5],
                 [0, 0, 0, 0, 0, 0]])
    # B = np.array([[0, 0.5, 0, 1.75, -0.7987805, -0.9363173],
    #               [0, 0, 2, -3.3548387, 2.59581882, 0.8922853],
    #               [0, 0, 0, 0.67741935, 0.65853659, 0.8558952],
    #               [0, 0, 0, 0, -0.543554, 0.0713246],
    #               [0, 0, 0, 0, 0, -0.8922853],
    #               [0, 0, 0, 0, 0, 0]])
    # B = np.array([[0, 0, 0, 0, 0, 0],
    #               [0, 0, 0, 0, 0, 0],
    #               [0, 0, 0, -0.4428571, 1.01826616, 0.18324152],
    #               [0, 0, 0, 0, 1.00898811, 1.0266744],
    #               [0, 0, 0, 0, 0, 0],
    #               [0, 0, 0, 0, 0, 0]])
    # B = np.array([[0, 0, 0, 0, 0, 0],
    #               [0, 0, 0, 0, 0, 0],
    #               [0, 0, 0, 0.55031766, 1.1930593, -0.212938],
    #               [0, 0, 0, 0, -1.1725067, 1.18328841],
    #               [0, 0, 0, 0, 0, 0],
    #               [0, 0, 0, 0, 0, 0]])

    # Function call
    X = inf_to_cov(V, B, domain)

    # Display the result
    np.set_printoptions(precision=5)
    print("Covariance Matrix X:")
    print(X)
    print(B)
    print(V)


if __name__ == "__main__":
    run_tests()