import numpy as np

def reversal(B, V, n0, n1, n2, n3):
    """
    Arc reversal between two nodes using Bayes' rule.

    Parameters:
    B (numpy.ndarray): An n x n strictly upper triangular matrix, composed of strictly upper triangular submatrices.
    V (numpy.ndarray): An n x 1 vector with non-negative (including inf) entries.
    n0 (int): Size of vector node x0.
    n1 (int): Size of vector node x1.
    n2 (int): Size of vector node x2.
    n3 (int): Size of vector node x3.

    Returns:
    B (numpy.ndarray): Updated matrix with reversed arcs.
    V (numpy.ndarray): Updated vector with adjusted variances.
    """

    # Iterate from n0 + n1 down to n0 + 1
    for i in range(n0 + n1, n0, -1):
        for j in range(n0 + n1 + 1, n0 + n1 + n2 + 1):
            if B[i - 1, j - 1] != 0:
                # Update the matrix B by adjusting the appropriate elements
                if n0 >= 1:
                    B[:n0, j - 1] += B[i - 1, j - 1] * B[:n0, i - 1]
                
                if i - 1 > n0:
                    B[n0:i-1, j - 1] += B[i - 1, j - 1] * B[n0:i-1, i - 1]

                if j - 1 > n0 + n1:
                    B[n0 + n1:j - 1, j - 1] += B[i - 1, j - 1] * B[n0 + n1:j - 1, i - 1]
                
                # Update based on the variance matrix V
                if V[i - 1] == 0:
                    B[j - 1, i - 1] = 0
                else:
                    if V[i - 1] != np.inf and V[j - 1] != np.inf:
                        if V[j - 1] == 0:
                            V[j - 1] = B[i - 1, j - 1] ** 2 * V[i - 1]
                            V[i - 1] = 0
                            B[j - 1, i - 1] = 1 / B[i - 1, j - 1]
                        else:
                            Vj_old = V[j - 1]
                            V[j - 1] += B[i - 1, j - 1] ** 2 * V[i - 1]
                            V_ratio = V[i - 1] / V[j - 1]
                            V[i - 1] = Vj_old * V_ratio
                            B[j - 1, i - 1] = B[i - 1, j - 1] * V_ratio
                    else:
                        if V[j - 1] != np.inf:
                            B[j - 1, i - 1] = 1 / B[i - 1, j - 1]
                        else:
                            B[j - 1, i - 1] = 0
                        
                        if V[i - 1] == np.inf and V[j - 1] != np.inf:
                            V[i - 1] = V[j - 1] * B[j - 1, i - 1] ** 2
                        
                        V[j - 1] = np.inf

                # Zero out the current entry
                B[i - 1, j - 1] = 0

                # Further update B based on the reversal process
                if n0 >= 1:
                    B[:n0, i - 1] -= B[j - 1, i - 1] * B[:n0, j - 1]
                
                if i - 1 > n0:
                    B[n0:i-1, i - 1] -= B[j - 1, i - 1] * B[n0:i-1, j - 1]
                
                if j - 1 > n0 + n1:
                    B[n0 + n1:j - 1, i - 1] -= B[j - 1, i - 1] * B[n0 + n1:j - 1, j - 1]

    return B, V

# Test cases for the reversal function
if __name__ == "__main__":
    # Initial setup of inputs
    n0 = 2
    n1 = 2
    n2 = 2
    n3 = 0

    # Vector V Test case
    V = np.array([16.0, 1.0, 36.0, 49.0, 4.0, 25.0], dtype=float)
    #V = np.array([16.0, 1.0, 36.0])

    # Matrix B Test case
    B = np.array([
        [0, 0.5, -1.75, -0.125, 1, 0.5],
        [0, 0, 5, 0.5, -1, 0.5],
        [0, 0, 0, -0.5, 1, -0.5],
        [0, 0, 0, 0, 1, 0.5],
        [0, 0, 0, 0, 0, 0.5],
        [0, 0, 0, 0, 0, 0]
    ])

    '''B = np.array([[0, 0.5, -1.75],
                    [0, 0, 5.0],
                    [0, 0, 0]])'''

    # Function call
    B, V = reversal(B, V, n0, n1, n2, n3)

    # Display results
    print("V:", V)
    print("B:\n", B)