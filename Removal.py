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
        B, V = reversal(B, V, n0, n1, n2-1, 0)
        print(B)
        print(V)

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
            # Check for infinity before updating V[N-1]
            if V[i - 1] != np.inf and V[N - 1] != np.inf:
                V[N - 1] += B[i - 1, N - 1] * B[i - 1, N - 1] * V[i - 1]
            else:
                V[N - 1] = np.inf

    # Set appropriate entries in V and B to 0 after removing node x1
    V[n0:n0 + n1] = 0
    B[n0:n0 + n1, :] = 0
    B[:, n0:n0 + n1] = 0

    return B, V


# Test cases
def run_tests():
    n0 = 2
    n1 = 2
    n2 = 2
    n3 = 0

    # Vector V Test cases 1 to 4
    # V = np.array([16, 1, 36])
    # V = np.array([16, 1, 36, 49])
    V = np.array([16.0, 1.0, 36.0, 49.0, 4.0, 25.0])
    # V = np.array([np.inf, np.inf, 70, 49.271, 0, 0])

    # Matrix B Test cases 1 to 5
    # B = np.array([[0, 0.5, -1.75],
    #               [0, 0, 5],
    #               [0, 0, 0]])
    # B = np.array([[0, 0.5, -1.75, -0.125],
    #               [0, 0, 5, 0.5],
    #               [0, 0, 0, -0.5],
    #               [0, 0, 0, 0]])
    B = np.array([[0, 0.5, -1.75, -0.125, 1, 0.5],
                  [0, 0, 5, 0.5, -1, 0.5],
                  [0, 0, 0, -0.5, 1, -0.5],
                  [0, 0, 0, 0, 1, 0.5],
                  [0, 0, 0, 0, 0, 0.5],
                  [0, 0, 0, 0, 0, 0]])
    # B = np.array([[0, 0, 0, 0, 0, 0],
    #               [0, 0, 0, 0, 0, 0],
    #               [0, 0, 0, -0.4428571, 1.01826616, 0.18324152],
    #               [0, 0, 0, 0, 1.00898811, 1.0266744],
    #               [0, 0, 0, 0, 0, 0],
    #               [0, 0, 0, 0, 0, 0]])

    # Function call
    B, V = removal(B, V, n0, n1, n2)

    # Display the results
    np.set_printoptions(precision=5)
    print("Updated Matrix B:")
    print(B)
    print("Updated Vector V:")
    print(V)

if __name__ == "__main__":
    run_tests()